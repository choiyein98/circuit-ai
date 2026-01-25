import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import math
from PIL import Image
from collections import defaultdict
import gc

# ==========================================
# [설정] BrainBoard V67: Robust Hybrid
# ==========================================
st.set_page_config(page_title="BrainBoard V67: Hybrid", layout="wide")

REAL_MODEL_PATH = 'best(3).pt' 
MODEL_SYM_PATH = 'symbol.pt'

# ==========================================
# [Helper Functions]
# ==========================================
def resize_image_smart(image, max_size=1024):
    h, w = image.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image

def get_center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = ((box1[2]-box1[0])*(box1[3]-box1[1])) + ((box2[2]-box2[0])*(box2[3]-box2[1])) - inter
    return inter / union if union > 0 else 0

def normalize_name(name):
    name = name.lower()
    if 'res' in name: return 'resistor'
    if 'cap' in name: return 'capacitor'
    if 'wire' in name: return 'wire'
    if any(x in name for x in ['source', 'batt', 'volt', 'vdc']): return 'source'
    if any(x in name for x in ['leg', 'pin', 'lead']): return 'leg'
    return name

def solve_overlap_real(parts):
    if not parts: return []
    # 신뢰도 순으로 정렬
    parts.sort(key=lambda x: x.get('conf', 0), reverse=True)
    final = []
    for curr in parts:
        is_dup = False
        for k in final:
            iou = calculate_iou(curr['box'], k['box'])
            dist = math.sqrt((curr['center'][0]-k['center'][0])**2 + (curr['center'][1]-k['center'][1])**2)
            # 겹치거나 너무 가까우면 중복 제거 (거리 기준 60px)
            if curr['name'] != 'leg' and (iou > 0.4 or dist < 60): 
                is_dup = True; break
        if not is_dup: final.append(curr)
    return final

# ==========================================
# [Logic] 위치 기반 순서 추출 (Spatial Sort)
# ==========================================
def extract_spatial_sequence(parts, image_width):
    # 1. X좌표 기준 정렬
    sorted_parts = sorted(parts, key=lambda x: x['center'][0])
    
    sequence = []
    current_stage = []
    
    if not sorted_parts: return []

    current_stage.append(sorted_parts[0])
    last_x = sorted_parts[0]['center'][0]
    
    # 2. 그룹화 (이미지 너비의 15% 이내면 같은 단계로 간주)
    threshold = image_width * 0.15 
    
    for i in range(1, len(sorted_parts)):
        curr = sorted_parts[i]
        curr_x = curr['center'][0]
        
        if abs(curr_x - last_x) < threshold:
            current_stage.append(curr)
        else:
            # Y좌표 정렬 (위->아래)
            current_stage.sort(key=lambda x: x['center'][1])
            sequence.append(current_stage)
            current_stage = [curr]
            last_x = curr_x
            
    if current_stage:
        current_stage.sort(key=lambda x: x['center'][1])
        sequence.append(current_stage)
        
    return sequence

def format_sequence(seq):
    formatted = []
    for stage in seq:
        names = [p['name'] for p in stage]
        if len(names) > 1:
            formatted.append(f"[{' & '.join(names)}]")
        else:
            formatted.append(names[0])
    return " → ".join(formatted)

# ==========================================
# [Analysis 1] Schematic
# ==========================================
def analyze_schematic(img, model):
    img = resize_image_smart(img)
    w = img.shape[1]
    
    results = model.predict(source=img, save=False, conf=0.05, verbose=False)
    raw_parts = []
    
    for box in results[0].boxes:
        raw_name = model.names[int(box.cls[0])]
        norm_name = normalize_name(raw_name)
        # 위치 비교용이므로 와이어 제외
        if norm_name == 'wire' or norm_name == 'leg': continue
        
        coords = box.xyxy[0].tolist()
        raw_parts.append({'name': norm_name, 'box': coords, 'center': get_center(coords), 'conf': float(box.conf[0])})

    parts = []
    raw_parts.sort(key=lambda x: x['conf'], reverse=True)
    for p in raw_parts:
        if not any(calculate_iou(p['box'], k['box']) > 0.1 for k in parts): parts.append(p)

    if parts and not any(p['name'] == 'source' for p in parts):
         leftmost = min(parts, key=lambda p: p['center'][0])
         leftmost['name'] = 'source'

    # 시각화
    for p in parts:
        x1, y1, x2, y2 = map(int, p['box'])
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img, p['name'], (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    sequence = extract_spatial_sequence(parts, w)
    return img, {'parts': parts, 'sequence': sequence, 'seq_str': format_sequence(sequence)}

# ==========================================
# [Analysis 2] Real Board (복구된 인식 로직)
# ==========================================
def analyze_real(img, model):
    img = resize_image_smart(img)
    h, w, _ = img.shape
    
    # 1. 인식 (Threshold 튜닝)
    res = model.predict(source=img, conf=0.10, verbose=False)
    raw_objects = []
    
    for b in res[0].boxes:
        raw_name = model.names[int(b.cls[0])]
        norm_name = normalize_name(raw_name)
        conf = float(b.conf[0])
        
        # [수정] 커패시터 중복 방지를 위해 임계값 살짝 상향
        if norm_name == 'capacitor' and conf < 0.20: continue 
        if norm_name == 'resistor' and conf < 0.25: continue
        if 'breadboard' in raw_name: continue
        
        coords = b.xyxy[0].tolist()
        raw_objects.append({'name': norm_name, 'box': coords, 'center': get_center(coords), 'conf': conf})

    # 2. 부품 분리 및 중복 제거
    parts_candidates = [p for p in raw_objects if p['name'] != 'leg']
    legs = [p for p in raw_objects if p['name'] == 'leg']
    
    parts = solve_overlap_real(parts_candidates) # 여기서 겹친 Capacitor 제거됨

    # 3. [복구됨] Source 유무 판단 (와이어 위치 기반)
    TOP_RAIL = h * 0.20; BOTTOM_RAIL = h * 0.80
    has_source = False
    
    if any(p['name'] == 'source' for p in parts): has_source = True
    
    if not has_source:
        # 와이어나 핀이 전원 레일에 있으면 Source가 있다고 판단!
        for p in raw_objects: 
            if p['center'][1] < TOP_RAIL or p['center'][1] > BOTTOM_RAIL:
                if p['name'] == 'wire' or p['name'] == 'leg':
                    has_source = True; break
    
    # Source 가상 부품 추가
    if has_source and not any(p['name'] == 'source' for p in parts):
        parts.append({'name': 'source', 'box': [0,0,0,0], 'center': (0,0), 'conf': 1.0})

    # 4. 시각화 (Source는 박스 그리지 않고 텍스트로만 표시하거나, 0,0 박스라 안 그려짐)
    for p in parts:
        if p['name'] == 'wire': continue # 와이어는 화면에서 숨김
        
        color = (0, 255, 0)
        if p['name'] == 'source': color = (0, 255, 255)
        
        if p['box'][2] > 0: # 실제 박스가 있는 부품만 그리기
            x1, y1, x2, y2 = map(int, p['box'])
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            cv2.putText(img, p['name'].upper(), (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        elif p['name'] == 'source':
            # 가상 Source는 화면 좌상단에 표시
            cv2.putText(img, "POWER DETECTED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # 5. 위치 기반 순서 추출 (와이어 제외하고 부품만)
    main_parts = [p for p in parts if p['name'] != 'wire']
    sequence = extract_spatial_sequence(main_parts, w)
    
    return img, {'parts': parts, 'sequence': sequence, 'seq_str': format_sequence(sequence)}

# ==========================================
# [Main UI]
# ==========================================
st.title("🧠 BrainBoard V67: Robust Hybrid")
st.markdown("### 📍 인식률 복구 + 직관적 위치 비교")

@st.cache_resource
def load_models():
    gc.collect()
    return YOLO(REAL_MODEL_PATH), YOLO(MODEL_SYM_PATH)

try:
    model_real, model_sym = load_models()
    st.sidebar.success("✅ 시스템 준비 완료")
except: st.stop()

col1, col2 = st.columns(2)
ref_file = col1.file_uploader("1. 회로도", type=['jpg', 'png', 'jpeg'])
tgt_file = col2.file_uploader("2. 실물 사진", type=['jpg', 'png', 'jpeg'])

if ref_file and tgt_file:
    ref_image = Image.open(ref_file)
    tgt_image = Image.open(tgt_file)
    ref_cv = cv2.cvtColor(np.array(ref_image), cv2.COLOR_RGB2BGR)
    tgt_cv = cv2.cvtColor(np.array(tgt_image), cv2.COLOR_RGB2BGR)

    if st.button("🚀 분석 실행"):
        gc.collect()
        with st.spinner("부품 인식 및 배치 분석 중..."):
            
            res_ref_img, ref_data = analyze_schematic(ref_cv.copy(), model_sym)
            res_tgt_img, tgt_data = analyze_real(tgt_cv.copy(), model_real)

            # 1. BOM Check
            st.subheader("1. 부품 개수 확인")
            ref_counts = defaultdict(int)
            tgt_counts = defaultdict(int)
            for p in ref_data['parts']: ref_counts[p['name']] += 1
            for p in tgt_data['parts']: tgt_counts[p['name']] += 1
            
            # wire는 개수 비교에서 제외
            all_keys = set(ref_counts.keys()) | set(tgt_counts.keys()) - {'wire'}
            
            bom_match = True
            bom_data = []
            for k in all_keys:
                r = ref_counts[k]; t = tgt_counts[k]
                status = "✅ 일치" if r == t else "❌ 불일치"
                bom_data.append({"부품명": k.upper(), "회로도": r, "실물": t, "상태": status})
                if r != t: bom_match = False
            st.table(bom_data)

            # 2. Sequence Check
            st.subheader("2. 배치 순서 비교 (Left -> Right)")
            
            st.info(f"📜 **회로도 순서:** {ref_data['seq_str']}")
            st.info(f"📸 **실물 배치:** {tgt_data['seq_str']}")
            
            # 단순 문자열 비교 대신 단계별 비교
            ref_seq = ref_data['sequence']
            tgt_seq = tgt_data['sequence']
            
            is_seq_match = True
            
            # 단계 수가 다르면 길이 비교
            if len(ref_seq) != len(tgt_seq):
                 st.warning("⚠️ 배치 단계(Column) 수가 다릅니다. (회로도와 실물의 간격 차이일 수 있습니다)")
            
            # 가능한 범위 내에서 비교
            min_len = min(len(ref_seq), len(tgt_seq))
            for i in range(min_len):
                r_names = sorted([p['name'] for p in ref_seq[i]])
                t_names = sorted([p['name'] for p in tgt_seq[i]])
                
                if r_names == t_names:
                    st.success(f"✅ Step {i+1}: {r_names} - 일치")
                else:
                    st.error(f"❌ Step {i+1}: 불일치 (회로도:{r_names} vs 실물:{t_names})")
                    is_seq_match = False

            if is_seq_match and bom_match and (len(ref_seq) == len(tgt_seq)):
                st.success("🎉 **완벽합니다! 부품 구성과 배치 순서가 일치합니다.**")
                st.balloons()
            
            st.image(cv2.cvtColor(res_ref_img, cv2.COLOR_BGR2RGB), caption="회로도 분석", use_column_width=True)
            st.image(cv2.cvtColor(res_tgt_img, cv2.COLOR_BGR2RGB), caption="실물 분석 (인식 복구됨)", use_column_width=True)
            
            del res_ref_img, res_tgt_img
            gc.collect()
