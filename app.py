import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import math
from PIL import Image
from collections import defaultdict
import gc

# ==========================================
# [설정] BrainBoard V66: Spatial Matching
# ==========================================
st.set_page_config(page_title="BrainBoard V66: Spatial", layout="wide")

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
    if 'wire' in name: return 'wire' # 와이어는 무시할 예정
    if any(x in name for x in ['source', 'batt', 'volt', 'vdc']): return 'source'
    return name

def solve_overlap(parts):
    if not parts: return []
    parts.sort(key=lambda x: x.get('conf', 0), reverse=True)
    final = []
    for curr in parts:
        is_dup = False
        for k in final:
            iou = calculate_iou(curr['box'], k['box'])
            if iou > 0.4: is_dup = True; break
        if not is_dup: final.append(curr)
    return final

# ==========================================
# [Core Logic] 위치 기반 시퀀스 추출
# ==========================================
def extract_spatial_sequence(parts, image_width):
    """
    부품을 왼쪽에서 오른쪽으로 정렬하고, 
    X좌표가 비슷하면 같은 '단계(Stage)'로 묶습니다.
    """
    # 1. X좌표 기준으로 정렬
    sorted_parts = sorted(parts, key=lambda x: x['center'][0])
    
    sequence = []
    current_stage = []
    
    if not sorted_parts: return []

    # 첫 번째 부품
    current_stage.append(sorted_parts[0])
    last_x = sorted_parts[0]['center'][0]
    
    # 2. 그룹화 (X좌표 차이가 이미지 너비의 15% 이내면 같은 그룹)
    threshold = image_width * 0.15 
    
    for i in range(1, len(sorted_parts)):
        curr = sorted_parts[i]
        curr_x = curr['center'][0]
        
        if abs(curr_x - last_x) < threshold:
            # 같은 그룹 (예: 병렬 배치)
            current_stage.append(curr)
        else:
            # 새로운 그룹 (다음 단계)
            # 현재 그룹 내에서는 Y좌표(위->아래)로 정렬
            current_stage.sort(key=lambda x: x['center'][1])
            sequence.append(current_stage)
            
            # 초기화
            current_stage = [curr]
            last_x = curr_x
            
    # 마지막 그룹 추가
    if current_stage:
        current_stage.sort(key=lambda x: x['center'][1])
        sequence.append(current_stage)
        
    return sequence

def format_sequence(seq):
    """사람이 보기 좋은 텍스트로 변환"""
    formatted = []
    for stage in seq:
        names = [p['name'] for p in stage]
        if len(names) > 1:
            formatted.append(f"[{' & '.join(names)}]") # 병렬/같은위치
        else:
            formatted.append(names[0])
    return " → ".join(formatted)

# ==========================================
# [Analysis 1] Schematic
# ==========================================
def analyze_schematic(img, model):
    img = resize_image_smart(img)
    h, w, _ = img.shape
    results = model.predict(source=img, save=False, conf=0.05, verbose=False)
    
    raw_parts = []
    for box in results[0].boxes:
        raw_name = model.names[int(box.cls[0])]
        norm_name = normalize_name(raw_name)
        if norm_name == 'wire' or norm_name == 'leg': continue # 위치 비교에선 제외
        
        coords = box.xyxy[0].tolist()
        raw_parts.append({'name': norm_name, 'box': coords, 'center': get_center(coords), 'conf': float(box.conf[0])})

    parts = solve_overlap(raw_parts)

    # 전원 보정 (없으면 제일 왼쪽)
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
# [Analysis 2] Real Board
# ==========================================
def analyze_real(img, model):
    img = resize_image_smart(img)
    h, w, _ = img.shape
    
    res = model.predict(source=img, conf=0.10, verbose=False)
    raw_parts = []
    
    for b in res[0].boxes:
        raw_name = model.names[int(b.cls[0])]
        norm_name = normalize_name(raw_name)
        
        # 위치 대조 방식이므로 wire, leg 등은 무시하고 주요 부품만 봅니다
        if norm_name == 'wire' or norm_name == 'leg': continue
        if 'breadboard' in raw_name: continue
        
        conf = float(b.conf[0])
        if norm_name == 'resistor' and conf < 0.25: continue
        if norm_name == 'capacitor' and conf < 0.15: continue
        
        coords = b.xyxy[0].tolist()
        raw_parts.append({'name': norm_name, 'box': coords, 'center': get_center(coords), 'conf': conf})

    parts = solve_overlap(raw_parts)

    # 전원(Source)이 없으면 와이어 위치로 추정하지 않고, 
    # 그냥 없으면 없는대로 둡니다 (위치 대조니까 정확한 부품 인식이 중요)
    
    # 시각화
    for p in parts:
        color = (0, 255, 0)
        if p['name'] == 'source': color = (0, 255, 255)
        x1, y1, x2, y2 = map(int, p['box'])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.putText(img, p['name'].upper(), (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    sequence = extract_spatial_sequence(parts, w)
    return img, {'parts': parts, 'sequence': sequence, 'seq_str': format_sequence(sequence)}

# ==========================================
# [Main UI]
# ==========================================
st.title("🧠 BrainBoard V66: Spatial Matcher")
st.markdown("### 📍 부품 위치 및 배치 순서 비교 (Simple & Robust)")

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

    if st.button("🚀 위치/순서 비교 실행"):
        gc.collect()
        with st.spinner("부품의 배치를 분석 중입니다..."):
            
            res_ref_img, ref_data = analyze_schematic(ref_cv.copy(), model_sym)
            res_tgt_img, tgt_data = analyze_real(tgt_cv.copy(), model_real)

            # 1. BOM Check
            st.subheader("1. 부품 개수 확인")
            ref_counts = defaultdict(int)
            tgt_counts = defaultdict(int)
            for p in ref_data['parts']: ref_counts[p['name']] += 1
            for p in tgt_data['parts']: tgt_counts[p['name']] += 1
            
            all_keys = set(ref_counts.keys()) | set(tgt_counts.keys())
            bom_match = True
            bom_data = []
            for k in all_keys:
                r = ref_counts[k]; t = tgt_counts[k]
                status = "✅ 일치" if r == t else "❌ 불일치"
                bom_data.append({"부품명": k.upper(), "회로도": r, "실물": t, "상태": status})
                if r != t: bom_match = False
            st.table(bom_data)

            # 2. Sequence Check (핵심)
            st.subheader("2. 배치 순서 비교 (Left -> Right)")
            
            ref_seq_str = ref_data['seq_str']
            tgt_seq_str = tgt_data['seq_str']
            
            st.info(f"📜 **회로도 순서:** {ref_seq_str}")
            st.info(f"📸 **실물 배치:** {tgt_seq_str}")
            
            # 단순 문자열 비교가 아니라 단계별 구성요소 비교
            ref_seq = ref_data['sequence']
            tgt_seq = tgt_data['sequence']
            
            is_seq_match = True
            
            if len(ref_seq) != len(tgt_seq):
                st.error("⚠️ **배치 단계(Column) 수가 다릅니다.** 부품이 너무 몰려있거나 퍼져있지 않은지 확인하세요.")
                is_seq_match = False
            else:
                for i, (r_stage, t_stage) in enumerate(zip(ref_seq, tgt_seq)):
                    r_names = sorted([p['name'] for p in r_stage])
                    t_names = sorted([p['name'] for p in t_stage])
                    
                    if r_names == t_names:
                        st.success(f"✅ Step {i+1}: {r_names} 배치 일치")
                    else:
                        st.error(f"❌ Step {i+1}: 불일치 (회로도:{r_names} vs 실물:{t_names})")
                        is_seq_match = False

            if is_seq_match and bom_match:
                st.success("🎉 **완벽합니다! 부품의 종류, 개수, 배치 순서가 모두 일치합니다.**")
                st.balloons()
            
            st.image(cv2.cvtColor(res_ref_img, cv2.COLOR_BGR2RGB), caption="회로도 배치 분석", use_column_width=True)
            st.image(cv2.cvtColor(res_tgt_img, cv2.COLOR_BGR2RGB), caption="실물 배치 분석 (와이어 무시)", use_column_width=True)
            
            del res_ref_img, res_tgt_img
            gc.collect()
