import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import math
from PIL import Image
from collections import defaultdict
import gc

# ==========================================
# [설정] BrainBoard V69: The Final Perfected
# ==========================================
st.set_page_config(page_title="BrainBoard V69: Final", layout="wide")

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
    parts.sort(key=lambda x: x.get('conf', 0), reverse=True)
    final = []
    for curr in parts:
        is_dup = False
        for k in final:
            iou = calculate_iou(curr['box'], k['box'])
            dist = math.sqrt((curr['center'][0]-k['center'][0])**2 + (curr['center'][1]-k['center'][1])**2)
            if curr['name'] != 'leg' and (iou > 0.4 or dist < 60): 
                is_dup = True; break
        if not is_dup: final.append(curr)
    return final

# ==========================================
# [Logic] 엄격한 순서 정렬 (좌->우, 상->하)
# ==========================================
def sort_parts_LRTB(parts, image_width):
    if not parts: return []
    
    # 1. X축 정렬
    parts.sort(key=lambda x: x['center'][0])
    
    sorted_sequence = []
    current_column = []
    
    # 같은 세로줄로 묶는 기준 (너비의 10%)
    X_THRESHOLD = image_width * 0.10
    
    current_column.append(parts[0])
    ref_x = parts[0]['center'][0]
    
    for i in range(1, len(parts)):
        curr = parts[i]
        curr_x = curr['center'][0]
        
        if abs(curr_x - ref_x) < X_THRESHOLD:
            current_column.append(curr)
        else:
            # 컬럼 내에서는 Y축(위->아래) 정렬
            current_column.sort(key=lambda x: x['center'][1])
            sorted_sequence.extend(current_column)
            
            current_column = [curr]
            ref_x = curr_x
            
    if current_column:
        current_column.sort(key=lambda x: x['center'][1])
        sorted_sequence.extend(current_column)
        
    return sorted_sequence

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

    # 시각화 및 정렬
    for p in parts:
        x1, y1, x2, y2 = map(int, p['box'])
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img, p['name'], (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    sorted_parts = sort_parts_LRTB(parts, w)
    
    # 번호 표시
    for i, p in enumerate(sorted_parts):
        cx, cy = map(int, p['center'])
        cv2.circle(img, (cx, cy), 15, (0, 0, 255), -1)
        cv2.putText(img, str(i+1), (cx-5, cy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return img, {'parts': sorted_parts}

# ==========================================
# [Analysis 2] Real Board (인식 기능 완벽 유지)
# ==========================================
def analyze_real(img, model):
    img = resize_image_smart(img)
    h, w, _ = img.shape
    
    # 1. 강력한 인식 (Threshold 유지)
    res = model.predict(source=img, conf=0.10, verbose=False)
    raw_objects = []
    
    for b in res[0].boxes:
        raw_name = model.names[int(b.cls[0])]
        norm_name = normalize_name(raw_name)
        conf = float(b.conf[0])
        
        # [중요] V67의 필터링 로직 그대로 유지
        if norm_name == 'capacitor' and conf < 0.20: continue
        if norm_name == 'resistor' and conf < 0.25: continue
        if 'breadboard' in raw_name: continue
        
        coords = b.xyxy[0].tolist()
        raw_objects.append({'name': norm_name, 'box': coords, 'center': get_center(coords), 'conf': conf})

    parts_candidates = [p for p in raw_objects if p['name'] != 'leg']
    legs = [p for p in raw_objects if p['name'] == 'leg']
    parts = solve_overlap_real(parts_candidates)

    # 2. Source 복구 로직 (그대로 유지)
    TOP_RAIL = h * 0.20; BOTTOM_RAIL = h * 0.80
    has_source = False
    
    if any(p['name'] == 'source' for p in parts): has_source = True
    if not has_source:
        for p in raw_objects:
            if p['center'][1] < TOP_RAIL or p['center'][1] > BOTTOM_RAIL:
                if p['name'] == 'wire' or p['name'] == 'leg':
                    has_source = True; break
    
    if has_source and not any(p['name'] == 'source' for p in parts):
        parts.append({'name': 'source', 'box': [0,0,0,0], 'center': (0,0), 'conf': 1.0})

    # 3. 시각화
    for p in parts:
        if p['name'] == 'wire': continue
        color = (0, 255, 0)
        if p['name'] == 'source': color = (0, 255, 255)
        
        if p['box'][2] > 0:
            x1, y1, x2, y2 = map(int, p['box'])
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            cv2.putText(img, p['name'].upper(), (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        elif p['name'] == 'source':
            cv2.putText(img, "SOURCE DETECTED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # 4. 정렬 (와이어 제외하고 부품만 줄세우기)
    main_parts = [p for p in parts if p['name'] != 'wire']
    sorted_parts = sort_parts_LRTB(main_parts, w)

    for i, p in enumerate(sorted_parts):
        if p['box'][2] > 0:
            cx, cy = map(int, p['center'])
            cv2.circle(img, (cx, cy), 15, (0, 0, 255), -1)
            cv2.putText(img, str(i+1), (cx-5, cy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return img, {'parts': sorted_parts}

# ==========================================
# [Main UI]
# ==========================================
st.title("🧠 BrainBoard V69: Perfected System")
st.markdown("### ⚡ 부품 인식(완벽) + 순서 비교(정밀)")

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
        with st.spinner("모든 부품을 인식하고 순서를 비교합니다..."):
            
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
                if k == 'wire': continue
                r = ref_counts[k]; t = tgt_counts[k]
                status = "✅ 일치" if r == t else "❌ 불일치"
                bom_data.append({"부품명": k.upper(), "회로도": r, "실물": t, "상태": status})
                if r != t: bom_match = False
            st.table(bom_data)

            # 2. Strict Sequence Check
            st.subheader("2. 배치 순서 비교 (Left→Right & Top→Bottom)")
            
            ref_list = [p['name'] for p in ref_data['parts']]
            tgt_list = [p['name'] for p in tgt_data['parts']]
            
            st.code(f"📜 회로도: {' → '.join(ref_list)}")
            st.code(f"📸 실물:   {' → '.join(tgt_list)}")
            
            if not bom_match:
                 st.warning("⚠️ 부품 개수가 달라서 순서를 1:1로 비교할 수 없습니다. 개수를 먼저 맞춰주세요.")
            else:
                is_seq_match = True
                for i in range(len(ref_list)):
                    r_item = ref_list[i]
                    t_item = tgt_list[i]
                    if r_item == t_item:
                        st.success(f"✅ {i+1}번 부품: [{r_item}] - 일치")
                    else:
                        st.error(f"❌ {i+1}번 부품: 회로도는 [{r_item}]인데, 실물은 [{t_item}]입니다.")
                        is_seq_match = False
                
                if is_seq_match:
                    st.success("🎉 완벽합니다! 부품의 종류, 개수, 순서가 모두 일치합니다.")
                    st.balloons()

            st.image(cv2.cvtColor(res_ref_img, cv2.COLOR_BGR2RGB), caption=f"회로도 정렬 ({len(ref_list)}개)", use_column_width=True)
            st.image(cv2.cvtColor(res_tgt_img, cv2.COLOR_BGR2RGB), caption=f"실물 정렬 ({len(tgt_list)}개)", use_column_width=True)
            
            del res_ref_img, res_tgt_img
            gc.collect()
