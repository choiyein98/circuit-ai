import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import math
from PIL import Image

# ==========================================
# [설정] BrainBoard V57: Sensitivity Fix
# ==========================================
st.set_page_config(page_title="BrainBoard V57", layout="wide")

# [모델 설정]
REAL_MODEL_PATHS = ['best.pt', 'best(2).pt', 'best(3).pt']
MODEL_SYM_PATH = 'symbol.pt'
LEG_EXTENSION_RANGE = 180
SHORT_CIRCUIT_IOU = 0.6

# ==========================================
# [Helper Functions] 공통 함수
# ==========================================
def calculate_iou(box1, box2):
    x1, y1, x2, y2 = max(box1[0], box2[0]), max(box1[1], box2[1]), min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

def get_center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

# ==========================================
# [중복 제거 1] V48 스타일 (회로도용)
# ==========================================
def solve_overlap_schematic_v48(parts, distance_threshold=80):
    if not parts: return []
    parts.sort(key=lambda x: x['conf'], reverse=True)
    final_parts = []
    for current in parts:
        is_duplicate = False
        for kept in final_parts:
            iou = calculate_iou(current['box'], kept['box'])
            cx1, cy1 = current['center']
            cx2, cy2 = kept['center']
            dist = math.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)
            if iou > 0.1 or dist < distance_threshold:
                is_duplicate = True; break
        if not is_duplicate: final_parts.append(current)
    return final_parts

# ==========================================
# [중복 제거 2] V35 스타일 (실물용)
# ==========================================
def solve_overlap_real_v35(parts, dist_thresh=60, iou_thresh=0.4):
    if not parts: return []
    parts.sort(key=lambda x: x.get('conf', 0), reverse=True)
    final = []
    for curr in parts:
        is_dup = False
        for k in final:
            x1 = max(curr['box'][0], k['box'][0])
            y1 = max(curr['box'][1], k['box'][1])
            x2 = min(curr['box'][2], k['box'][2])
            y2 = min(curr['box'][3], k['box'][3])
            inter_area = max(0, x2-x1) * max(0, y2-y1)
            area_curr = (curr['box'][2]-curr['box'][0]) * (curr['box'][3]-curr['box'][1])
            area_k = (k['box'][2]-k['box'][0]) * (k['box'][3]-k['box'][1])
            min_area = min(area_curr, area_k)
            ratio = inter_area / min_area if min_area > 0 else 0
            iou = calculate_iou(curr['box'], k['box'])
            if ratio > 0.8: is_dup = True; break
            if iou > iou_thresh: is_dup = True; break
            dist = math.sqrt((curr['center'][0]-k['center'][0])**2 + (curr['center'][1]-k['center'][1])**2)
            if dist < dist_thresh: is_dup = True; break
        if not is_dup: final.append(curr)
    return final

# ==========================================
# [분석 1] 회로도 분석
# ==========================================
def analyze_schematic(img, model):
    results = model.predict(source=img, save=False, conf=0.05, verbose=False)
    boxes = results[0].boxes
    raw_parts = []
    for box in boxes:
        cls_id = int(box.cls[0])
        name = model.names[cls_id].lower()
        conf = float(box.conf[0])
        coords = box.xyxy[0].tolist()
        center = get_center(coords)
        base_name = name.split('_')[0].split(' ')[0]
        if base_name in ['vdc', 'vsource', 'battery', 'voltage', 'v']: base_name = 'source'
        if base_name in ['cap', 'c', 'capacitor']: base_name = 'capacitor'
        if base_name in ['res', 'r', 'resistor']: base_name = 'resistor'
        raw_parts.append({'name': base_name, 'box': coords, 'center': center, 'conf': conf})

    clean_parts = solve_overlap_schematic_v48(raw_parts)

    if clean_parts:
        has_source = any(p['name'] == 'source' for p in clean_parts)
        if not has_source:
            leftmost_part = min(clean_parts, key=lambda p: p['center'][0])
            leftmost_part['name'] = 'source'

    summary = {'total': 0, 'details': {}}
    for part in clean_parts:
        name = part['name']
        x1, y1, x2, y2 = map(int, part['box'])
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img, f"{name}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        summary['total'] += 1
        summary['details'][name] = summary['details'].get(name, 0) + 1
    return img, summary

# ==========================================
# [분석 2] 실물 보드 분석 (저항 인식률 개선)
# ==========================================
def analyze_real_ensemble(img, model_list):
    h, w, _ = img.shape
    raw_bodies = []
    raw_pins = [] 
    
    # 1. 앙상블 탐지
    for model in model_list:
        res = model.predict(source=img, conf=0.10, verbose=False)
        boxes = res[0].boxes
        for b in boxes:
            name = model.names[int(b.cls[0])].lower()
            coords = b.xyxy[0].tolist()
            center = get_center(coords)
            conf = float(b.conf[0])
            
            # [수정됨] 저항(Resistor)의 인식 임계값을 0.60 -> 0.25로 대폭 완화
            if 'cap' in name: min_conf = 0.15
            elif 'res' in name: min_conf = 0.25 # 기존 0.60에서 수정됨 (작은 저항 인식 강화)
            elif 'wire' in name: min_conf = 0.15
            else: min_conf = 0.25
            
            if conf < min_conf: continue

            if any(x in name for x in ['pin', 'leg', 'lead']) and 'wire' not in name:
                raw_pins.append({'center': center, 'box': coords})
            elif 'breadboard' in name:
                continue
            else:
                raw_bodies.append({'name': name, 'box': coords, 'center': center, 'conf': conf, 
                                   'is_on': False, 'is_short': False})

    # 2. 중복 제거
    clean_bodies = solve_overlap_real_v35(raw_bodies, dist_thresh=60, iou_thresh=0.4)
    
    # 3. 연결 로직 (V35 Logic)
    power_active = False
    for b in clean_bodies:
        if 'wire' in b['name'] and b['center'][1] < h * 0.45:
            power_active = True; break
    if not power_active:
        for p in raw_pins:
            if p['center'][1] < h * 0.45:
                power_active = True; break

    if power_active:
        for comp in clean_bodies:
            cy = comp['center'][1]
            if cy < h*0.48 or cy > h*0.52: comp['is_on'] = True

        for _ in range(3): 
            for comp in clean_bodies:
                if comp['is_on']: continue 
                cx, cy = comp['center']
                for p in raw_pins:
                    px, py = p['center']
                    if py < h*0.48 or py > h*0.52:
                         dist = math.sqrt((cx - px)**2 + (cy - py)**2)
                         if dist < LEG_EXTENSION_RANGE: comp['is_on'] = True; break
                if comp['is_on']: continue
                for other in clean_bodies:
                    if not other['is_on']: continue
                    ocx, ocy = other['center']
                    dist = math.sqrt((cx - ocx)**2 + (cy - ocy)**2)
                    if dist < LEG_EXTENSION_RANGE * 1.5: comp['is_on'] = True; break

    # 4. 쇼트(Short) 감지
    for i, c1 in enumerate(clean_bodies):
        if 'wire' in c1['name']: continue
        for j, c2 in enumerate(clean_bodies):
            if i >= j: continue
            if 'wire' in c2['name']: continue
            overlap_ratio = calculate_iou(c1['box'], c2['box'])
            if overlap_ratio > SHORT_CIRCUIT_IOU:
                c1['is_short'] = True
                c2['is_short'] = True

    # 5. 결과 시각화
    summary = {'total': 0, 'on': 0, 'off': 0, 'short': 0, 'details': {}}
    
    for comp in clean_bodies:
        is_on = comp['is_on']
        is_short = comp['is_short']
        raw_name = comp['name']
        
        norm_name = raw_name
        label_name = "" 
        if 'res' in raw_name: norm_name = 'resistor'; label_name = "RES"
        elif 'cap' in raw_name: norm_name = 'capacitor'; label_name = "CAP"
        elif 'wire' in raw_name: label_name = "WIRE"
        else: label_name = raw_name[:3].upper()
        
        if 'wire' not in raw_name:
            if norm_name not in summary['details']: summary['details'][norm_name] = {'count': 0}
            summary['details'][norm_name]['count'] += 1

        if is_short:
            color = (0, 0, 255)    # Red
            status_text = "SHORT!" 
            summary['short'] += 1
            summary['off'] += 1
        elif is_on:
            summary['on'] += 1
            color = (0, 255, 0)   # Green (무조건 초록)
            status_text = "ON"
        else:
            color = (0, 0, 255)   # Red
            status_text = "OFF"
            summary['off'] += 1
        
        summary['total'] += 1
        
        x1, y1, x2, y2 = map(int, comp['box'])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.putText(img, f"{label_name}:{status_text}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
    return img, summary

# ==========================================
# [Main UI]
# ==========================================
st.title("🧠 BrainBoard V57: Resistor Sensitivity Fix")
st.markdown("""
### ✅ 업데이트 내역
- **저항 인식률 개선**: 저항(Resistor) 감지 민감도를 대폭 완화하여(60% -> 25%) 인식되지 않던 부품을 잡아냅니다.
""")

@st.cache_resource
def load_models():
    real_models = []
    try:
        for path in REAL_MODEL_PATHS:
            try: real_models.append(YOLO(path))
            except: pass
        sym_model = YOLO(MODEL_SYM_PATH)
    except Exception: return [], None
    return real_models, sym_model

models_real, model_sym = load_models()

if not models_real:
    st.error("❌ 모델 로드 실패")
    st.stop()

col1, col2 = st.columns(2)
ref_file = col1.file_uploader("1. 회로도", type=['jpg', 'png', 'jpeg'])
tgt_file = col2.file_uploader("2. 실물 사진", type=['jpg', 'png', 'jpeg'])

if ref_file and tgt_file:
    ref_cv = cv2.cvtColor(np.array(Image.open(ref_file)), cv2.COLOR_RGB2BGR)
    tgt_cv = cv2.cvtColor(np.array(Image.open(tgt_file)), cv2.COLOR_RGB2BGR)

    if st.button("🚀 검증 실행"):
        res_ref, ref_data = analyze_schematic(ref_cv.copy(), model_sym)
        res_tgt, tgt_data = analyze_real_ensemble(tgt_cv.copy(), models_real)
        
        st.divider()
        st.subheader("📊 검증 결과")

        all_keys = set(ref_data['details'].keys()) | set(tgt_data['details'].keys())
        match_all = True
        for k in all_keys:
            if k in ['text', 'source']: continue
            r_cnt = ref_data['details'].get(k, 0)
            t_cnt = tgt_data['details'].get(k, {}).get('count', 0)
            
            if r_cnt == t_cnt:
                st.success(f"✅ {k.upper()}: 수량 일치 ({r_cnt}개)")
            else:
                match_all = False
                st.error(f"⚠️ {k.upper()}: 수량 불일치 (회로도 {r_cnt} vs 실물 {t_cnt})")

        if tgt_data['short'] > 0:
            st.error(f"🚨 **합선 경고**: {tgt_data['short']}개의 부품이 겹쳐 있어 위험합니다.")
        elif tgt_data['off'] > 0:
            st.warning(f"⚠️ **연결 끊김**: {tgt_data['off']}개의 부품이 전원에 연결되지 않았습니다.")
        elif match_all:
            st.balloons()
            st.success("🎉 모든 연결이 정상입니다 (ALL GREEN)!")

        col_r1, col_r2 = st.columns(2)
        with col_r1: st.image(cv2.cvtColor(res_ref, cv2.COLOR_BGR2RGB), caption="회로도", use_column_width=True)
        with col_r2: st.image(cv2.cvtColor(res_tgt, cv2.COLOR_BGR2RGB), caption="실물 검증", use_column_width=True)
