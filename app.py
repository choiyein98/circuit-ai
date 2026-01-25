import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import math
from PIL import Image

# ==========================================
# [설정] BrainBoard V55: Hybrid Edition
# ==========================================
st.set_page_config(page_title="BrainBoard V55: Hybrid", layout="wide")

# [모델 설정]
REAL_MODEL_PATHS = ['best.pt', 'best(2).pt', 'best(3).pt']
MODEL_SYM_PATH = 'symbol.pt'

# [V35 로직 상수 복원] 거리 기반 연결성의 핵심
LEG_EXTENSION_RANGE = 180 

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

def dist(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

# ==========================================
# [중복 제거] 기존 로직 유지
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
            d = dist((cx1, cy1), (cx2, cy2))
            if iou > 0.1 or d < distance_threshold:
                is_duplicate = True; break
        if not is_duplicate: final_parts.append(current)
    return final_parts

def solve_overlap_real_v35(parts, dist_thresh=60, iou_thresh=0.4):
    if not parts: return []
    parts.sort(key=lambda x: x.get('conf', 0), reverse=True)
    final = []
    for curr in parts:
        is_dup = False
        for k in final:
            iou = calculate_iou(curr['box'], k['box'])
            if iou > iou_thresh: is_dup = True; break
            d = dist(curr['center'], k['center'])
            if d < dist_thresh: is_dup = True; break
        if not is_dup: final.append(curr)
    return final

# ==========================================
# [분석 1] 회로도 (V48 로직 유지 - 2번째 사진 스타일)
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
        
        base_name = name.split('_')[0].split(' ')[0]
        if base_name in ['vdc', 'vsource', 'battery', 'voltage', 'v']: base_name = 'source'
        if base_name in ['cap', 'c', 'capacitor']: base_name = 'capacitor'
        if base_name in ['res', 'r', 'resistor']: base_name = 'resistor'
        
        raw_parts.append({'name': base_name, 'box': coords, 'center': get_center(coords), 'conf': conf})

    clean_parts = solve_overlap_schematic_v48(raw_parts)

    summary = {'total': 0, 'details': {}}
    for part in clean_parts:
        name = part['name']
        x1, y1, x2, y2 = map(int, part['box'])
        
        # 디자인: 파란색 박스 (2번째 사진 스타일)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img, f"{name}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        summary['total'] += 1
        summary['details'][name] = summary['details'].get(name, 0) + 1
        
    return img, summary

# ==========================================
# [분석 2] 실물 보드 (V35 로직 복원 + 엔지니어링 체크)
# ==========================================
def analyze_real_hybrid(img, model_list):
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
            conf = float(b.conf[0])
            
            # 민감도 설정 (V35 Original)
            if 'cap' in name: min_conf = 0.15
            elif 'res' in name: min_conf = 0.60
            elif 'wire' in name: min_conf = 0.15
            else: min_conf = 0.25
                
            if conf < min_conf: continue

            if any(x in name for x in ['pin', 'leg', 'lead']) and 'wire' not in name:
                raw_pins.append({'center': get_center(coords), 'box': coords})
            else:
                raw_bodies.append({'name': name, 'box': coords, 'center': get_center(coords), 'conf': conf, 'is_on': False, 'status': 'OFF'})

    clean_bodies = solve_overlap_real_v35(raw_bodies)

    # ---------------------------------------------------------
    # [Logic Restoration] V35의 '거리 기반' 연결성 판단 (관대함)
    # ---------------------------------------------------------
    # 1. 전원부 탐지 (Dynamic Power Rail)
    breadboard_box = [0, 0, w, h]
    for b in clean_bodies:
        if 'breadboard' in b['name']: breadboard_box = b['box']; break
    
    bb_y1, bb_y2 = breadboard_box[1], breadboard_box[3]
    bb_h = bb_y2 - bb_y1
    
    # 상단 20%, 하단 20%를 전원 레일로 간주 (좀 더 넓게 잡음)
    power_rail_top = bb_y1 + (bb_h * 0.20)
    power_rail_bot = bb_y2 - (bb_h * 0.20)

    # 2. 전원 활성화 여부 (와이어나 핀이 전원부에 있는가?)
    power_active = False
    for b in clean_bodies:
        if 'wire' in b['name'] and (b['center'][1] < power_rail_top or b['center'][1] > power_rail_bot):
            power_active = True; break
    if not power_active:
        for p in raw_pins:
            if p['center'][1] < power_rail_top or p['center'][1] > power_rail_bot:
                power_active = True; break
                
    # 3. 연결 전파 (V35 Recursive Logic)
    if power_active:
        # A. 전원부에 직접 닿은 부품 ON
        for comp in clean_bodies:
            cy = comp['center'][1]
            if cy < power_rail_top or cy > power_rail_bot:
                comp['is_on'] = True

        # B. 거리 기반 전파 (3 Pass)
        for _ in range(3):
            for comp in clean_bodies:
                if comp['is_on']: continue
                cx, cy = comp['center']
                
                # 핀과의 거리
                for p in raw_pins:
                    py = p['center'][1]
                    if py < power_rail_top or py > power_rail_bot:
                        if dist((cx,cy), p['center']) < LEG_EXTENSION_RANGE:
                            comp['is_on'] = True; break
                
                if comp['is_on']: continue

                # 다른 켜진 부품과의 거리 (와이어 포함)
                for other in clean_bodies:
                    if not other['is_on']: continue
                    # 와이어라면 더 멀리서도 연결 인정
                    range_mult = 1.5 if 'wire' in other['name'] else 1.0
                    if dist((cx,cy), other['center']) < LEG_EXTENSION_RANGE * range_mult:
                        comp['is_on'] = True; break

    # ---------------------------------------------------------
    # [Engineering Safety] 쇼트(Short) 감지 추가
    # ---------------------------------------------------------
    # 같은 종류의 부품이 너무 가까이 겹쳐 있으면 물리적 충돌/쇼트 의심
    for i, c1 in enumerate(clean_bodies):
        if 'wire' in c1['name'] or 'breadboard' in c1['name']: continue
        for j, c2 in enumerate(clean_bodies):
            if i >= j: continue
            if 'wire' in c2['name'] or 'breadboard' in c2['name']: continue
            
            # IoU가 너무 높으면 (80% 이상 겹치면) 이상한 배치
            if calculate_iou(c1['box'], c2['box']) > 0.8:
                c1['status'] = 'SHORT'
                c2['status'] = 'SHORT'

    summary = {'total': 0, 'on': 0, 'off': 0, 'details': {}}

    for comp in clean_bodies:
        raw_name = comp['name']
        if 'breadboard' in raw_name: continue

        norm_name = raw_name
        label_name = ""
        if 'res' in raw_name: norm_name = 'resistor'; label_name = "RES"
        elif 'cap' in raw_name: norm_name = 'capacitor'; label_name = "CAP"
        elif 'wire' in raw_name: label_name = "WIRE"
        else: label_name = raw_name[:3].upper()
        
        if 'wire' not in raw_name:
            if norm_name not in summary['details']: summary['details'][norm_name] = {'count': 0}
            summary['details'][norm_name]['count'] += 1

        # 상태 결정 (V35 결과 + Short 체크)
        is_short = (comp.get('status') == 'SHORT')
        is_on = comp['is_on']

        if is_short:
            color = (0, 0, 255) # Red (Danger)
            status_text = "SHORT"
            summary['off'] += 1
        elif is_on:
            color = (0, 255, 0) # Green (OK - 3번째 사진 스타일)
            status_text = "ON"
            summary['on'] += 1
        else:
            color = (0, 0, 255) # Red (OFF)
            status_text = "OFF"
            summary['off'] += 1
            
        summary['total'] += 1
        
        # 그리기
        x1, y1, x2, y2 = map(int, comp['box'])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        # 3번째 사진처럼 간결한 텍스트
        cv2.putText(img, f"{label_name}_{status_text}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return img, summary

# ==========================================
# [Main UI]
# ==========================================
st.title("🧠 BrainBoard V55: Hybrid Edition")
st.markdown("### ✅ 완벽한 인식(V35) + 엔지니어링 검증")

@st.cache_resource
def load_models():
    real_models = []
    try:
        for path in REAL_MODEL_PATHS:
            try: real_models.append(YOLO(path))
            except: pass
        sym_model = YOLO(MODEL_SYM_PATH)
    except: return [], None
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

    if st.button("🚀 분석 실행"):
        res_ref, ref_data = analyze_schematic(ref_cv.copy(), model_sym)
        res_tgt, tgt_data = analyze_real_hybrid(tgt_cv.copy(), models_real)

        st.image(cv2.cvtColor(res_ref, cv2.COLOR_BGR2RGB), caption="회로도 (Schematic)", use_column_width=True)
        st.image(cv2.cvtColor(res_tgt, cv2.COLOR_BGR2RGB), caption="실물 검증 (Hybrid V55)", use_column_width=True)
        
        # [엔지니어링 리포트]
        st.divider()
        st.subheader("📋 검증 리포트")
        
        all_keys = set(ref_data['details'].keys()) | set(tgt_data['details'].keys())
        match_all = True
        
        for k in all_keys:
            if k in ['text', 'source']: continue
            r_cnt = ref_data['details'].get(k, 0)
            t_cnt = tgt_data['details'].get(k, {}).get('count', 0)
            
            if r_cnt == t_cnt:
                st.success(f"✅ {k.upper()}: 개수 일치 ({r_cnt}개)")
            else:
                match_all = False
                st.error(f"⚠️ {k.upper()}: 개수 불일치 (회로도 {r_cnt} vs 실물 {t_cnt})")
                
        if match_all and tgt_data['off'] == 0:
            st.balloons()
            st.success("🎉 회로 구성 및 연결이 완벽합니다!")
        elif tgt_data['off'] > 0:
             st.warning(f"⚠️ 일부 부품이 연결되지 않았거나(OFF) 쇼트(SHORT)가 의심됩니다. ({tgt_data['off']}개)")
