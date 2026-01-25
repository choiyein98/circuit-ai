import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import math
from PIL import Image

# ==========================================
# [설정] V49: 회로도 성능 유지 + 실물 로직 강화
# ==========================================
st.set_page_config(page_title="BrainBoard V49: Integrated System", layout="wide")

MODEL_REAL_PATH = 'best(3).pt'  # 실물 모델
MODEL_SYM_PATH = 'symbol.pt'    # 회로도 모델

# 실물 연결 감지 범위 및 신뢰도 최적화
LEG_EXTENSION_RANGE = 180       
CONFIDENCE_MAP_REAL = {
    'resistor': 0.20,  # 저항 인식률 대폭 강화
    'capacitor': 0.35,
    'wire': 0.15,
    'default': 0.25
}

# ==========================================
# [1. 공통 유틸리티 함수]
# ==========================================
def calculate_iou(box1, box2):
    x1, y1, x2, y2 = max(box1[0], box2[0]), max(box1[1], box2[1]), min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

def solve_overlap(parts, dist_thresh=40, is_schematic=False):
    if not parts: return []
    # 회로도는 작은 것 우선, 실물은 신뢰도 우선
    if is_schematic:
        parts.sort(key=lambda x: (x['box'][2]-x['box'][0]) * (x['box'][3]-x['box'][1]))
    else:
        parts.sort(key=lambda x: x.get('conf', 0), reverse=True)
    
    final = []
    for curr in parts:
        is_dup = False
        for k in final:
            iou = calculate_iou(curr['box'], k['box'])
            dist = math.sqrt((curr['center'][0]-k['center'][0])**2 + (curr['center'][1]-k['center'][1])**2)
            if iou > 0.4 or dist < dist_thresh:
                is_dup = True; break
        if not is_dup: final.append(curr)
    return final

def get_center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

# ==========================================
# [2. 회로도 분석 (기존 성능 유지)]
# ==========================================
def analyze_schematic(img, model):
    res = model.predict(source=img, conf=0.15, verbose=False)
    raw = []
    for b in res[0].boxes:
        name = model.names[int(b.cls[0])].lower()
        coords = b.xyxy[0].tolist()
        base_name = name.split('_')[0]
        if any(x in base_name for x in ['vdc', 'source', 'volt']): base_name = 'source'
        elif any(x in base_name for x in ['cap', 'c']): base_name = 'capacitor'
        elif any(x in base_name for x in ['res', 'r']): base_name = 'resistor'
        raw.append({'name': base_name, 'box': coords, 'center': get_center(coords)})

    clean = solve_overlap(raw, dist_thresh=30, is_schematic=True)
    if clean and not any(p['name'] == 'source' for p in clean):
        min(clean, key=lambda p: p['center'][0])['name'] = 'source'

    summary = {'details': {}}
    for p in clean:
        x1, y1, x2, y2 = map(int, p['box'])
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2) # 파란색
        cv2.putText(img, p['name'], (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        summary['details'][p['name']] = summary['details'].get(p['name'], 0) + 1
    return img, summary

# ==========================================
# [3. 실물 보드 분석 (우리가 만든 강화 로직)]
# ==========================================
def analyze_real(img, model):
    h, w, _ = img.shape
    res = model.predict(source=img, conf=0.10, verbose=False)
    
    bodies, pins = [], []
    for b in res[0].boxes:
        name = model.names[int(b.cls[0])].lower()
        conf = float(b.conf[0])
        coords = b.xyxy[0].tolist()
        center = get_center(coords)
        
        # 신뢰도 필터링
        thresh = CONFIDENCE_MAP_REAL.get('default')
        for k in CONFIDENCE_MAP_REAL:
            if k in name: thresh = CONFIDENCE_MAP_REAL[k]; break
        if conf < thresh: continue

        if any(x in name for x in ['pin', 'leg', 'lead']):
            pins.append({'center': center})
        elif name not in ['breadboard', 'hole']:
            bodies.append({'name': name, 'box': coords, 'center': center, 'conf': conf, 'is_on': False})

    clean_bodies = solve_overlap(bodies, dist_thresh=40, is_schematic=False)

    # 연결 로직 (3단계 전파)
    # 1. 전원 와이어 확인 (상단/하단 레일)
    power_active = any('wire' in b['name'] and (b['center'][1] < h*0.45 or b['center'][1] > h*0.55) for b in clean_bodies)
    
    if power_active:
        # 2. 핀 접촉 기반 활성화
        for comp in clean_bodies:
            cx, cy = comp['center']
            for p in pins:
                dist = math.sqrt((cx-p['center'][0])**2 + (cy-p['center'][1])**2)
                if dist < LEG_EXTENSION_RANGE:
                    comp['is_on'] = True; break
        
        # 3. 인접 부품 간 전파 (와이어 등)
        for _ in range(2):
            for b1 in clean_bodies:
                if b1['is_on']: continue
                for b2 in clean_bodies:
                    if b2['is_on'] and math.sqrt((b1['center'][0]-b2['center'][0])**2 + (b1['center'][1]-b2['center'][1])**2) < LEG_EXTENSION_RANGE:
                        b1['is_on'] = True; break

    summary = {'off': 0, 'details': {}}
    for b in clean_bodies:
        color = (0, 255, 0) if b['is_on'] else (0, 0, 255) # ON=초록, OFF=빨강
        if not b['is_on']: summary['off'] += 1
        
        x1, y1, x2, y2 = map(int, b['box'])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.putText(img, f"{b['name']}: {'ON' if b['is_on'] else 'OFF'}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        norm_name = 'resistor' if 'res' in b['name'] else 'capacitor' if 'cap' in b['name'] else 'source' if 'volt' in b['name'] else b['name']
        if 'wire' not in norm_name:
            if norm_name not in summary['details']: summary['details'][norm_name] = {'count': 0}
            summary['details'][norm_name]['count'] += 1
            
    return img, summary

# ==========================================
# [4. UI 및 실행]
# ==========================================
st.title("🧠 BrainBoard V49: Final Hybrid System")

@st.cache_resource
def load_models():
    return YOLO(MODEL_REAL_PATH), YOLO(MODEL_SYM_PATH)

model_real, model_sym = load_models()

col1, col2 = st.columns(2)
ref_file = col1.file_uploader("1. 회로도 업로드", type=['jpg', 'png', 'jpeg'])
tgt_file = col2.file_uploader("2. 실물 사진 업로드", type=['jpg', 'png', 'jpeg'])

if ref_file and tgt_file:
    if st.button("🚀 하이브리드 분석 시작"):
        ref_cv = cv2.cvtColor(np.array(Image.open(ref_file)), cv2.COLOR_RGB2BGR)
        tgt_cv = cv2.cvtColor(np.array(Image.open(tgt_file)), cv2.COLOR_RGB2BGR)

        res_ref, data_ref = analyze_schematic(ref_cv, model_sym)
        res_tgt, data_tgt = analyze_real(tgt_cv, model_real)

        st.divider()
        # 부품 비교 로직
        all_comps = set(data_ref['details'].keys()) | set(data_tgt['details'].keys())
        for c in sorted(all_comps):
            r = data_ref['details'].get(c, 0)
            t = data_tgt['details'].get(c, {}).get('count', 0)
            if r == t: st.success(f"✅ {c.upper()} 일치: {r}개")
            else: st.error(f"⚠️ {c.upper()} 불일치: 회로도 {r}개 vs 실물 {t}개")

        st.image(cv2.cvtColor(res_ref, cv2.COLOR_BGR2RGB), caption="PSpice 회로도 분석 (안정 모드)", use_column_width=True)
        st.image(cv2.cvtColor(res_tgt, cv2.COLOR_BGR2RGB), caption=f"실물 분석 (강화 모드 - OFF: {data_tgt['off']})", use_column_width=True)
