import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import math
from PIL import Image

# ==========================================
# [1. 설정 및 경로]
# ==========================================
st.set_page_config(page_title="BrainBoard V53: Hybrid Final", layout="wide")

MODEL_REAL_PATH = 'best(3).pt'
MODEL_SYM_PATH = 'symbol.pt'

# 실물 부품별 신뢰도 임계값
CONFIDENCE_MAP_REAL = {
    'led': 0.50,
    'capacitor': 0.40,
    'voltage': 0.25,
    'source': 0.25,
    'resistor': 0.65, # 실물 저항은 엄격하게
    'wire': 0.25,
    'default': 0.30
}

# ==========================================
# [2. 핵심 유틸리티 함수]
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

def normalize_name(raw_name):
    name = raw_name.lower().strip()
    if any(x in name for x in ['res', 'r']): return 'RESISTOR'
    if any(x in name for x in ['cap', 'c']): return 'CAPACITOR'
    if any(x in name for x in ['v', 'volt', 'batt', 'source', 'vdc']): return 'SOURCE'
    if any(x in name for x in ['pin', 'leg', 'lead']): return 'PIN'
    if 'wire' in name: return 'WIRE'
    return 'OTHER'

# [V52 필터] 와이어 오인식 방지 (가로세로 비율 체크)
def fix_misclassified_wire(box, name):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    if w == 0 or h == 0: return name
    ratio = max(w, h) / min(w, h)
    # 저항 몸체 치고는 너무 길쭉하면 100% 와이어입니다.
    if name == 'RESISTOR' and ratio > 6.0:
        return 'WIRE'
    return name

# [V53 개선] 중복 제거 기준 분리
def solve_overlap(parts, is_schematic=False):
    if not parts: return []
    
    if is_schematic:
        # 회로도: 면적 작은 것 우선 (기존 잘 되던 V35 로직)
        parts.sort(key=lambda x: (x['box'][2]-x['box'][0]) * (x['box'][3]-x['box'][1]))
        dist_thresh = 20 # 부품과 텍스트가 붙어있으므로 아주 좁게 설정
    else:
        # 실물: 신뢰도 높은 것 우선 (V48 로직)
        parts.sort(key=lambda x: x.get('conf', 0), reverse=True)
        dist_thresh = 80 # 그림자나 다리 중복 제거를 위해 넓게 설정

    final = []
    for curr in parts:
        is_dup = False
        for k in final:
            dist = math.sqrt((curr['center'][0]-k['center'][0])**2 + (curr['center'][1]-k['center'][1])**2)
            if dist < dist_thresh:
                is_dup = True; break
        if not is_dup:
            final.append(curr)
    return final

# ==========================================
# [3. 분석 엔진]
# ==========================================
def analyze_schematic(img, model):
    # 기존에 잘 잡히던 설정값 그대로 유지
    res = model.predict(source=img, conf=0.15, verbose=False)
    raw = []
    for b in res[0].boxes:
        name = normalize_name(model.names[int(b.cls[0])])
        if name == 'OTHER': continue
        coords = b.xyxy[0].tolist()
        raw.append({'name': name, 'box': coords, 'center': get_center(coords)})
    
    clean = solve_overlap(raw, is_schematic=True)
    
    # 전원 보정
    if clean and not any(p['name'] == 'SOURCE' for p in clean):
        min(clean, key=lambda p: p['center'][0])['name'] = 'SOURCE'

    summary = {}
    for p in clean:
        x1, y1, x2, y2 = map(int, p['box'])
        color = (255, 0, 0) if p['name'] == 'SOURCE' else (0, 0, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, p['name'], (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        summary[p['name']] = summary.get(p['name'], 0) + 1
    return img, summary

def analyze_real(img, model):
    res = model.predict(source=img, conf=0.10, verbose=False)
    raw = []
    for b in res[0].boxes:
        name = model.names[int(b.cls[0])].lower()
        conf = float(b.conf[0])
        
        # 신뢰도 필터
        thresh = CONFIDENCE_MAP_REAL.get('default')
        for k in CONFIDENCE_MAP_REAL:
            if k in name: thresh = CONFIDENCE_MAP_REAL[k]; break
        if conf < thresh: continue
        
        coords = b.xyxy[0].tolist()
        norm_name = normalize_name(name)
        # [V52] 와이어 필터 적용
        norm_name = fix_misclassified_wire(coords, norm_name)
        
        if norm_name in ['OTHER', 'PIN']: continue # PIN은 화면엔 보여도 개수에서 뺌
        
        raw.append({'name': norm_name, 'box': coords, 'center': get_center(coords), 'conf': conf})

    clean = solve_overlap(raw, is_schematic=False)
    
    summary = {}
    for b in clean:
        x1, y1, x2, y2 = map(int, b['box'])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(img, b['name'], (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        if b['name'] != 'WIRE':
            summary[b['name']] = summary.get(b['name'], 0) + 1
            
    # 와이어가 많으면 전원 1개 가상 추가 (사용자 로직 유지)
    if sum(1 for b in clean if b['name'] == 'WIRE') >= 2:
        summary['SOURCE'] = summary.get('SOURCE', 0) + 1
        
    return img, summary

# ==========================================
# [4. UI 및 실행]
# ==========================================
st.title("🧠 BrainBoard V53: Perfect Hybrid")

@st.cache_resource
def load_models():
    return YOLO(MODEL_REAL_PATH), YOLO(MODEL_SYM_PATH)

model_real, model_sym = load_models()

col1, col2 = st.columns(2)
ref_file = col1.file_uploader("1. 회로도 업로드", type=['jpg', 'png', 'jpeg'])
tgt_file = col2.file_uploader("2. 실물 사진 업로드", type=['jpg', 'png', 'jpeg'])

if ref_file and tgt_file:
    if st.button("🚀 정밀 분석 시작"):
        ref_cv = cv2.cvtColor(np.array(Image.open(ref_file)), cv2.COLOR_RGB2BGR)
        tgt_cv = cv2.cvtColor(np.array(Image.open(tgt_file)), cv2.COLOR_RGB2BGR)

        res_ref, data_ref = analyze_schematic(ref_cv.copy(), model_sym)
        res_tgt, data_tgt = analyze_real(tgt_cv.copy(), model_real)

        st.divider()
        all_parts = set(data_ref.keys()) | set(data_tgt.keys())
        for p in sorted(all_parts):
            r, t = data_ref.get(p, 0), data_tgt.get(p, 0)
            if r == t: st.success(f"✅ {p.upper()} 일치: {r}개")
            else: st.error(f"⚠️ {p.upper()} 불일치: 회로도 {r}개 vs 실물 {t}개")

        st.image(cv2.cvtColor(res_ref, cv2.COLOR_BGR2RGB), caption="회로도 분석 (기존 성능 복구)", use_column_width=True)
        st.image(cv2.cvtColor(res_tgt, cv2.COLOR_BGR2RGB), caption="실물 분석 (와이어 오인식 제거)", use_column_width=True)
