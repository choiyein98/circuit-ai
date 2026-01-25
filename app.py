import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import math
from PIL import Image

# ==========================================
# [1. 설정]
# ==========================================
st.set_page_config(page_title="BrainBoard V50: Universal Detection", layout="wide")

MODEL_REAL_PATH = 'best(3).pt'
MODEL_SYM_PATH = 'symbol.pt'

# ==========================================
# [2. 범용 유틸리티 함수]
# ==========================================
def get_center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

def solve_overlap_universal(parts, iou_thresh=0.25, is_schematic=False):
    if not parts: return []
    
    # 신뢰도 순 정렬 (확실한 부품부터 선점)
    parts.sort(key=lambda x: x.get('conf', 0), reverse=True)
    
    final = []
    for curr in parts:
        is_dup = False
        for k in final:
            # IoU(겹침 정도) 계산
            x1, y1, x2, y2 = max(curr['box'][0], k['box'][0]), max(curr['box'][1], k['box'][1]), \
                             min(curr['box'][2], k['box'][2]), min(curr['box'][3], k['box'][3])
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            area1 = (curr['box'][2] - curr['box'][0]) * (curr['box'][3] - curr['box'][1])
            area2 = (k['box'][2] - k['box'][0]) * (k['box'][3] - k['box'][1])
            iou = inter / (area1 + area2 - inter) if (area1 + area2 - inter) > 0 else 0
            
            # 너무 가깝거나 많이 겹치면 중복 제거
            if iou > iou_thresh:
                is_dup = True; break
                
        if not is_dup:
            final.append(curr)
    return final

# ==========================================
# [3. 범용 분석 엔진]
# ==========================================
def analyze_universal(img, model, is_schematic=False):
    # [핵심 1] 이미지 크기에 상관없이 YOLO 학습 규격(640)으로 내부 리사이즈
    # 이렇게 해야 먼 거리 사진도, 가까운 사진도 일관되게 인식합니다.
    h, w, _ = img.shape
    results = model.predict(source=img, conf=0.25, imgsz=640, verbose=False)
    
    raw = []
    for b in results[0].boxes:
        name = model.names[int(b.cls[0])].lower()
        conf = float(b.conf[0])
        coords = b.xyxy[0].tolist()
        
        # 이름 정규화 (부품군 통합)
        if 'res' in name: norm_name = 'resistor'
        elif 'cap' in name: norm_name = 'capacitor'
        elif any(x in name for x in ['v', 'volt', 'batt', 'source']): norm_name = 'source'
        else: norm_name = name
        
        if norm_name in ['breadboard', 'hole', 'text']: continue
        
        raw.append({
            'name': norm_name,
            'box': coords,
            'center': get_center(coords),
            'conf': conf
        })
    
    # [핵심 2] 중복 제거 로직 가동
    clean = solve_overlap_universal(raw, iou_thresh=0.2, is_schematic=is_schematic)
    
    # [핵심 3] 회로도 전원 자동 보정 (전원이 안 잡힐 경우 대비)
    if is_schematic and clean and not any(p['name'] == 'source' for p in clean):
        min(clean, key=lambda p: p['center'][0])['name'] = 'source'

    summary = {}
    for p in clean:
        x1, y1, x2, y2 = map(int, p['box'])
        # 회로도는 파랑, 실물은 초록
        color = (255, 0, 0) if is_schematic else (0, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.putText(img, p['name'], (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        if 'wire' not in p['name']:
            summary[p['name']] = summary.get(p['name'], 0) + 1
            
    return img, summary

# ==========================================
# [4. UI 및 실행]
# ==========================================
st.title("🧠 BrainBoard V50: Universal System")
st.info("💡 어떤 사진이든 640px 기반으로 자동 최적화하여 분석합니다.")

@st.cache_resource
def load_models():
    return YOLO(MODEL_REAL_PATH), YOLO(MODEL_SYM_PATH)

model_real, model_sym = load_models()

col1, col2 = st.columns(2)
ref_file = col1.file_uploader("1. 회로도(Schematic)", type=['jpg', 'png', 'jpeg'])
tgt_file = col2.file_uploader("2. 실물(Real Board)", type=['jpg', 'png', 'jpeg'])

if ref_file and tgt_file:
    if st.button("🚀 범용 정밀 분석 시작"):
        # PIL 이미지를 BGR 넘파이 배열로 변환
        ref_cv = cv2.cvtColor(np.array(Image.open(ref_file)), cv2.COLOR_RGB2BGR)
        tgt_cv = cv2.cvtColor(np.array(Image.open(tgt_file)), cv2.COLOR_RGB2BGR)

        res_ref, data_ref = analyze_universal(ref_cv, model_sym, is_schematic=True)
        res_tgt, data_tgt = analyze_universal(tgt_cv, model_real, is_schematic=False)

        st.divider()
        # 비교 결과 출력
        all_parts = set(data_ref.keys()) | set(data_tgt.keys())
        for p in sorted(all_parts):
            r, t = data_ref.get(p, 0), data_tgt.get(p, 0)
            if r == t: st.success(f"✅ {p.upper()} 일치: {r}개")
            else: st.error(f"⚠️ {p.upper()} 불일치: 회로도 {r}개 vs 실물 {t}개")

        st.image(cv2.cvtColor(res_ref, cv2.COLOR_BGR2RGB), caption="분석 결과 (회로도)", use_column_width=True)
        st.image(cv2.cvtColor(res_tgt, cv2.COLOR_BGR2RGB), caption="분석 결과 (실물)", use_column_width=True)
