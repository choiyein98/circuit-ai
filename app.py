import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import math
from PIL import Image

# ==========================================
# [1. 설정 및 모델 선택 기능]
# ==========================================
st.set_page_config(page_title="BrainBoard V55: Model Switcher", layout="wide")

# 깃허브에 올린 실물 모델 리스트 (파일명과 정확히 일치해야 함)
REAL_MODEL_LIST = ['best(3).pt', 'best(2).pt', 'best.pt']
selected_real = st.sidebar.selectbox("🔬 실물 분석 모델 선택", REAL_MODEL_LIST)

MODEL_SYM_PATH = 'symbol.pt' # 회로도는 하나로 고정

# [범용 임계값]
REAL_CONF_LIMIT = 0.70  # 너무 높으면 놓치고, 낮으면 노이즈가 생기니 0.70 추천
REAL_IOU_LIMIT = 0.1   # 중복 박스 제거 강도 (독하게!)

# ==========================================
# [2. 범용 엔진]
# ==========================================
def solve_overlap_universal(parts, iou_thresh=0.1, is_schematic=False):
    if not parts: return []
    parts.sort(key=lambda x: x.get('conf', 0), reverse=True)
    
    final = []
    for curr in parts:
        is_dup = False
        for k in final:
            # IoU 계산
            x1, y1, x2, y2 = max(curr['box'][0], k['box'][0]), max(curr['box'][1], k['box'][1]), \
                             min(curr['box'][2], k['box'][2]), min(curr['box'][3], k['box'][3])
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            area1 = (curr['box'][2]-curr['box'][0]) * (curr['box'][3]-curr['box'][1])
            area2 = (k['box'][2]-k['box'][0]) * (k['box'][3]-k['box'][1])
            iou = inter / (area1 + area2 - inter) if (area1 + area2 - inter) > 0 else 0
            
            # 중심 거리 계산
            dist = math.sqrt(((curr['box'][0]+curr['box'][2])/2 - (k['box'][0]+k['box'][2])/2)**2 + 
                             ((curr['box'][1]+curr['box'][3])/2 - (k['box'][1]+k['box'][3])/2)**2)
            
            # 회로도는 좁게, 실물은 넓게 중복 제거
            dist_limit = 30 if is_schematic else 80
            if iou > iou_thresh or dist < dist_limit:
                is_dup = True; break
        if not is_dup: final.append(curr)
    return final

def analyze_engine(img, model, is_schematic=False):
    # imgsz=640으로 리사이즈하여 범용성 확보
    res = model.predict(source=img, conf=0.25 if is_schematic else REAL_CONF_LIMIT, imgsz=640, verbose=False)
    
    raw = []
    for b in res[0].boxes:
        name = model.names[int(b.cls[0])].lower()
        coords = b.xyxy[0].tolist()
        
        # 이름 표준화 (사용자 요청: Body만 카운트)
        if any(x in name for x in ['res', 'r']): norm_name = 'RESISTOR'
        elif any(x in name for x in ['cap', 'c']): norm_name = 'CAPACITOR'
        elif any(x in name for x in ['v', 'volt', 'batt', 'source', 'vdc']): norm_name = 'SOURCE'
        else: continue # PIN, WIRE 등은 카운트에서 제외
        
        raw.append({'name': norm_name, 'box': coords, 'conf': float(b.conf[0])})

    clean = solve_overlap_universal(raw, is_schematic=is_schematic)
    
    # 회로도 전원 자동 보정
    if is_schematic and clean and not any(p['name'] == 'SOURCE' for p in clean):
        min(clean, key=lambda p: p['box'][0])['name'] = 'SOURCE'

    summary = {}
    for p in clean:
        x1, y1, x2, y2 = map(int, p['box'])
        color = (255, 0, 0) if is_schematic else (0, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.putText(img, p['name'], (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        summary[p['name']] = summary.get(p['name'], 0) + 1
        
    return img, summary

# ==========================================
# [3. 메인 UI]
# ==========================================
st.title(f"🧠 BrainBoard V55: Multi-Model Sync")
st.sidebar.info(f"현재 사용 모델: {selected_real}")

@st.cache_resource
def load_models(real_p, sym_p):
    return YOLO(real_p), YOLO(sym_p)

try:
    model_real, model_sym = load_models(selected_real, MODEL_SYM_PATH)
    st.sidebar.success("✅ 모델 로드 성공")
except Exception as e:
    st.error(f"모델 파일을 찾을 수 없습니다. (깃허브 파일명 확인 필요): {e}")
    st.stop()

col1, col2 = st.columns(2)
ref_file = col1.file_uploader("1. 회로도(Schematic) 업로드", type=['jpg', 'png', 'jpeg'])
tgt_file = col2.file_uploader("2. 실물(Real Board) 업로드", type=['jpg', 'png', 'jpeg'])

if ref_file and tgt_file:
    if st.button("🚀 전체 회로 정밀 매칭 시작"):
        ref_cv = cv2.cvtColor(np.array(Image.open(ref_file)), cv2.COLOR_RGB2BGR)
        tgt_cv = cv2.cvtColor(np.array(Image.open(tgt_file)), cv2.COLOR_RGB2BGR)

        res_ref, data_ref = analyze_engine(ref_cv, model_sym, is_schematic=True)
        res_tgt, data_tgt = analyze_engine(tgt_cv, model_real, is_schematic=False)

        st.divider()
        st.subheader("📊 부품 일치 통계")
        all_parts = set(data_ref.keys()) | set(data_tgt.keys())
        for p in sorted(all_parts):
            r, t = data_ref.get(p, 0), data_tgt.get(p, 0)
            if r == t: st.success(f"✅ {p}: 회로도 {r}개 / 실물 {t}개 (일치)")
            else: st.error(f"⚠️ {p}: 회로도 {r}개 / 실물 {t}개 (불일치)")

        st.image(cv2.cvtColor(res_ref, cv2.COLOR_BGR2RGB), caption=f"회로도 분석 (symbol.pt)", use_column_width=True)
        st.image(cv2.cvtColor(res_tgt, cv2.COLOR_BGR2RGB), caption=f"실물 분석 ({selected_real})", use_column_width=True)
