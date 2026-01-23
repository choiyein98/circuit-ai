import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import math
from PIL import Image

# ==========================================
# [설정 및 상수]
# ==========================================
st.set_page_config(page_title="BrainBoard V44", layout="wide")

MODEL_REAL_PATH = 'best.pt'    # 실물 보드용 모델
MODEL_SYM_PATH = 'symbol.pt'   # 회로도용 모델
PIN_SENSITIVITY = 140          # 핀과 부품 간 연결 감지 범위 (픽셀 단위)

# ==========================================
# [Helper Functions]
# ==========================================
def solve_overlap(parts, dist_thresh=60):
    """
    중복 감지된 객체들을 거리 기준으로 필터링 (Conf 높은 것 우선)
    """
    if not parts: return []
    if 'conf' in parts[0]:
        parts.sort(key=lambda x: x.get('conf', 0), reverse=True)
    
    final = []
    for curr in parts:
        if not any(math.sqrt((curr['center'][0]-k['center'][0])**2 + (curr['center'][1]-k['center'][1])**2) < dist_thresh for k in final):
            final.append(curr)
    return final

# ==========================================
# [분석 함수 1: 회로도 (Schematic)]
# ==========================================
def analyze_schematic(img, model):
    # Streamlit에서는 이미지를 numpy array로 바로 받으므로 imread 삭제
    
    # 모델 추론
    res = model.predict(source=img, conf=0.15, verbose=False)
    
    raw = []
    for b in res[0].boxes:
        raw.append({
            'name': model.names[int(b.cls[0])].lower(), 
            'box': b.xyxy[0].tolist(), 
            'center': ((b.xyxy[0][0]+b.xyxy[0][2])/2, (b.xyxy[0][1]+b.xyxy[0][3])/2),
            'conf': float(b.conf[0])
        })
    
    clean = solve_overlap(raw)
    
    for p in clean:
        name = p['name']
        if p['center'][0] < img.shape[1] * 0.25: name = 'source'
        elif 'cap' in name: name = 'capacitor'
        elif 'res' in name: name = 'resistor'
        
        x1, y1, x2, y2 = map(int, p['box'])
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img, name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    return img

# ==========================================
# [분석 함수 2: 실물 (Real Board)]
# ==========================================
def analyze_real(img, model):
    # Streamlit에서는 이미지를 numpy array로 바로 받으므로 imread 삭제
    h, w, _ = img.shape
    
    # 모델 추론
    res = model.predict(source=img, conf=0.1, verbose=False)
    
    bodies = [] 
    pins = []   
    
    for b in res[0].boxes:
        name = model.names[int(b.cls[0])].lower()
        coords = b.xyxy[0].tolist()
        center = ((coords[0]+coords[2])/2, (coords[1]+coords[3])/2)
        conf = float(b.conf[0])
        
        # [Wire 및 Pin 분류 로직]
        if any(x in name for x in ['pin', 'leg', 'lead']) and 'wire' not in name:
            pins.append(center)
        elif 'breadboard' in name:
            continue
        else:
            bodies.append({'name': name, 'box': coords, 'center': center, 'conf': conf})

    clean_bodies = solve_overlap(bodies, 60)
    
    # [전원 활성화 로직]
    power_active = any(p[1] < h * 0.45 for p in pins)
    
    if not power_active:
        for b in clean_bodies:
            if 'wire' in b['name'] and b['center'][1] < h * 0.45:
                power_active = True
                break
    
    off_count = 0
    
    for comp in clean_bodies:
        cx, cy = comp['center']
        name = comp['name']
        is_on = False
        
        if 'wire' in name:
            color = (0, 165, 255) # 주황색 (OpenCV는 BGR)
            status = "WIRE"
            is_on = True 
        else:
            if power_active:
                if cy < h*0.48 or cy > h*0.52: 
                    is_on = True
                else:
                    for px, py in pins:
                        if math.sqrt((cx-px)**2 + (cy-py)**2) < PIN_SENSITIVITY:
                            if py < h*0.48 or py > h*0.52:
                                is_on = True; break
            
            if is_on:
                color = (0, 255, 0) # 초록 (ON)
                status = "ON"
            else:
                color = (0, 0, 255) # 빨강 (OFF)
                status = "OFF"
                off_count += 1
        
        x1, y1, x2, y2 = map(int, comp['box'])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.putText(img, status, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
    return img, off_count

# ==========================================
# [WEB APP UI] Streamlit Main Code
# ==========================================
st.title("🧠 BrainBoard V44: AI Circuit Verifier")
st.markdown("### PSpice 회로도와 실제 브레드보드 사진을 업로드하세요.")

@st.cache_resource
def load_models():
    return YOLO(MODEL_REAL_PATH), YOLO(MODEL_SYM_PATH)

try:
    model_real, model_sym = load_models()
    st.sidebar.success("✅ AI 모델 로드 완료!")
except Exception as e:
    st.error(f"모델 파일을 찾을 수 없습니다: {e}")
    st.stop()

col1, col2 = st.columns(2)
ref_file = col1.file_uploader("1. 회로도(Schematic) 업로드", type=['jpg', 'png', 'jpeg'])
tgt_file = col2.file_uploader("2. 실물(Real Board) 업로드", type=['jpg', 'png', 'jpeg'])

if ref_file and tgt_file:
    # 파일 업로더 객체를 OpenCV 이미지로 변환
    ref_image = Image.open(ref_file)
    tgt_image = Image.open(tgt_file)
    ref_cv = cv2.cvtColor(np.array(ref_image), cv2.COLOR_RGB2BGR)
    tgt_cv = cv2.cvtColor(np.array(tgt_image), cv2.COLOR_RGB2BGR)

    if st.button("🚀 회로 검증 시작 (Analyze)"):
        with st.spinner("AI가 회로를 분석 중입니다..."):
            # 이미지 경로 대신 이미지 배열 자체를 전달
            res_ref_img = analyze_schematic(ref_cv.copy(), model_sym)
            res_tgt_img, off_count = analyze_real(tgt_cv.copy(), model_real)

            st.divider()
            
            # 결과 텍스트 출력
            if off_count == 0:
                st.success("🎉 Perfect! 모든 부품이 정상적으로 연결되었습니다.")
            else:
                st.error(f"❌ 오류 발견: {off_count}개의 부품이 연결되지 않았거나(OFF) 비정상입니다.")
                st.warning("팁: 전원 연결 상태와 핀이 브레드보드에 깊게 꽂혔는지 확인하세요.")

            # 결과 이미지 출력
            st.image(cv2.cvtColor(res_ref_img, cv2.COLOR_BGR2RGB), caption="PSpice 회로도 분석", use_container_width=True)
            st.image(cv2.cvtColor(res_tgt_img, cv2.COLOR_BGR2RGB), caption=f"실물 보드 분석 (비정상 부품: {off_count})", use_container_width=True)
