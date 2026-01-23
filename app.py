import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import math
from PIL import Image

# ==========================================
# [1. 설정 및 라이브러리] - 요청하신 조건 반영
# ==========================================
st.set_page_config(page_title="BrainBoard V5 (Distance Logic)", layout="wide")

MODEL_REAL_PATH = 'best.pt'      # 실물 브레드보드 분석용
MODEL_SYM_PATH = 'symbol.pt'     # 회로도 기호 분석용
PIN_SENSITIVITY = 140            # 핀과 부품 간의 거리 허용 오차 (픽셀 단위)

# ==========================================
# [2. solve_overlap (중복 제거 함수)]
# ==========================================
def solve_overlap(parts, dist_thresh=60):
    """
    기능: YOLO가 겹치는 박스들을 정리하는 함수
    로직: conf 높은 순 정렬 -> 중심점 거리 계산 -> 중복 제거
    """
    if not parts: return []
    # 신뢰도(conf)가 높은 순서대로 정렬 (딕셔너리에 conf가 있는 경우)
    if 'conf' in parts[0]:
        parts.sort(key=lambda x: x.get('conf', 0), reverse=True)
    
    final = []
    for curr in parts:
        # 현재 박스와 이미 선택된 박스들의 중심점 거리를 계산
        is_dup = False
        for k in final:
            dist = math.sqrt((curr['center'][0]-k['center'][0])**2 + (curr['center'][1]-k['center'][1])**2)
            if dist < dist_thresh: # 거리가 가까우면 중복 간주
                is_dup = True; break
        if not is_dup:
            final.append(curr)
    return final

def get_center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

# ==========================================
# [3. analyze_schematic (회로도 분석 함수)]
# ==========================================
def analyze_schematic(img, model):
    # 이미지를 읽고 YOLO 추론
    res = model.predict(source=img, conf=0.15, verbose=False)
    
    raw = []
    for b in res[0].boxes:
        raw.append({
            'name': model.names[int(b.cls[0])].lower(), 
            'box': b.xyxy[0].tolist(), 
            'center': get_center(b.xyxy[0].tolist()),
            'conf': float(b.conf[0])
        })
    
    # 중복 제거
    clean = solve_overlap(raw)
    
    for p in clean:
        name = p['name']
        # 위치 기반 이름 보정: 왼쪽 25% 영역은 'source'로 강제 변경
        if p['center'][0] < img.shape[1] * 0.25: 
            name = 'source'
        elif 'cap' in name: name = 'capacitor'
        elif 'res' in name: name = 'resistor'
        
        # 파란색 박스와 이름 그리기
        x1, y1, x2, y2 = map(int, p['box'])
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img, name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
    # 요약 정보 생성
    summary = {'total': len(clean), 'details': {}}
    for p in clean:
        summary['details'][p['name']] = summary['details'].get(p['name'], 0) + 1
        
    return img, summary

# ==========================================
# [4. analyze_real (실물 회로 분석 함수)] - 핵심 로직
# ==========================================
def analyze_real(img, model):
    h, w, _ = img.shape
    res = model.predict(source=img, conf=0.1, verbose=False)
    
    bodies = []
    pins = []
    
    for b in res[0].boxes:
        name = model.names[int(b.cls[0])].lower()
        coords = b.xyxy[0].tolist()
        center = get_center(coords)
        conf = float(b.conf[0])
        
        # 객체 분류
        # pins: pin, leg, lead가 포함된 객체 (단, wire는 제외하여 시각화 가능하게 함)
        if any(x in name for x in ['pin', 'leg', 'lead']) and 'wire' not in name:
            pins.append(center) # 좌표만 저장
        elif 'breadboard' in name:
            continue
        else:
            # bodies: 그 외 부품들 (저항, 커패시터, 와이어 등)
            # [개선 제안 반영]: wire도 bodies에 포함시켜 화면에 그리도록 함
            bodies.append({'name': name, 'box': coords, 'center': center, 'conf': conf})

    clean_bodies = solve_overlap(bodies, 60)
    
    # 전원 공급 확인 (power_active)
    # h * 0.45: 이미지 상단 45% 지점에 핀이 있는지 확인
    # 추가로: 상단에 위치한 'wire'도 전원 공급원으로 간주 (로직 보강)
    power_active = any(p[1] < h * 0.45 for p in pins)
    if not power_active:
         for b in clean_bodies:
            if 'wire' in b['name'] and b['center'][1] < h * 0.45:
                power_active = True; break
    
    off_count = 0
    
    # 연결 상태 판단 (ON/OFF)
    for comp in clean_bodies:
        cx, cy = comp['center']
        name = comp['name']
        is_on = False
        
        if 'wire' in name:
            # 와이어는 항상 ON (주황색)으로 표시 (개선 사항)
            is_on = True
            color = (0, 165, 255) # 주황색
            status = "WIRE"
        else:
            if power_active:
                # 1. 직접 연결: 부품 중심이 전원 레일 영역(중앙 제외 상하단)에 위치
                # (중앙 분리대를 h*0.48 ~ h*0.52로 가정)
                if cy < h*0.48 or cy > h*0.52:
                    is_on = True
                else:
                    # 2. 간접 연결: 부품 근처(PIN_SENSITIVITY 이내)에 있는 핀이 전원 영역에 있을 때
                    for px, py in pins:
                        if math.sqrt((cx-px)**2 + (cy-py)**2) < PIN_SENSITIVITY:
                            if py < h*0.48 or py > h*0.52:
                                is_on = True; break
            
            if is_on:
                color = (0, 255, 0) # ON (초록색)
                status = "ON"
            else:
                color = (0, 0, 255) # OFF (빨간색)
                status = "OFF"
                off_count += 1
        
        # 박스와 텍스트 그리기
        x1, y1, x2, y2 = map(int, comp['box'])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.putText(img, status, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
    return img, {'off': off_count, 'total': len(clean_bodies), 'details': {}}

# ==========================================
# [5. 메인 실행부 (Streamlit 변환)]
# ==========================================
# Tkinter 대신 Streamlit 사용 (웹 환경 호환)

st.title("🧠 BrainBoard V5: Simple Distance Logic")
st.markdown("### 요청하신 명세서(V5) 로직으로 분석합니다.")

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
    ref_image = Image.open(ref_file)
    tgt_image = Image.open(tgt_file)
    
    ref_cv = cv2.cvtColor(np.array(ref_image), cv2.COLOR_RGB2BGR)
    tgt_cv = cv2.cvtColor(np.array(tgt_image), cv2.COLOR_RGB2BGR)

    if st.button("🚀 분석 실행 (Distance Mode)"):
        with st.spinner("분석 중..."):
            res_ref_img, ref_data = analyze_schematic(ref_cv.copy(), model_sym)
            res_tgt_img, tgt_data = analyze_real(tgt_cv.copy(), model_real)

            st.divider()
            
            # 결과 병합 및 출력
            # 설명에 있는 "해상도가 너무 크면 리사이징" 로직은 Streamlit이 알아서 처리하므로 생략 가능하나
            # 명시적으로 보여주기 위해 컬럼으로 나눔
            
            st.image(cv2.cvtColor(res_ref_img, cv2.COLOR_BGR2RGB), caption="회로도 분석 결과", use_column_width=True)
            st.image(cv2.cvtColor(res_tgt_img, cv2.COLOR_BGR2RGB), caption=f"실물 분석 결과 (OFF: {tgt_data['off']}개)", use_column_width=True)
            
            if tgt_data['off'] == 0:
                st.success("✅ 모든 부품 연결 성공 (ON)")
            else:
                st.error(f"❌ {tgt_data['off']}개의 부품이 연결되지 않았습니다 (OFF)")
