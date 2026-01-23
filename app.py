import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import math
from PIL import Image

# ==========================================
# [1. 설정 및 라이브러리]
# ==========================================
st.set_page_config(page_title="BrainBoard V5 Final", layout="wide")

MODEL_REAL_PATH = 'best.pt'      # 실물 브레드보드 분석용
MODEL_SYM_PATH = 'symbol.pt'     # 회로도 기호 분석용
PIN_SENSITIVITY = 140            # 핀과 부품 간의 거리 허용 오차 (픽셀)

# ==========================================
# [2. 중복 제거 함수 (회로도 인식률 개선용 수정)]
# ==========================================
def solve_overlap(parts, dist_thresh=30): # 거리 기준을 60->30으로 줄여서 붙어있는 부품 보존
    """
    기능: 겹치는 박스들을 정리 (회로도에서 부품이 사라지는 문제 해결을 위해 튜닝됨)
    """
    if not parts: return []
    # 신뢰도 높은 순 정렬
    if 'conf' in parts[0]:
        parts.sort(key=lambda x: x.get('conf', 0), reverse=True)
    
    final = []
    for curr in parts:
        is_dup = False
        for k in final:
            # 중심점 거리 계산
            dist = math.sqrt((curr['center'][0]-k['center'][0])**2 + (curr['center'][1]-k['center'][1])**2)
            if dist < dist_thresh: # 너무 가까우면 중복으로 간주
                is_dup = True; break
        if not is_dup:
            final.append(curr)
    return final

def get_center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

# ==========================================
# [3. 회로도 분석 (인식률 대폭 강화)]
# ==========================================
def analyze_schematic(img, model):
    # [수정] conf=0.05로 낮춰서 희미하거나 작은 부품도 강제로 잡게 함
    res = model.predict(source=img, conf=0.05, verbose=False)
    
    raw = []
    for b in res[0].boxes:
        raw.append({
            'name': model.names[int(b.cls[0])].lower(), 
            'box': b.xyxy[0].tolist(), 
            'center': get_center(b.xyxy[0].tolist()),
            'conf': float(b.conf[0])
        })
    
    # 중복 제거 실행 (기준을 완화했으므로 더 많이 남음)
    clean = solve_overlap(raw, dist_thresh=30)
    
    for p in clean:
        name = p['name']
        # 위치 기반 이름 보정 (왼쪽=Source)
        if p['center'][0] < img.shape[1] * 0.25: 
            name = 'source'
        elif 'cap' in name: name = 'capacitor'
        elif 'res' in name: name = 'resistor'
        
        # 시각화
        x1, y1, x2, y2 = map(int, p['box'])
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img, name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
    summary = {'total': len(clean), 'details': {}}
    for p in clean:
        summary['details'][p['name']] = summary['details'].get(p['name'], 0) + 1
        
    return img, summary

# ==========================================
# [4. 실물 분석 (ON/OFF 만 깔끔하게 표시)]
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
        
        # 1. 핀/다리/와이어 분류 (좌표 계산용으로만 사용, 화면엔 안 그림)
        # 'wire'는 전원 연결 확인용 핀으로 간주 (시각화 대상에서는 제외할 수도, 포함할 수도 있음)
        # 요청: 온오프 여부만 인식 -> wire도 부품으로 쳐서 ON 표시 해줌
        if any(x in name for x in ['pin', 'leg', 'lead']) and 'wire' not in name:
            pins.append(center) 
        elif 'breadboard' in name:
            continue
        else:
            # bodies: 시각화 대상 (저항, 커패시터, 와이어 등)
            bodies.append({'name': name, 'box': coords, 'center': center, 'conf': conf})

    # 중복 제거
    clean_bodies = solve_overlap(bodies, 60)
    
    # [V5 로직] 전원 공급 확인
    # 상단 45% 지점에 핀이 있거나, 와이어가 있으면 전원 ON
    power_active = any(p[1] < h * 0.45 for p in pins)
    if not power_active:
         for b in clean_bodies:
            if 'wire' in b['name'] and b['center'][1] < h * 0.45:
                power_active = True; break
    
    off_count = 0
    
    # [V5 로직] 연결 상태 판단 & 시각화
    for comp in clean_bodies:
        cx, cy = comp['center']
        name = comp['name']
        is_on = False
        
        # 와이어는 항상 ON (주황색)
        if 'wire' in name:
            is_on = True
            color = (0, 165, 255) # 주황색
            status = "WIRE"
        else:
            # 일반 부품 ON/OFF 판단
            if power_active:
                # 1. 직접 연결 (상단/하단 레일 영역)
                if cy < h*0.48 or cy > h*0.52:
                    is_on = True
                else:
                    # 2. 간접 연결 (핀과 거리 140px 이내)
                    for px, py in pins:
                        if math.sqrt((cx-px)**2 + (cy-py)**2) < PIN_SENSITIVITY:
                            # 그 핀이 전원 영역에 있어야 함
                            if py < h*0.48 or py > h*0.52:
                                is_on = True; break
            
            if is_on:
                color = (0, 255, 0) # 초록 (ON)
                status = "ON"
            else:
                color = (0, 0, 255) # 빨강 (OFF)
                status = "OFF"
                off_count += 1
        
        # [요청 반영] 다리/점/선 그리기 삭제 -> 오직 박스와 텍스트만 표시
        x1, y1, x2, y2 = map(int, comp['box'])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.putText(img, status, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
    return img, {'off': off_count, 'total': len(clean_bodies), 'details': {}}

# ==========================================
# [5. 메인 UI (Streamlit)]
# ==========================================
st.title("🧠 BrainBoard V5: Simple Check")
st.markdown("### 회로도(부품 인식 강화) vs 실물(ON/OFF 단순 확인)")

@st.cache_resource
def load_models():
    return YOLO(MODEL_REAL_PATH), YOLO(MODEL_SYM_PATH)

try:
    model_real, model_sym = load_models()
    st.sidebar.success("✅ 모델 로드 성공")
except Exception as e:
    st.error(f"모델 로드 실패: {e}")
    st.stop()

col1, col2 = st.columns(2)
ref_file = col1.file_uploader("1. 회로도 업로드", type=['jpg', 'png'])
tgt_file = col2.file_uploader("2. 실물 사진 업로드", type=['jpg', 'png'])

if ref_file and tgt_file:
    ref_image = Image.open(ref_file)
    tgt_image = Image.open(tgt_file)
    
    ref_cv = cv2.cvtColor(np.array(ref_image), cv2.COLOR_RGB2BGR)
    tgt_cv = cv2.cvtColor(np.array(tgt_image), cv2.COLOR_RGB2BGR)

    if st.button("🚀 분석 실행"):
        with st.spinner("분석 중..."):
            res_ref_img, ref_data = analyze_schematic(ref_cv.copy(), model_sym)
            res_tgt_img, tgt_data = analyze_real(tgt_cv.copy(), model_real)

            st.divider()
            
            # 결과 이미지 출력
            st.image(cv2.cvtColor(res_ref_img, cv2.COLOR_BGR2RGB), caption="회로도 분석 (인식률 Up)", use_column_width=True)
            st.image(cv2.cvtColor(res_tgt_img, cv2.COLOR_BGR2RGB), caption=f"실물 분석 (OFF: {tgt_data['off']}개)", use_column_width=True)
            
            if tgt_data['off'] == 0:
                st.success("✅ 모든 부품 전원 연결 확인됨 (ON)")
            else:
                st.error(f"❌ {tgt_data['off']}개 부품 연결 확인 필요 (OFF)")
