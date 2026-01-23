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
# 연결 감지 거리 (픽셀) - 와이어와 부품이 이 거리 안에 있으면 연결된 것으로 간주
CONNECTION_THRESHOLD = 100       

# ==========================================
# [2. 유틸리티 함수 (중복 제거 및 좌표 계산)]
# ==========================================
def calculate_iou(box1, box2):
    """두 박스의 겹치는 비율(IoU) 계산"""
    x1, y1, x2, y2 = max(box1[0], box2[0]), max(box1[1], box2[1]), min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

def solve_overlap(parts, dist_thresh=30, iou_thresh=0.3):
    """
    기능: 겹치는 박스들을 정리 (거리 + IoU 기준)
    """
    if not parts: return []
    # 신뢰도 높은 순 정렬
    if 'conf' in parts[0]:
        parts.sort(key=lambda x: x.get('conf', 0), reverse=True)
    
    final = []
    for curr in parts:
        is_dup = False
        for k in final:
            # 1. 중심점 거리 계산
            dist = math.sqrt((curr['center'][0]-k['center'][0])**2 + (curr['center'][1]-k['center'][1])**2)
            # 2. 겹치는 면적 계산 (IoU)
            iou = calculate_iou(curr['box'], k['box'])
            
            # 거리가 매우 가깝거나, 면적이 많이 겹치면 중복으로 간주
            if dist < dist_thresh or iou > iou_thresh:
                is_dup = True; break
        if not is_dup:
            final.append(curr)
    return final

def get_center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

# ==========================================
# [3. 회로도 분석 (오인식 감소를 위해 conf 상향 조정)]
# ==========================================
def analyze_schematic(img, model):
    # [수정 핵심] 엉뚱한 커패시터 인식을 막기 위해 신뢰도(conf)를 0.05 -> 0.25로 상향
    # 이 값을 높일수록 AI가 확실한 것만 잡습니다. (오인식 감소, 미인식 증가 가능성 있음)
    conf_threshold = 0.25 
    res = model.predict(source=img, conf=conf_threshold, verbose=False)
    
    raw = []
    for b in res[0].boxes:
        raw.append({
            'name': model.names[int(b.cls[0])].lower(), 
            'box': b.xyxy[0].tolist(), 
            'center': get_center(b.xyxy[0].tolist()),
            'conf': float(b.conf[0])
        })
    
    # 중복 제거 (거리 30px 또는 IoU 0.3 이상이면 제거)
    clean = solve_overlap(raw, dist_thresh=30, iou_thresh=0.3)
    
    for p in clean:
        name = p['name']
        # 위치 기반 이름 보정 (왼쪽=Source)
        if p['center'][0] < img.shape[1] * 0.25: 
            name = 'source'
        elif 'cap' in name: name = 'capacitor'
        elif 'res' in name: name = 'resistor'
        
        x1, y1, x2, y2 = map(int, p['box'])
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img, name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
    summary = {'total': len(clean), 'details': {}}
    for p in clean:
        summary['details'][p['name']] = summary['details'].get(p['name'], 0) + 1
        
    return img, summary

# ==========================================
# [4. 실물 분석 (변경 없음 - 기존 로직 유지)]
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
        
        # 핀 분류 (좌표용)
        if any(x in name for x in ['pin', 'leg', 'lead']) and 'wire' not in name:
            pins.append(center) 
        elif 'breadboard' in name:
            continue
        else:
            # bodies: 저항, 커패시터, 와이어 등 모든 부품
            bodies.append({'name': name, 'box': coords, 'center': center, 'conf': conf, 'is_on': False})

    clean_bodies = solve_overlap(bodies, 60)
    
    # [1단계] 전원 레일 활성화 확인
    power_active = any(p[1] < h * 0.45 for p in pins)
    if not power_active:
         for b in clean_bodies:
            if 'wire' in b['name'] and b['center'][1] < h * 0.45:
                power_active = True; break
    
    # [2단계] 연결 상태 판단 (전파 로직 적용)
    if power_active:
        # 1. 직접 연결
        for comp in clean_bodies:
            cy = comp['center'][1]
            if cy < h*0.48 or cy > h*0.52: 
                comp['is_on'] = True

        # 2. 간접 연결 (Propagation - 2회 반복)
        for _ in range(2): 
            for comp in clean_bodies:
                if comp['is_on']: continue 
                
                # 내 근처에 켜진 부품 확인
                cx, cy = comp['center']
                for other in clean_bodies:
                    if not other['is_on']: continue
                    ocx, ocy = other['center']
                    dist = math.sqrt((cx-ocx)**2 + (cy-ocy)**2)
                    if dist < CONNECTION_THRESHOLD:
                        comp['is_on'] = True
                        break
                
                # 내 근처에 전원 핀 확인
                if not comp['is_on']:
                    for px, py in pins:
                        if math.sqrt((cx-px)**2 + (cy-py)**2) < CONNECTION_THRESHOLD:
                             if py < h*0.48 or py > h*0.52:
                                comp['is_on'] = True; break

    off_count = 0
    
    # [3단계] 시각화
    for comp in clean_bodies:
        is_on = comp['is_on']
        
        if is_on:
            color = (0, 255, 0) # 초록 (ON)
            status = "ON"
        else:
            color = (0, 0, 255) # 빨강 (OFF)
            status = "OFF"
            off_count += 1
        
        x1, y1, x2, y2 = map(int, comp['box'])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.putText(img, status, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
    return img, {'off': off_count, 'total': len(clean_bodies), 'details': {}}

# ==========================================
# [5. 메인 UI (Streamlit)]
# ==========================================
st.title("🧠 BrainBoard V5: Circuit Check")
st.markdown("### 회로도 vs 실물 연결 상태(ON/OFF) 확인")

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
ref_file = col1.file_uploader("1. 회로도 업로드", type=['jpg', 'png', 'jpeg'])
tgt_file = col2.file_uploader("2. 실물 사진 업로드", type=['jpg', 'png', 'jpeg'])

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
            st.image(cv2.cvtColor(res_ref_img, cv2.COLOR_BGR2RGB), caption="회로도 분석", use_column_width=True)
            st.image(cv2.cvtColor(res_tgt_img, cv2.COLOR_BGR2RGB), caption=f"실물 분석 (OFF 개수: {tgt_data['off']})", use_column_width=True)
            
            if tgt_data['off'] == 0:
                st.success("✅ 모든 부품 전원 연결 확인됨 (All ON)")
            else:
                st.error(f"❌ {tgt_data['off']}개 부품이 연결되지 않았습니다 (OFF)")
