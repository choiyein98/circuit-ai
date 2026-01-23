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
CONNECTION_THRESHOLD = 100       # 연결 감지 거리 (픽셀)

# ==========================================
# [2. 강력한 중복 제거 함수 (박스 안에 박스 제거)]
# ==========================================
def solve_overlap(parts, overlap_thresh=0.5):
    """
    기능: 겹치거나 포함된 박스를 강력하게 제거 (NMS)
    - overlap_thresh: 겹치는 비율이 이보다 높으면 중복으로 간주
    """
    if not parts: return []
    
    # 1. 신뢰도(conf) 높은 순서대로 정렬 (중요)
    parts.sort(key=lambda x: x['conf'], reverse=True)
    
    final = []
    for curr in parts:
        is_dup = False
        for kept in final:
            # 두 박스의 교집합(Intersection) 영역 계산
            x1 = max(curr['box'][0], kept['box'][0])
            y1 = max(curr['box'][1], kept['box'][1])
            x2 = min(curr['box'][2], kept['box'][2])
            y2 = min(curr['box'][3], kept['box'][3])
            
            inter_w = max(0, x2 - x1)
            inter_h = max(0, y2 - y1)
            inter_area = inter_w * inter_h
            
            if inter_area > 0:
                # 각 박스의 넓이
                area_curr = (curr['box'][2]-curr['box'][0]) * (curr['box'][3]-curr['box'][1])
                area_kept = (kept['box'][2]-kept['box'][0]) * (kept['box'][3]-kept['box'][1])
                
                # [핵심 로직] "작은 박스가 큰 박스 안에 포함되었는지" 확인
                # 교집합 영역이 작은 박스 넓이의 50% 이상을 차지하면 중복으로 간주
                min_area = min(area_curr, area_kept)
                overlap_ratio = inter_area / min_area
                
                if overlap_ratio > overlap_thresh:
                    is_dup = True
                    break
        
        if not is_dup:
            final.append(curr)
            
    return final

def get_center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

# ==========================================
# [3. 회로도 분석 (정상화)]
# ==========================================
def analyze_schematic(img, model):
    # [수정] 신뢰도를 0.20으로 올려서 노이즈 제거 (너무 낮추면 박스가 난무함)
    res = model.predict(source=img, conf=0.20, verbose=False)
    
    raw = []
    for b in res[0].boxes:
        raw.append({
            'name': model.names[int(b.cls[0])].lower(), 
            'box': b.xyxy[0].tolist(), 
            'center': get_center(b.xyxy[0].tolist()),
            'conf': float(b.conf[0])
        })
    
    # [수정] 강력한 중복 제거 실행 (겹침 허용치 0.1 -> 조금만 겹쳐도, 혹은 포함되면 제거)
    clean = solve_overlap(raw, overlap_thresh=0.1)
    
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
# [4. 실물 분석 (기존 로직 유지)]
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
        
        if any(x in name for x in ['pin', 'leg', 'lead']) and 'wire' not in name:
            pins.append(center) 
        elif 'breadboard' in name:
            continue
        else:
            bodies.append({'name': name, 'box': coords, 'center': center, 'conf': conf, 'is_on': False})

    # 실물은 기존 방식대로 중복 제거
    clean_bodies = solve_overlap(bodies, overlap_thresh=0.3)
    
    # 전원 확인
    power_active = any(p[1] < h * 0.45 for p in pins)
    if not power_active:
         for b in clean_bodies:
            if 'wire' in b['name'] and b['center'][1] < h * 0.45:
                power_active = True; break
    
    # 연결 확인
    if power_active:
        # 직접 연결
        for comp in clean_bodies:
            cy = comp['center'][1]
            if cy < h*0.48 or cy > h*0.52: comp['is_on'] = True

        # 전파 (Propagation)
        for _ in range(2): 
            for comp in clean_bodies:
                if comp['is_on']: continue 
                cx, cy = comp['center']
                for other in clean_bodies:
                    if not other['is_on']: continue
                    ocx, ocy = other['center']
                    dist = math.sqrt((cx-ocx)**2 + (cy-ocy)**2)
                    if dist < CONNECTION_THRESHOLD:
                        comp['is_on'] = True; break
                
                if not comp['is_on']:
                    for px, py in pins:
                        if math.sqrt((cx-px)**2 + (cy-py)**2) < CONNECTION_THRESHOLD:
                             if py < h*0.48 or py > h*0.52:
                                comp['is_on'] = True; break

    off_count = 0
    
    for comp in clean_bodies:
        is_on = comp['is_on']
        if is_on:
            color = (0, 255, 0) # ON
            status = "ON"
        else:
            color = (0, 0, 255) # OFF
            status = "OFF"
            off_count += 1
        
        x1, y1, x2, y2 = map(int, comp['box'])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.putText(img, status, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
    return img, {'off': off_count, 'total': len(clean_bodies), 'details': {}}

# ==========================================
# [5. 메인 UI]
# ==========================================
st.title("🧠 BrainBoard V5: Final Fix")
st.markdown("### 회로도 중복 인식 문제 해결됨")

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
            st.image(cv2.cvtColor(res_ref_img, cv2.COLOR_BGR2RGB), caption="회로도 분석 (깔끔하게 보정됨)", use_column_width=True)
            st.image(cv2.cvtColor(res_tgt_img, cv2.COLOR_BGR2RGB), caption=f"실물 분석 (OFF 개수: {tgt_data['off']})", use_column_width=True)
            
            if tgt_data['off'] == 0:
                st.success("✅ 모든 부품 전원 연결 확인됨 (All ON)")
            else:
                st.error(f"❌ {tgt_data['off']}개 부품이 연결되지 않았습니다 (OFF)")
