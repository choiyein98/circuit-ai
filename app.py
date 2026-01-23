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
# [2. 유틸리티 함수 (NMS: 중복 제거)]
# ==========================================
def calculate_iou(box1, box2):
    """두 박스의 겹치는 비율(IoU) 계산"""
    x1, y1, x2, y2 = max(box1[0], box2[0]), max(box1[1], box2[1]), min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

def solve_overlap(parts, dist_thresh=0, iou_thresh=0.5):
    """
    기능: 겹치는 박스들을 정리
    """
    if not parts: return []
    if 'conf' in parts[0]:
        parts.sort(key=lambda x: x.get('conf', 0), reverse=True)
    
    final = []
    for curr in parts:
        is_dup = False
        for k in final:
            iou = calculate_iou(curr['box'], k['box'])
            if iou > iou_thresh:
                is_dup = True; break
            
            if dist_thresh > 0:
                dist = math.sqrt((curr['center'][0]-k['center'][0])**2 + (curr['center'][1]-k['center'][1])**2)
                if dist < dist_thresh:
                    is_dup = True; break
                    
        if not is_dup:
            final.append(curr)
    return final

def get_center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

# ==========================================
# [3. 회로도 분석 (놓치는 부품 방지)]
# ==========================================
def analyze_schematic(img, model):
    # [수정] 아주 낮은 신뢰도(0.01)로 설정하여 부품을 절대 놓치지 않게 함
    # 대신 중복 제거(NMS)가 중요해짐
    res = model.predict(source=img, conf=0.01, verbose=False)
    
    raw = []
    for b in res[0].boxes:
        raw.append({
            'name': model.names[int(b.cls[0])].lower(), 
            'box': b.xyxy[0].tolist(), 
            'center': get_center(b.xyxy[0].tolist()),
            'conf': float(b.conf[0])
        })
    
    # 중복 제거 (IoU 0.4 기준)
    clean = solve_overlap(raw, dist_thresh=0, iou_thresh=0.4)
    
    for p in clean:
        name = p['name']
        if p['center'][0] < img.shape[1] * 0.25: 
            name = 'source'
        elif 'cap' in name: name = 'capacitor'
        elif 'res' in name: name = 'resistor'
        
        # 이름 정규화 (비교를 위해)
        p['normalized_name'] = name

        x1, y1, x2, y2 = map(int, p['box'])
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img, name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
    summary = {'total': len(clean), 'details': {}}
    for p in clean:
        n = p['normalized_name']
        summary['details'][n] = summary['details'].get(n, 0) + 1
        
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
            # 이름 정규화
            norm_name = name
            if 'cap' in name: norm_name = 'capacitor'
            elif 'res' in name: norm_name = 'resistor'
            elif 'wire' in name: norm_name = 'wire' # 와이어는 별도 카운트 안 함(보통)
            
            bodies.append({'name': name, 'normalized_name': norm_name, 'box': coords, 'center': center, 'conf': conf, 'is_on': False})

    clean_bodies = solve_overlap(bodies, dist_thresh=60, iou_thresh=0.3)
    
    # 전원 확인
    power_active = any(p[1] < h * 0.45 for p in pins)
    if not power_active:
         for b in clean_bodies:
            if 'wire' in b['name'] and b['center'][1] < h * 0.45:
                power_active = True; break
    
    # 연결 확인
    if power_active:
        for comp in clean_bodies:
            cy = comp['center'][1]
            if cy < h*0.48 or cy > h*0.52: comp['is_on'] = True

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
    
    # 시각화 및 카운트
    summary_details = {}
    for comp in clean_bodies:
        is_on = comp['is_on']
        n = comp['normalized_name']
        
        # 와이어, 전원 등은 비교 대상에서 제외하거나 포함 여부 결정
        if 'wire' not in n and 'source' not in n:
             summary_details[n] = summary_details.get(n, 0) + 1

        if is_on:
            color = (0, 255, 0)
            status = "ON"
        else:
            color = (0, 0, 255)
            status = "OFF"
            off_count += 1
        
        x1, y1, x2, y2 = map(int, comp['box'])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.putText(img, status, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
    return img, {'off': off_count, 'total': len(clean_bodies), 'details': summary_details}

# ==========================================
# [5. 메인 UI (Streamlit)]
# ==========================================
st.title("🧠 BrainBoard V5: Strict Verification")
st.markdown("### 회로도와 실물의 [부품 개수]와 [연결 상태]를 모두 검증합니다.")

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

    if st.button("🚀 정밀 분석 실행"):
        with st.spinner("분석 중..."):
            res_ref_img, ref_data = analyze_schematic(ref_cv.copy(), model_sym)
            res_tgt_img, tgt_data = analyze_real(tgt_cv.copy(), model_real)

            st.divider()
            
            # [핵심] 비교 로직
            # 회로도에서 파악된 부품 수와 실물 부품 수를 비교
            mismatch_errors = []
            
            # 비교할 주요 부품 목록
            check_list = ['resistor', 'capacitor'] 
            
            for part in check_list:
                ref_cnt = ref_data['details'].get(part, 0)
                tgt_cnt = tgt_data['details'].get(part, 0)
                
                if ref_cnt != tgt_cnt:
                    mismatch_errors.append(f"⚠️ {part.upper()} 개수 불일치: 회로도({ref_cnt}) vs 실물({tgt_cnt})")
            
            # 결과 출력
            st.image(cv2.cvtColor(res_ref_img, cv2.COLOR_BGR2RGB), caption="회로도 분석", use_column_width=True)
            st.image(cv2.cvtColor(res_tgt_img, cv2.COLOR_BGR2RGB), caption=f"실물 분석 (OFF: {tgt_data['off']})", use_column_width=True)
            
            # 최종 판정
            if mismatch_errors:
                st.error("❌ 회로 구성이 일치하지 않습니다 (부품 누락/과잉)")
                for err in mismatch_errors:
                    st.write(err)
            elif tgt_data['off'] > 0:
                st.error(f"❌ 부품 연결 끊김 발견 ({tgt_data['off']}개 OFF)")
            else:
                st.success("✅ 완벽합니다! (부품 일치 & 전원 연결 성공)")
