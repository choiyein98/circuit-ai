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
# [2. 유틸리티 함수 (중복 제거)]
# ==========================================
def calculate_iou(box1, box2):
    x1, y1, x2, y2 = max(box1[0], box2[0]), max(box1[1], box2[1]), min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

def solve_overlap(parts, dist_thresh=0, iou_thresh=0.5):
    if not parts: return []
    if 'conf' in parts[0]:
        parts.sort(key=lambda x: x.get('conf', 0), reverse=True)
    
    final = []
    for curr in parts:
        is_dup = False
        for k in final:
            # 1. IoU(면적 겹침) 체크
            iou = calculate_iou(curr['box'], k['box'])
            if iou > iou_thresh:
                is_dup = True; break
            
            # 2. 거리 체크 (옵션)
            if dist_thresh > 0:
                dist = math.sqrt((curr['center'][0]-k['center'][0])**2 + (curr['center'][1]-k['center'][1])**2)
                if dist < dist_thresh:
                    is_dup = True; break
                    
            # 3. 포함 관계 체크 (박스 안에 박스 제거)
            # 큰 박스 안에 작은 박스가 80% 이상 들어가면 중복 처리
            x1 = max(curr['box'][0], k['box'][0])
            y1 = max(curr['box'][1], k['box'][1])
            x2 = min(curr['box'][2], k['box'][2])
            y2 = min(curr['box'][3], k['box'][3])
            inter_area = max(0, x2-x1) * max(0, y2-y1)
            curr_area = (curr['box'][2]-curr['box'][0]) * (curr['box'][3]-curr['box'][1])
            if curr_area > 0 and (inter_area / curr_area) > 0.8:
                is_dup = True; break

        if not is_dup:
            final.append(curr)
    return final

def get_center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

# ==========================================
# [3. 회로도 분석 (커패시터 전용 로직 적용)]
# ==========================================
def analyze_schematic(img, model):
    # 일단 기준을 최대한 낮춰서(0.1) 다 가져옵니다.
    res = model.predict(source=img, conf=0.1, verbose=False)
    
    raw = []
    for b in res[0].boxes:
        cls_id = int(b.cls[0])
        raw_name = model.names[cls_id].lower()
        conf = float(b.conf[0])
        
        # [핵심 수정] 부품별로 통과 기준을 다르게 적용
        if 'cap' in raw_name:
            # 커패시터는 잘 안잡히므로 0.1(10%)만 넘어도 통과!
            min_conf = 0.10
        else:
            # 저항, 소스 등은 엉뚱한거 잡지 않게 0.30(30%) 이상이어야 통과
            min_conf = 0.30
            
        if conf < min_conf:
            continue # 기준 미달 탈락

        raw.append({
            'name': raw_name, 
            'box': b.xyxy[0].tolist(), 
            'center': get_center(b.xyxy[0].tolist()),
            'conf': conf
        })
    
    # 중복 제거 (거리체크 X, 겹침 10% 이상이면 제거)
    clean = solve_overlap(raw, dist_thresh=0, iou_thresh=0.1)
    
    summary_details = {}
    
    for p in clean:
        raw_name = p['name']
        name = raw_name
        
        # 이름 정규화
        if p['center'][0] < img.shape[1] * 0.25: name = 'source'
        elif 'cap' in raw_name: name = 'capacitor'
        elif 'res' in raw_name: name = 'resistor'
        elif 'ind' in raw_name: name = 'inductor'
        elif 'dio' in raw_name: name = 'diode'
        
        # 그리기
        x1, y1, x2, y2 = map(int, p['box'])
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img, name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        summary_details[name] = summary_details.get(name, 0) + 1
        
    return img, {'total': len(clean), 'details': summary_details}

# ==========================================
# [4. 실물 분석 (완벽함 - 수정 없음)]
# ==========================================
def analyze_real(img, model):
    h, w, _ = img.shape
    # 실물은 노이즈가 많으므로 0.25 유지
    res = model.predict(source=img, conf=0.25, verbose=False)
    
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

    clean_bodies = solve_overlap(bodies, dist_thresh=60, iou_thresh=0.3)
    
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

        # 전파 (2회)
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
    real_details = {} 
    
    for comp in clean_bodies:
        is_on = comp['is_on']
        raw_name = comp['name']
        
        norm_name = raw_name
        if 'res' in raw_name: norm_name = 'resistor'
        elif 'cap' in raw_name: norm_name = 'capacitor'
        
        if 'wire' not in raw_name:
            real_details[norm_name] = real_details.get(norm_name, 0) + 1

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
        
    return img, {'off': off_count, 'total': len(clean_bodies), 'details': real_details}

# ==========================================
# [5. 메인 UI]
# ==========================================
st.title("🧠 BrainBoard V5: Final Verification")
st.markdown("### 1. 부품 일치 여부 확인")
st.markdown("### 2. 전원 연결 상태(ON/OFF) 확인")

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
            
            # 불일치 검사
            mismatch_errors = []
            target_parts = ['resistor', 'capacitor']
            
            for part in target_parts:
                ref_cnt = ref_data['details'].get(part, 0)
                tgt_cnt = tgt_data['details'].get(part, 0)
                
                if ref_cnt != tgt_cnt:
                    mismatch_errors.append(f"⚠️ {part.upper()} 불일치: 회로도 {ref_cnt}개 vs 실물 {tgt_cnt}개")
            
            # 이미지 출력
            st.image(cv2.cvtColor(res_ref_img, cv2.COLOR_BGR2RGB), caption="회로도 분석", use_column_width=True)
            st.image(cv2.cvtColor(res_tgt_img, cv2.COLOR_BGR2RGB), caption=f"실물 분석 (OFF: {tgt_data['off']})", use_column_width=True)
            
            # 최종 결과
            if mismatch_errors:
                st.error("❌ 회로 구성이 다릅니다 (부품 개수 불일치)")
                for err in mismatch_errors:
                    st.write(err)
            elif tgt_data['off'] > 0:
                st.error(f"❌ 부품 연결이 끊어졌습니다 ({tgt_data['off']}개 OFF)")
            else:
                st.success("✅ 완벽합니다! (부품 일치 & 전원 연결 성공)")
