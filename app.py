import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import math
from PIL import Image

# ==========================================
# [1. 설정 및 라이브러리]
# ==========================================
st.set_page_config(page_title="BrainBoard V9 (Body Count / Leg Connect)", layout="wide")

MODEL_REAL_PATH = 'best.pt'      # 실물 모델
MODEL_SYM_PATH = 'symbol.pt'     # 회로도 모델

# [핵심 설정]
REAL_CONF_THRESH = 0.35          # 실물: 몸통을 확실히 잡기 위해 높임 (오인식 방지)
SCHEMATIC_CONF_THRESH = 0.10     # 회로도: 일단 다 잡기 위해 낮춤

# 연결 감지 범위 (몸통에서 다리가 뻗어나가는 범위라고 가정)
# 이 값을 늘리면 부품이 멀리 있어도 연결된 것으로 간주합니다.
LEG_EXTENSION_RANGE = 180        

# ==========================================
# [2. 유틸리티 함수]
# ==========================================
def calculate_iou(box1, box2):
    x1, y1, x2, y2 = max(box1[0], box2[0]), max(box1[1], box2[1]), min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

def solve_overlap(parts, dist_thresh=0, iou_thresh=0.4):
    """
    강력한 중복 제거 (토너먼트 방식)
    """
    if not parts: return []
    # 신뢰도 높은 순으로 정렬 (확률 높은게 짱)
    parts.sort(key=lambda x: x.get('conf', 0), reverse=True)
    
    final = []
    for curr in parts:
        is_dup = False
        for k in final:
            # 1. 면적 겹침 (IoU)
            iou = calculate_iou(curr['box'], k['box'])
            if iou > iou_thresh:
                is_dup = True; break
            
            # 2. 포함 관계 (큰 박스 안에 작은 박스)
            x1 = max(curr['box'][0], k['box'][0])
            y1 = max(curr['box'][1], k['box'][1])
            x2 = min(curr['box'][2], k['box'][2])
            y2 = min(curr['box'][3], k['box'][3])
            inter_area = max(0, x2-x1) * max(0, y2-y1)
            curr_area = (curr['box'][2]-curr['box'][0]) * (curr['box'][3]-curr['box'][1])
            
            # 70% 이상 포함되면 중복 처리
            if curr_area > 0 and (inter_area / curr_area) > 0.7:
                is_dup = True; break

            # 3. 거리 (너무 가까우면 같은 부품으로 간주)
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
# [3. 회로도 분석 (기준 완화 + 강력 분류)]
# ==========================================
def analyze_schematic(img, model):
    # 1. 0.10으로 낮춰서 일단 다 찾습니다. (놓치는 것 방지)
    res = model.predict(source=img, conf=SCHEMATIC_CONF_THRESH, verbose=False)
    
    raw = []
    for b in res[0].boxes:
        cls_id = int(b.cls[0])
        raw_name = model.names[cls_id].lower()
        conf = float(b.conf[0])
        
        raw.append({
            'name': raw_name, 
            'box': b.xyxy[0].tolist(), 
            'center': get_center(b.xyxy[0].tolist()),
            'conf': conf
        })
    
    # 2. 중복 제거 (겹치면 점수 높은 놈만 남김)
    clean = solve_overlap(raw, dist_thresh=0, iou_thresh=0.2)
    
    # 3. [강제 보정] 가장 왼쪽 = 전원 (Source)
    leftmost_idx = -1
    min_x = float('inf')
    if clean:
        for i, p in enumerate(clean):
            if p['center'][0] < min_x:
                min_x = p['center'][0]
                leftmost_idx = i

    summary_details = {}
    
    for i, p in enumerate(clean):
        raw_name = p['name']
        name = raw_name 
        
        # 이름 단순화
        if 'cap' in raw_name: name = 'capacitor'
        elif 'res' in raw_name: name = 'resistor'
        elif 'ind' in raw_name: name = 'inductor'
        elif 'dio' in raw_name: name = 'diode'
        elif any(x in raw_name for x in ['volt', 'batt', 'source']): name = 'source'

        # 가장 왼쪽은 무조건 Source
        if i == leftmost_idx:
            name = 'source'
        
        # 시각화
        x1, y1, x2, y2 = map(int, p['box'])
        box_color = (255, 0, 0) if name == 'source' else (0, 0, 255)
        
        cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(img, name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
        
        summary_details[name] = summary_details.get(name, 0) + 1
        
    return img, {'total': len(clean), 'details': summary_details}

# ==========================================
# [4. 실물 분석 (몸통 카운팅 + 다리 연결 확인)]
# ==========================================
def analyze_real(img, model):
    h, w, _ = img.shape
    
    # 1. 몸통 인식을 위해 기준을 0.35로 높임 (잡동사니 제거)
    res = model.predict(source=img, conf=REAL_CONF_THRESH, verbose=False)
    
    bodies = []
    pins = [] # 핀/와이어 (연결 매개체)
    
    for b in res[0].boxes:
        name = model.names[int(b.cls[0])].lower()
        coords = b.xyxy[0].tolist()
        center = get_center(coords)
        conf = float(b.conf[0])
        
        # 핀/와이어는 연결 확인용으로 따로 뺌 (개수엔 포함 안 함)
        if any(x in name for x in ['pin', 'leg', 'lead', 'wire']):
            pins.append({'center': center, 'box': coords})
        elif 'breadboard' in name:
            continue
        else:
            # 저항, 커패시터 등 '몸통'
            bodies.append({'name': name, 'box': coords, 'center': center, 'conf': conf, 'is_on': False})

    # 중복 제거 (확실한 몸통만 남김)
    clean_bodies = solve_overlap(bodies, dist_thresh=50, iou_thresh=0.3)
    
    # ----------------------------------------------------
    # [연결 로직 수정] 몸통 중심이 아닌 '영역'으로 판단
    # ----------------------------------------------------
    
    # 1. 전원 공급원(핀/와이어) 찾기 (상단 45%)
    power_sources = []
    for p in pins:
        if p['center'][1] < h * 0.45:
            power_sources.append(p)
    
    # 전원이 하나라도 있으면 활성화 시작
    power_active = len(power_sources) > 0
    if not power_active:
         # 핀이 없으면 상단에 있는 와이어형 부품이라도 찾음
         for b in clean_bodies:
            if 'wire' in b['name'] and b['center'][1] < h * 0.45:
                power_active = True
                power_sources.append(b) # 얘도 전원 소스 취급
                break

    # 2. 연결 상태 전파 (몸통 + 다리길이 고려)
    if power_active:
        # (1) 직접 연결: 상단/하단 레일에 몸통이 걸쳐있는 경우
        for comp in clean_bodies:
            cy = comp['center'][1]
            # 상단(0.48 이하) 또는 하단(0.52 이상) 레일 영역
            if cy < h*0.48 or cy > h*0.52: 
                comp['is_on'] = True

        # (2) 간접 연결: 전원 소스나 이미 켜진 부품 근처에 있는 경우
        # 반복 횟수를 늘려(3회) 멀리 있는 부품까지 전기가 흐르게 함
        for _ in range(3): 
            for comp in clean_bodies:
                if comp['is_on']: continue 
                
                cx, cy = comp['center']
                
                # A. 전원 핀/와이어와 가까운가? (다리 길이 고려하여 거리 기준 LEG_EXTENSION_RANGE 사용)
                for src in power_sources:
                    src_x, src_y = src['center']
                    dist = math.sqrt((cx - src_x)**2 + (cy - src_y)**2)
                    if dist < LEG_EXTENSION_RANGE:
                        comp['is_on'] = True; break
                
                if comp['is_on']: continue

                # B. 이미 켜진 다른 부품과 가까운가?
                for other in clean_bodies:
                    if not other['is_on']: continue
                    ocx, ocy = other['center']
                    dist = math.sqrt((cx - ocx)**2 + (cy - ocy)**2)
                    
                    # 두 부품 간의 거리가 (다리길이 * 1.5) 이내면 연결된 것으로 간주
                    if dist < LEG_EXTENSION_RANGE * 1.5:
                        comp['is_on'] = True; break

    off_count = 0
    real_details = {} 
    
    for comp in clean_bodies:
        is_on = comp['is_on']
        raw_name = comp['name']
        
        # 카운팅용 이름 정규화
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
        # 박스 위에 ON/OFF 표시
        cv2.putText(img, status, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
    return img, {'off': off_count, 'total': len(clean_bodies), 'details': real_details}

# ==========================================
# [5. 메인 UI]
# ==========================================
st.title("🧠 BrainBoard V9 (Body Count / Leg Connect)")
st.markdown("### 1. 부품 일치 여부 (몸통 인식)")
st.markdown("### 2. 연결 상태 (다리 범위 포함)")

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
        with st.spinner("AI 분석 중..."):
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
            
            if mismatch_errors:
                st.error("❌ 회로 구성이 다릅니다 (부품 개수 불일치)")
                for err in mismatch_errors:
                    st.write(err)
            elif tgt_data['off'] > 0:
                st.error(f"❌ 부품 연결이 끊어졌습니다 ({tgt_data['off']}개 OFF)")
            else:
                st.success("✅ 완벽합니다! (부품 일치 & 전원 연결 성공)")
