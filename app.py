import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import math
from PIL import Image

# ==========================================
# [1. 설정 및 라이브러리]
# ==========================================
st.set_page_config(page_title="BrainBoard V10 (Wire Fix)", layout="wide")

MODEL_REAL_PATH = 'best.pt'      # 실물 모델
MODEL_SYM_PATH = 'symbol.pt'     # 회로도 모델

# 연결 감지 범위 (몸통에서 다리가 뻗어나가는 범위)
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
            
            if curr_area > 0 and (inter_area / curr_area) > 0.7:
                is_dup = True; break

            # 3. 거리
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
# [3. 회로도 분석 (기존 성공 로직 유지)]
# ==========================================
def analyze_schematic(img, model):
    # 회로도는 0.10으로 낮게 잡아서 다 찾아냄
    res = model.predict(source=img, conf=0.10, verbose=False)
    
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
    
    clean = solve_overlap(raw, dist_thresh=0, iou_thresh=0.2)
    
    # [강제 보정] 가장 왼쪽 = 전원 (Source)
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
# [4. 실물 분석 (와이어 인식 강화)]
# ==========================================
def analyze_real(img, model):
    h, w, _ = img.shape
    
    # 1. 기본 스캔 (낮게 시작)
    res = model.predict(source=img, conf=0.10, verbose=False)
    
    bodies = []
    pins = [] 
    
    for b in res[0].boxes:
        name = model.names[int(b.cls[0])].lower()
        coords = b.xyxy[0].tolist()
        center = get_center(coords)
        conf = float(b.conf[0])
        
        # [핵심 수정 1] 부품별 인식 기준 (Dynamic Threshold)
        if 'cap' in name:
            min_conf = 0.45   # 커패시터: 엄격
        elif 'res' in name:
            min_conf = 0.35   # 저항: 적당히 엄격
        elif 'wire' in name:
            min_conf = 0.15   # [NEW] 와이어: 낮게 잡아서 인식률 높임
        else:
            min_conf = 0.25
            
        if conf < min_conf: continue

        # [핵심 수정 2] 와이어를 'bodies'에 포함시킴 (기존엔 pins로 뺐었음)
        # 이제 와이어도 박스가 쳐지고 ON/OFF 로직에 참여함
        if any(x in name for x in ['pin', 'leg', 'lead']) and 'wire' not in name:
            pins.append({'center': center, 'box': coords})
        elif 'breadboard' in name:
            continue
        else:
            # 저항, 커패시터, **와이어** 포함
            bodies.append({'name': name, 'box': coords, 'center': center, 'conf': conf, 'is_on': False})

    # 중복 제거
    clean_bodies = solve_overlap(bodies, dist_thresh=50, iou_thresh=0.3)
    
    # ----------------------------------------------------
    # [연결 로직]
    # ----------------------------------------------------
    
    # 1. 전원 공급원 찾기 (상단 45%)
    power_active = False
    
    # 핀이나 와이어가 상단에 있으면 전원 ON
    for b in clean_bodies:
        if 'wire' in b['name'] and b['center'][1] < h * 0.45:
            power_active = True; break
            
    if not power_active:
        for p in pins:
            if p['center'][1] < h * 0.45:
                power_active = True; break

    # 2. 연결 상태 전파
    if power_active:
        # (1) 직접 연결: 상단/하단 레일 영역
        for comp in clean_bodies:
            cy = comp['center'][1]
            if cy < h*0.48 or cy > h*0.52: 
                comp['is_on'] = True

        # (2) 간접 연결 (3회 반복 - 멀리 퍼지도록)
        for _ in range(3): 
            for comp in clean_bodies:
                if comp['is_on']: continue 
                
                cx, cy = comp['center']
                
                # A. 핀(다리)과 가까운가?
                for p in pins:
                    px, py = p['center']
                    # 핀이 상단/하단 전원부에 있거나
                    if py < h*0.48 or py > h*0.52:
                         dist = math.sqrt((cx - px)**2 + (cy - py)**2)
                         if dist < LEG_EXTENSION_RANGE:
                             comp['is_on'] = True; break

                if comp['is_on']: continue

                # B. 이미 켜진 다른 부품(와이어 포함)과 가까운가?
                for other in clean_bodies:
                    if not other['is_on']: continue
                    ocx, ocy = other['center']
                    dist = math.sqrt((cx - ocx)**2 + (cy - ocy)**2)
                    
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
        
        # 와이어는 개수 비교에서는 제외 (하지만 화면엔 표시됨)
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
st.title("🧠 BrainBoard V10 (Wire Fix)")
st.markdown("### 1. 부품 일치 여부 (와이어 인식 추가)")
st.markdown("### 2. 연결 상태 확인")

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
