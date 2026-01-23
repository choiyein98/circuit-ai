import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import math
from PIL import Image

# ==========================================
# [1. 설정 및 라이브러리]
# ==========================================
st.set_page_config(page_title="BrainBoard V11 (Tournament Logic)", layout="wide")

MODEL_REAL_PATH = 'best.pt'
MODEL_SYM_PATH = 'symbol.pt'

# 연결 감지 범위
LEG_EXTENSION_RANGE = 180        

# ==========================================
# [2. 유틸리티 함수: 강력한 토너먼트 로직]
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
    [토너먼트 로직]
    1. 점수(conf)가 높은 순서대로 줄을 세웁니다. (Winner 후보)
    2. 1등부터 차례대로 내려가면서, 자신과 겹치는 하위권 박스들을 모두 제거합니다.
    """
    if not parts: return []
    # 1. 신뢰도 기준 내림차순 정렬 (점수 높은게 0번 인덱스)
    parts.sort(key=lambda x: x.get('conf', 0), reverse=True)
    
    final = []
    for curr in parts:
        is_dup = False
        for k in final:
            # 2-1. 면적 겹침 (IoU) 체크
            iou = calculate_iou(curr['box'], k['box'])
            if iou > iou_thresh:
                is_dup = True; break
            
            # 2-2. 포함 관계 (큰 박스 안에 작은 박스) 체크
            x1 = max(curr['box'][0], k['box'][0])
            y1 = max(curr['box'][1], k['box'][1])
            x2 = min(curr['box'][2], k['box'][2])
            y2 = min(curr['box'][3], k['box'][3])
            inter_area = max(0, x2-x1) * max(0, y2-y1)
            curr_area = (curr['box'][2]-curr['box'][0]) * (curr['box'][3]-curr['box'][1])
            
            # 70% 이상 먹혔으면 제거
            if curr_area > 0 and (inter_area / curr_area) > 0.7:
                is_dup = True; break

            # 2-3. 거리 체크
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
# [3. 회로도 분석: 그물망 수법]
# ==========================================
def analyze_schematic(img, model):
    # [전략] 인식률 0.01 (1%) -> 일단 눈에 보이는 건 다 잡습니다.
    # 그 후 토너먼트 로직으로 정리합니다. 이렇게 해야 놓치는 부품이 없습니다.
    res = model.predict(source=img, conf=0.01, verbose=False)
    
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
    
    # [토너먼트] 겹침 허용치 0.1 (조금만 겹쳐도 점수 높은 놈이 이김)
    clean = solve_overlap(raw, dist_thresh=0, iou_thresh=0.1)
    
    # [강제 보정] 왼쪽 = Source
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
# [4. 실물 분석: 엄격한 관리자 모드]
# ==========================================
def analyze_real(img, model):
    h, w, _ = img.shape
    
    # 1. 기본 스캔 (0.1로 시작하되 내부에서 엄격하게 컷)
    res = model.predict(source=img, conf=0.10, verbose=False)
    
    bodies = []
    pins = [] 
    
    for b in res[0].boxes:
        name = model.names[int(b.cls[0])].lower()
        coords = b.xyxy[0].tolist()
        center = get_center(coords)
        conf = float(b.conf[0])
        
        # [핵심] 부품별 커트라인 (Threshold)
        # 저항이 자꾸 3개로 잡히는 문제 해결 -> 0.40으로 대폭 상향
        if 'cap' in name: min_conf = 0.45      # 커패시터: 매우 엄격
        elif 'res' in name: min_conf = 0.40    # 저항: 엄격 (그림자/와이어 방지)
        elif 'wire' in name: min_conf = 0.15   # 와이어: 관대함
        else: min_conf = 0.25
            
        if conf < min_conf: continue

        # 와이어도 Body로 취급 (연결/시각화용)
        if any(x in name for x in ['pin', 'leg', 'lead']) and 'wire' not in name:
            pins.append({'center': center, 'box': coords})
        elif 'breadboard' in name:
            continue
        else:
            bodies.append({'name': name, 'box': coords, 'center': center, 'conf': conf, 'is_on': False})

    # 중복 제거
    clean_bodies = solve_overlap(bodies, dist_thresh=50, iou_thresh=0.3)
    
    # ----------------------------------------------------
    # [연결 로직]
    # ----------------------------------------------------
    power_active = False
    for b in clean_bodies:
        if 'wire' in b['name'] and b['center'][1] < h * 0.45:
            power_active = True; break
    if not power_active:
        for p in pins:
            if p['center'][1] < h * 0.45:
                power_active = True; break

    if power_active:
        # (1) 직접 연결
        for comp in clean_bodies:
            cy = comp['center'][1]
            if cy < h*0.48 or cy > h*0.52: 
                comp['is_on'] = True

        # (2) 간접 연결 (3회 전파)
        for _ in range(3): 
            for comp in clean_bodies:
                if comp['is_on']: continue 
                cx, cy = comp['center']
                
                # A. 핀(다리)
                for p in pins:
                    px, py = p['center']
                    if py < h*0.48 or py > h*0.52:
                         dist = math.sqrt((cx - px)**2 + (cy - py)**2)
                         if dist < LEG_EXTENSION_RANGE:
                             comp['is_on'] = True; break

                if comp['is_on']: continue

                # B. 다른 부품
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
        
        # 카운팅용 이름 (와이어 제외)
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
st.title("🧠 BrainBoard V11 (Tournament Logic)")
st.markdown("### 1. 부품 일치 여부 (토너먼트 선별)")
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
            
            # [핵심] 불일치 검사 및 메시지 생성
            mismatch_errors = []
            target_parts = ['resistor', 'capacitor', 'inductor'] # 검사 대상 추가
            
            for part in target_parts:
                ref_cnt = ref_data['details'].get(part, 0)
                tgt_cnt = tgt_data['details'].get(part, 0)
                
                # 하나라도 다르면 에러 메시지 추가
                if ref_cnt != tgt_cnt:
                    mismatch_errors.append(f"⚠️ {part.upper()} 불일치: 회로도 {ref_cnt}개 vs 실물 {tgt_cnt}개")
            
            # 이미지 출력
            st.image(cv2.cvtColor(res_ref_img, cv2.COLOR_BGR2RGB), caption="회로도 분석 (1% 탐지 + 토너먼트)", use_column_width=True)
            st.image(cv2.cvtColor(res_tgt_img, cv2.COLOR_BGR2RGB), caption=f"실물 분석 (OFF: {tgt_data['off']})", use_column_width=True)
            
            if mismatch_errors:
                st.error("❌ 회로 구성이 다릅니다 (부품 개수 불일치)")
                for err in mismatch_errors:
                    st.write(err) # 여기서 커패시터 불일치도 뜹니다.
            elif tgt_data['off'] > 0:
                st.error(f"❌ 부품 연결이 끊어졌습니다 ({tgt_data['off']}개 OFF)")
            else:
                st.success("✅ 완벽합니다! (부품 일치 & 전원 연결 성공)")
