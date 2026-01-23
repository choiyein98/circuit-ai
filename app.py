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

MODEL_REAL_PATH = 'best.pt'      
MODEL_SYM_PATH = 'symbol.pt'     

# 연결 감지 거리 (픽셀 단위) - 이 거리 안에 핀끼리 있으면 연결된 것으로 간주
CONNECTION_THRESHOLD = 90  

# ==========================================
# [Helper Functions]
# ==========================================
def calculate_iou(box1, box2):
    x1, y1, x2, y2 = max(box1[0], box2[0]), max(box1[1], box2[1]), min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

def solve_overlap(parts, distance_threshold=40):
    if not parts: return []
    parts.sort(key=lambda x: x.get('conf', 0), reverse=True)
    final_parts = []
    for current in parts:
        is_duplicate = False
        for kept in final_parts:
            iou = calculate_iou(current['box'], kept['box'])
            cx1, cy1 = current['center']
            cx2, cy2 = kept['center']
            dist = math.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)
            if iou > 0.3 or dist < distance_threshold:
                is_duplicate = True; break
        if not is_duplicate:
            final_parts.append(current)
    return final_parts

def get_center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

# ==========================================
# [분석 함수 1: 회로도 (Schematic)]
# ==========================================
def analyze_schematic(img, model):
    results = model.predict(source=img, save=False, conf=0.1, verbose=False)
    boxes = results[0].boxes
    
    raw_parts = []
    for box in boxes:
        cls_id = int(box.cls[0])
        name = model.names[cls_id].lower()
        conf = float(box.conf[0])
        coords = box.xyxy[0].tolist()
        center = get_center(coords)
        
        base_name = name.split('_')[0].split(' ')[0]
        if base_name in ['vdc', 'vsource', 'battery', 'voltage']: base_name = 'source'
        if base_name in ['cap', 'c', 'capacitor']: base_name = 'capacitor'
        if base_name in ['res', 'r', 'resistor']: base_name = 'resistor'
        
        raw_parts.append({'name': base_name, 'box': coords, 'center': center, 'conf': conf})

    clean_parts = solve_overlap(raw_parts, distance_threshold=30)
    
    # 회로도 시각화
    for part in clean_parts:
        name = part['name']
        x1, y1, x2, y2 = map(int, part['box'])
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img, name, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
    summary = {'total': len(clean_parts), 'details': {}}
    for part in clean_parts:
        n = part['name']
        summary['details'][n] = summary['details'].get(n, 0) + 1
        
    return img, summary

# ==========================================
# [분석 함수 2: 실물 (Real Board) - 다리 중심 시각화]
# ==========================================
def analyze_real(img, model):
    height, width, _ = img.shape
    
    # 1. 모델 예측
    results = model.predict(source=img, save=False, conf=0.15, verbose=False)
    boxes = results[0].boxes

    # 2. 객체 수집
    components = [] 
    legs = []       
    
    for box in boxes:
        cls_id = int(box.cls[0])
        name = model.names[cls_id].lower()
        coords = box.xyxy[0].tolist()
        center = get_center(coords)
        
        # 너무 작은 노이즈 제거
        if (coords[2]-coords[0]) * (coords[3]-coords[1]) < (width * height * 0.001): continue

        if any(x in name for x in ['pin', 'leg', 'lead', 'wire']):
            # 다리(Pin)는 좌표만 저장
            legs.append({'center': center, 'box': coords, 'type': 'pin'})
        elif 'breadboard' in name:
            continue
        else:
            # 몸통(Body) 저장
            components.append({
                'name': name, 'box': coords, 'center': center, 
                'my_legs': [], 'is_active': False
            })

    components = solve_overlap(components, distance_threshold=50)

    # 3. [가상 전원 레일] 표시
    top_rail_y = height * 0.20
    bottom_rail_y = height * 0.80
    
    cv2.rectangle(img, (0, 0), (width, int(top_rail_y)), (0, 255, 255), 1) 
    cv2.rectangle(img, (0, int(bottom_rail_y)), (width, height), (0, 255, 255), 1) 
    
    # 4. [다리 할당] 몸통과 가장 가까운 다리들을 찾아서 연결
    for comp in components:
        bw = comp['box'][2] - comp['box'][0]
        bh = comp['box'][3] - comp['box'][1]
        diag = math.sqrt(bw**2 + bh**2)
        search_radius = diag * 0.8  # 부품 크기 반경 내 검색

        for leg in legs:
            dist = math.sqrt((leg['center'][0]-comp['center'][0])**2 + (leg['center'][1]-comp['center'][1])**2)
            if dist < search_radius:
                comp['my_legs'].append(leg)
        
        # 다리가 인식 안 됐을 경우, 몸통 양 끝을 가상의 다리로 설정
        if len(comp['my_legs']) < 2:
            x1, y1, x2, y2 = comp['box']
            if bw > bh: # 가로형
                comp['my_legs'] = [{'center':(x1, (y1+y2)/2)}, {'center':(x2, (y1+y2)/2)}]
            else: # 세로형
                comp['my_legs'] = [{'center':((x1+x2)/2, y1)}, {'center':((x1+x2)/2, y2)}]

    # 5. [전류 흐름 시뮬레이션]
    # (A) 전원 소스 찾기 (레일에 닿은 다리)
    active_legs = [] 
    
    for comp in components:
        for leg in comp['my_legs']:
            ly = leg['center'][1]
            if ly < top_rail_y or ly > bottom_rail_y:
                comp['is_active'] = True
                active_legs.append(leg['center'])
    
    # (B) 전류 전파 (거리 기반)
    changed = True
    while changed:
        changed = False
        for comp in components:
            if comp['is_active']: 
                # 내가 켜졌으면 내 다리들도 전원 소스가 됨
                for leg in comp['my_legs']:
                    if leg['center'] not in active_legs:
                        active_legs.append(leg['center'])
                        changed = True
                continue
            
            # 내가 꺼져 있으면 주변에 활성 다리가 있는지 확인
            for my_leg in comp['my_legs']:
                for active_pt in active_legs:
                    dist = math.sqrt((my_leg['center'][0]-active_pt[0])**2 + (my_leg['center'][1]-active_pt[1])**2)
                    if dist < CONNECTION_THRESHOLD:
                        comp['is_active'] = True
                        changed = True
                        break 
                if comp['is_active']: break

    # 6. [시각화] 몸통은 얇게, 다리는 점으로!
    summary = {'total': 0, 'on': 0, 'off': 0, 'details': {}}
    
    for comp in components:
        name = comp['name']
        x1, y1, x2, y2 = map(int, comp['box'])
        center = comp['center']

        if comp['is_active']:
            color = (0, 255, 0) # 초록 (ON)
            status = "ON"
            summary['on'] += 1
        else:
            color = (0, 0, 255) # 빨강 (OFF)
            status = "OFF"
            summary['off'] += 1
            
        summary['total'] += 1
        
        # 1) 몸통 박스는 얇게 표시 (식별용)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
        
        # 2) [핵심] 다리 위치에 '점' 찍고 연결선 그리기
        for leg in comp['my_legs']:
            lx, ly = map(int, leg['center'])
            
            # 몸통 중심에서 다리까지 선 그리기
            cv2.line(img, (int(center[0]), int(center[1])), (lx, ly), color, 2)
            
            # 다리 끝부분에 원 그리기 (여기가 연결 포인트)
            cv2.circle(img, (lx, ly), 8, color, -1) 
            cv2.circle(img, (lx, ly), 8, (255, 255, 255), 2) # 흰 테두리

        # 3) 상태 텍스트
        cv2.putText(img, status, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        base_name = name.split('_')[0]
        summary['details'][base_name] = summary['details'].get(base_name, 0) + 1

    return img, summary

# ==========================================
# [Main UI Execution] - 여기가 있어야 화면이 나옵니다!
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

# 파일 업로더 레이아웃
col1, col2 = st.columns(2)
ref_file = col1.file_uploader("회로도(Schematic) 업로드", type=['jpg', 'png', 'jpeg'])
tgt_file = col2.file_uploader("실물(Real Board) 업로드", type=['jpg', 'png', 'jpeg'])

if ref_file and tgt_file:
    # 이미지 로드
    ref_image = Image.open(ref_file)
    tgt_image = Image.open(tgt_file)
    
    # OpenCV 포맷으로 변환 (RGB -> BGR)
    ref_cv = cv2.cvtColor(np.array(ref_image), cv2.COLOR_RGB2BGR)
    tgt_cv = cv2.cvtColor(np.array(tgt_image), cv2.COLOR_RGB2BGR)

    if st.button("🚀 회로 검증 시작 (Analyze)"):
        with st.spinner("AI가 회로를 분석 중입니다..."):
            # 분석 수행
            res_ref_img, ref_data = analyze_schematic(ref_cv.copy(), model_sym)
            res_tgt_img, tgt_data = analyze_real(tgt_cv.copy(), model_real)

            # 결과 리포트 표시
            st.divider()
            
            # 1. 개수 비교
            st.markdown("#### 1. 부품 개수 일치 여부")
            all_parts = set(ref_data['details'].keys()) | set(tgt_data['details'].keys())
            match_count = True
            for part in all_parts:
                c1 = ref_data['details'].get(part, 0)
                c2 = tgt_data['details'].get(part, 0)
                if c1 == c2:
                    st.write(f"- ✅ {part}: {c1}개 일치")
                else:
                    st.write(f"- ⚠️ {part}: 회로도 {c1}개 vs 실물 {c2}개")
                    match_count = False

            # 2. 연결 상태 비교
            st.markdown("#### 2. 전기적 연결 상태 (ON/OFF)")
            if tgt_data['off'] == 0:
                st.success(f"🎉 Perfect! 모든 부품({tgt_data['total']}개)이 정상 연결되었습니다.")
            else:
                st.error(f"❌ {tgt_data['off']}개의 부품이 연결되지 않았습니다. (빨간색 점 확인)")

            # 3. 이미지 출력 (BGR -> RGB 변환 필수)
            st.image(cv2.cvtColor(res_ref_img, cv2.COLOR_BGR2RGB), caption="PSpice 회로도 분석", use_column_width=True)
            st.image(cv2.cvtColor(res_tgt_img, cv2.COLOR_BGR2RGB), caption="실물 보드 분석 (점 = 다리 위치)", use_column_width=True)
