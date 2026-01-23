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

MODEL_REAL_PATH = 'best.pt'      # 실제 보드용 모델 경로
MODEL_SYM_PATH = 'symbol.pt'     # 회로도용 모델 경로 (symbol.pt 사용)
PROXIMITY_THRESHOLD = 50         # 같은 열(Column)로 판단할 거리 기준 (픽셀)

# ==========================================
# [Helper Functions]
# ==========================================
def calculate_iou(box1, box2):
    """두 박스의 겹치는 비율(IoU) 계산"""
    x1, y1, x2, y2 = max(box1[0], box2[0]), max(box1[1], box2[1]), min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

def solve_overlap(parts, distance_threshold=40):
    """중복 감지된 객체 필터링"""
    if not parts: return []
    # conf(신뢰도)가 높은 순서대로 정렬
    parts.sort(key=lambda x: x.get('conf', 0), reverse=True)
    
    final_parts = []
    for current in parts:
        is_duplicate = False
        for kept in final_parts:
            # 중심점 거리가 너무 가깝거나, IoU가 높으면 중복으로 간주
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

def check_vertical_alignment(pt1, pt2, tolerance=30):
    """두 점이 브레드보드상 같은 세로줄(Column)에 있는지 확인 (x좌표 비교)"""
    return abs(pt1[0] - pt2[0]) < tolerance

# ==========================================
# [분석 함수 1: 회로도 (Schematic) - 인식률 개선]
# ==========================================
def analyze_schematic(img, model):
    # conf를 0.1로 낮춰서 잘 못 잡던 부품도 잡도록 설정
    results = model.predict(source=img, save=False, conf=0.1, verbose=False)
    boxes = results[0].boxes
    
    raw_parts = []
    for box in boxes:
        cls_id = int(box.cls[0])
        name = model.names[cls_id].lower()
        conf = float(box.conf[0])
        coords = box.xyxy[0].tolist()
        center = get_center(coords)
        
        # 이름 정규화 (모델 클래스 이름 차이 보정)
        base_name = name.split('_')[0].split(' ')[0]
        if base_name in ['vdc', 'vsource', 'battery', 'voltage']: base_name = 'source'
        if base_name in ['cap', 'c', 'capacitor']: base_name = 'capacitor'
        if base_name in ['res', 'r', 'resistor']: base_name = 'resistor'
        
        raw_parts.append({'name': base_name, 'box': coords, 'center': center, 'conf': conf})

    # 중복 제거 수행
    clean_parts = solve_overlap(raw_parts, distance_threshold=30)

    # 가장 왼쪽 부품을 Source로 가정 (회로도 관례)
    if clean_parts:
        leftmost_part = min(clean_parts, key=lambda p: p['center'][0])
        if leftmost_part['name'] != 'source' and 'res' not in leftmost_part['name']: 
             # 저항이 아닐때만 강제 변환 (오인식 방지)
             pass 

    summary = {'total': 0, 'details': {}}
    for part in clean_parts:
        name = part['name']
        x1, y1, x2, y2 = map(int, part['box'])
        
        # 회로도에는 파란색으로 표시
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img, name, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        summary['total'] += 1
        summary['details'][name] = summary['details'].get(name, 0) + 1
        
    return img, summary

# ==========================================
# [분석 함수 2: 실물 (Real Board) - 전기적 연결 로직 적용]
# ==========================================
def analyze_real(img, model):
    height, width, _ = img.shape
    
    # 1. 모델 예측
    results = model.predict(source=img, save=False, conf=0.15, verbose=False)
    boxes = results[0].boxes

    # 2. 부품 분류 (Body와 Leg 분리)
    components = [] # 몸체
    legs = []       # 다리/핀
    
    for box in boxes:
        cls_id = int(box.cls[0])
        name = model.names[cls_id].lower()
        conf = float(box.conf[0])
        coords = box.xyxy[0].tolist()
        center = get_center(coords)
        
        # 너무 작은 노이즈 제거
        w_box, h_box = coords[2]-coords[0], coords[3]-coords[1]
        if w_box * h_box < (width * height * 0.001): continue

        if any(x in name for x in ['pin', 'leg', 'lead', 'wire']):
            # 와이어도 다리(연결점)의 일종으로 취급하여 좌표 수집
            legs.append({'center': center, 'box': coords})
        elif 'breadboard' in name:
            continue
        else:
            components.append({
                'name': name, 'box': coords, 'center': center, 
                'conf': conf, 'connected_nodes': set(), 'is_active': False
            })

    # 중복 부품 제거
    components = solve_overlap(components, distance_threshold=50)

    # 3. [가상 전원 레일 설정] (사용자 요청: 위/아래 가상 박스)
    # 이미지의 상단 15%는 VCC(전원), 하단 15%는 GND(접지) 또는 VCC 영역으로 가정
    top_rail_y = height * 0.15
    bottom_rail_y = height * 0.85
    
    # 가상 레일 시각화 (노란색 점선 박스)
    cv2.rectangle(img, (0, 0), (width, int(top_rail_y)), (0, 255, 255), 2) # Top Rail
    cv2.rectangle(img, (0, int(bottom_rail_y)), (width, height), (0, 255, 255), 2) # Bottom Rail
    cv2.putText(img, "Virtual Power Rail (VCC)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # 4. 부품별 다리(Leg) 할당 및 노드(Node) 연결
    # 다리가 감지되지 않은 부품은 부품 박스의 양 끝단을 다리 위치로 추정
    active_nodes = set(["VCC_RAIL"]) # 전기가 흐르는 노드 집합
    
    # 회로 구성을 위한 그래프
    # Node 정의: "Col_X" (세로줄), "VCC_RAIL", "GND_RAIL"
    
    for comp in components:
        comp_legs = []
        
        # A. 감지된 다리 중 부품 근처에 있는 것 찾기
        for leg in legs:
            lx, ly = leg['center']
            cx, cy = comp['center']
            # 부품 중심과 다리 사이의 거리
            dist = math.sqrt((lx-cx)**2 + (ly-cy)**2)
            # 부품 크기의 절반 정도 거리 내에 있으면 내 다리로 인정
            box_diag = math.sqrt((comp['box'][2]-comp['box'][0])**2 + (comp['box'][3]-comp['box'][1])**2)
            if dist < box_diag * 0.8:
                comp_legs.append(leg['center'])
        
        # B. 다리가 충분히 감지되지 않았다면 박스 좌우/상하 끝을 다리로 추정
        if len(comp_legs) < 2:
            x1, y1, x2, y2 = comp['box']
            if (x2-x1) > (y2-y1): # 가로로 긴 부품
                comp_legs = [(x1, (y1+y2)/2), (x2, (y1+y2)/2)]
            else: # 세로로 긴 부품
                comp_legs = [((x1+x2)/2, y1), ((x1+x2)/2, y2)]
        
        # C. 각 다리가 어느 노드(Node)에 꽂혔는지 판별
        for lx, ly in comp_legs:
            node_id = None
            
            # 1) 가상 전원 레일에 있는지 확인
            if ly < top_rail_y or ly > bottom_rail_y:
                node_id = "VCC_RAIL" # 편의상 위아래 모두 전원 공급처로 가정
            else:
                # 2) 브레드보드 내부 영역: X좌표(세로줄)를 기준으로 노드 ID 생성
                # 50픽셀 단위로 세로줄을 구분한다고 가정 (이미지 해상도에 따라 조정 가능)
                col_index = int(lx / PROXIMITY_THRESHOLD) 
                node_id = f"Col_{col_index}"
            
            comp['connected_nodes'].add(node_id)

    # 5. [전류 흐름 시뮬레이션] (BFS/Propagation)
    # VCC_RAIL에 연결된 부품부터 시작해서 전기를 퍼뜨림
    
    # 1단계: 전원에 직접 연결된 부품 활성화
    changed = True
    while changed:
        changed = False
        for comp in components:
            if comp['is_active']: continue
            
            # 내 다리 중 하나라도 활성 노드(전기가 흐르는 곳)에 연결되어 있으면 나도 켜짐
            if not comp['connected_nodes'].isdisjoint(active_nodes):
                comp['is_active'] = True
                # 내가 켜졌으면, 내가 연결된 다른 노드들도 전기가 흐르게 됨
                new_nodes = comp['connected_nodes'] - active_nodes
                if new_nodes:
                    active_nodes.update(new_nodes)
                    changed = True

    # 6. 결과 시각화
    summary = {'total': 0, 'on': 0, 'off': 0, 'details': {}}
    
    for comp in components:
        name = comp['name']
        x1, y1, x2, y2 = map(int, comp['box'])
        
        if comp['is_active']:
            color = (0, 255, 0) # 초록색 (ON)
            status = "ON"
            summary['on'] += 1
        else:
            color = (0, 0, 255) # 빨간색 (OFF)
            status = "OFF"
            summary['off'] += 1
            
        summary['total'] += 1
        
        # 박스 및 텍스트
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.putText(img, status, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # 부품 카운트
        base_name = name.split('_')[0]
        summary['details'][base_name] = summary['details'].get(base_name, 0) + 1

    return img, summary

# ==========================================
# [WEB APP UI] Streamlit Main Code
# ==========================================
st.title("🧠 BrainBoard V44: 전기적 연결 검증기")
st.markdown("### 가상 전원 레일 및 세로줄 연결 로직 적용")

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

    if st.button("🚀 회로 검증 시작"):
        with st.spinner("회로도와 브레드보드를 분석 중입니다..."):
            # 분석 실행
            res_ref_img, ref_data = analyze_schematic(ref_cv.copy(), model_sym)
            res_tgt_img, tgt_data = analyze_real(tgt_cv.copy(), model_real)

            # 결과 리포트
            st.divider()
            
            # 개수 비교
            st.markdown("#### 1. 부품 개수 확인")
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

            # 연결 상태 비교
            st.markdown("#### 2. 전기적 연결 상태 (ON/OFF)")
            if tgt_data['off'] == 0:
                st.success(f"🎉 모든 부품({tgt_data['total']}개)에 전원이 공급되고 있습니다! (All ON)")
            else:
                st.error(f"❌ {tgt_data['off']}개의 부품이 연결되지 않았습니다. (OFF)")
                st.info("💡 팁: 빨간색(OFF) 부품은 전원 레일과 끊어져 있거나, 같은 세로줄에 연결되지 않은 상태입니다.")

            # 이미지 출력
            st.image(cv2.cvtColor(res_ref_img, cv2.COLOR_BGR2RGB), caption="PSpice 회로도 분석 (인식률 개선)", use_column_width=True)
            st.image(cv2.cvtColor(res_tgt_img, cv2.COLOR_BGR2RGB), caption="실물 보드 분석 (가상 전원 레일 + 전기적 흐름)", use_column_width=True)
