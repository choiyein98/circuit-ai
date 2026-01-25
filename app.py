import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import math
from PIL import Image
import itertools

# ==========================================
# [설정] BrainBoard V54: Engineering Logic
# ==========================================
st.set_page_config(page_title="BrainBoard V54: Engineering", layout="wide")

# [모델 경로] 사용자가 제공한 파일명 유지
REAL_MODEL_PATHS = ['best.pt', 'best(2).pt', 'best(3).pt']
MODEL_SYM_PATH = 'symbol.pt'

# [엔지니어링 상수]
CONNECTION_THRESHOLD = 50   # 픽셀 단위: 이 거리 안이면 같은 노드(Node)로 간주
SHORT_CIRCUIT_IOU = 0.8     # 겹침 허용치

# ==========================================
# [Helper Class] 회로 검증용 노드 관리자
# ==========================================
class CircuitGraph:
    def __init__(self):
        self.nodes = [] # List of sets, each set contains point IDs or coords
        self.components = [] # List of {'name':, 'terminals': [(x,y), (x,y)], 'node_ids': [id1, id2]}

    def find_node(self, point):
        """특정 좌표가 속한 노드 ID를 반환 (없으면 생성)"""
        for i, node_group in enumerate(self.nodes):
            for existing_point in node_group:
                dist = math.sqrt((point[0]-existing_point[0])**2 + (point[1]-existing_point[1])**2)
                if dist < CONNECTION_THRESHOLD:
                    node_group.append(point)
                    return i
        
        # 새로운 노드 생성
        self.nodes.append([point])
        return len(self.nodes) - 1

    def add_component(self, name, box):
        """부품의 Bounding Box를 기반으로 양 끝단(Terminal)을 추정하여 노드에 등록"""
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        center = ((x1+x2)/2, (y1+y2)/2)
        
        # [Terminal Estimation] 가로/세로 비율에 따라 단자 위치 추정
        if w > h * 1.2: # 가로로 긴 부품 (Horizontal)
            t1 = (x1 + w*0.1, center[1]) # 왼쪽 끝
            t2 = (x2 - w*0.1, center[1]) # 오른쪽 끝
        elif h > w * 1.2: # 세로로 긴 부품 (Vertical)
            t1 = (center[0], y1 + h*0.1) # 위쪽 끝
            t2 = (center[0], y2 - h*0.1) # 아래쪽 끝
        else: # 정사각형에 가까움 -> 대각선 혹은 중심 근처 양옆 (Default)
            t1 = (x1 + w*0.2, y1 + h*0.2)
            t2 = (x2 - w*0.2, y2 - h*0.2)
            
        node_id1 = self.find_node(t1)
        node_id2 = self.find_node(t2)
        
        comp_info = {
            'name': name,
            'box': box,
            'terminals': [t1, t2],
            'node_ids': [node_id1, node_id2],
            'status': 'OK'
        }
        
        # [Rule 1] 단락(Short) 검사: 양 끝단이 같은 노드임
        if node_id1 == node_id2:
            comp_info['status'] = 'SHORT'
            
        self.components.append(comp_info)
        return comp_info

    def analyze_connectivity(self, power_rail_nodes):
        """전원부와 연결성 확인 (VCC/GND 연결 여부)"""
        # 간단한 그래프 탐색 대신, 현재 노드가 '다른 부품'과 연결되어 있는지 확인 (Open Check)
        # Power rail logic: Power Rail 영역에 있는 노드 ID를 식별
        
        node_connection_count = [0] * len(self.nodes)
        
        # 각 노드에 연결된 핀(Terminal) 개수 세기
        for comp in self.components:
            for nid in comp['node_ids']:
                node_connection_count[nid] += 1
                
        for comp in self.components:
            if comp['status'] == 'SHORT': continue
            
            n1, n2 = comp['node_ids']
            
            # [Rule 2] 단선(Open) 검사: 노드에 연결된 핀이 나 혼자뿐임
            if node_connection_count[n1] < 2 and n1 not in power_rail_nodes:
                comp['status'] = 'OPEN'
            elif node_connection_count[n2] < 2 and n2 not in power_rail_nodes:
                comp['status'] = 'OPEN'
            else:
                comp['status'] = 'CONNECTED'

# ==========================================
# [Helper Functions] 기본 기하학 함수
# ==========================================
def calculate_iou(box1, box2):
    x1, y1, x2, y2 = max(box1[0], box2[0]), max(box1[1], box2[1]), min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

def get_center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

# ==========================================
# [중복 제거] 기존 로직 유지
# ==========================================
def solve_overlap_schematic_v48(parts, distance_threshold=80):
    if not parts: return []
    parts.sort(key=lambda x: x['conf'], reverse=True)
    final_parts = []
    for current in parts:
        is_duplicate = False
        for kept in final_parts:
            iou = calculate_iou(current['box'], kept['box'])
            cx1, cy1 = current['center']
            cx2, cy2 = kept['center']
            dist = math.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)
            if iou > 0.1 or dist < distance_threshold:
                is_duplicate = True; break
        if not is_duplicate: final_parts.append(current)
    return final_parts

def solve_overlap_real_v35(parts, dist_thresh=60, iou_thresh=0.4):
    if not parts: return []
    parts.sort(key=lambda x: x.get('conf', 0), reverse=True)
    final = []
    for curr in parts:
        is_dup = False
        for k in final:
            x1 = max(curr['box'][0], k['box'][0])
            y1 = max(curr['box'][1], k['box'][1])
            x2 = min(curr['box'][2], k['box'][2])
            y2 = min(curr['box'][3], k['box'][3])
            inter_area = max(0, x2-x1) * max(0, y2-y1)
            area_curr = (curr['box'][2]-curr['box'][0]) * (curr['box'][3]-curr['box'][1])
            area_k = (k['box'][2]-k['box'][0]) * (k['box'][3]-k['box'][1])
            min_area = min(area_curr, area_k)
            ratio = inter_area / min_area if min_area > 0 else 0
            iou = calculate_iou(curr['box'], k['box'])
            if ratio > 0.8: is_dup = True; break
            if iou > iou_thresh: is_dup = True; break
            dist = math.sqrt((curr['center'][0]-k['center'][0])**2 + (curr['center'][1]-k['center'][1])**2)
            if dist < dist_thresh: is_dup = True; break
        if not is_dup: final.append(curr)
    return final

# ==========================================
# [분석 1] 회로도 (유지)
# ==========================================
def analyze_schematic(img, model):
    results = model.predict(source=img, save=False, conf=0.05, verbose=False)
    boxes = results[0].boxes
    raw_parts = []
    
    for box in boxes:
        cls_id = int(box.cls[0])
        name = model.names[cls_id].lower()
        conf = float(box.conf[0])
        coords = box.xyxy[0].tolist()
        center = get_center(coords)
        
        base_name = name.split('_')[0].split(' ')[0]
        if base_name in ['vdc', 'vsource', 'battery', 'voltage', 'v']: base_name = 'source'
        if base_name in ['cap', 'c', 'capacitor']: base_name = 'capacitor'
        if base_name in ['res', 'r', 'resistor']: base_name = 'resistor'
        
        raw_parts.append({'name': base_name, 'box': coords, 'center': center, 'conf': conf})

    clean_parts = solve_overlap_schematic_v48(raw_parts)

    if clean_parts:
        has_source = any(p['name'] == 'source' for p in clean_parts)
        if not has_source:
            leftmost_part = min(clean_parts, key=lambda p: p['center'][0])
            leftmost_part['name'] = 'source'

    summary = {'total': 0, 'details': {}}
    for part in clean_parts:
        name = part['name']
        x1, y1, x2, y2 = map(int, part['box'])
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img, f"{name}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        summary['total'] += 1
        summary['details'][name] = summary['details'].get(name, 0) + 1
        
    return img, summary

# ==========================================
# [분석 2] 실물 보드 (엔지니어링 로직 적용)
# ==========================================
def analyze_real_ensemble_engineering(img, model_list):
    h, w, _ = img.shape
    raw_bodies = []
    raw_pins = [] 
    
    # 1. 앙상블 탐지 (기존과 동일)
    for model in model_list:
        res = model.predict(source=img, conf=0.10, verbose=False)
        boxes = res[0].boxes
        for b in boxes:
            name = model.names[int(b.cls[0])].lower()
            coords = b.xyxy[0].tolist()
            center = get_center(coords)
            conf = float(b.conf[0])
            
            if 'cap' in name: min_conf = 0.15
            elif 'res' in name: min_conf = 0.60
            elif 'wire' in name: min_conf = 0.15
            else: min_conf = 0.25
                
            if conf < min_conf: continue

            if any(x in name for x in ['pin', 'leg', 'lead']) and 'wire' not in name:
                raw_pins.append({'center': center, 'box': coords})
            else:
                raw_bodies.append({'name': name, 'box': coords, 'center': center, 'conf': conf})

    # 2. 중복 제거
    clean_bodies = solve_overlap_real_v35(raw_bodies)
    
    # ----------------------------------------------------
    # [NEW] Engineering Logic Start
    # ----------------------------------------------------
    
    # A. 브레드보드 영역 식별 (Dynamic Calibration)
    breadboard_box = [0, 0, w, h] # Default: 전체 화면
    for comp in clean_bodies:
        if 'breadboard' in comp['name']:
            breadboard_box = comp['box']
            break
            
    bb_x1, bb_y1, bb_x2, bb_y2 = breadboard_box
    bb_h = bb_y2 - bb_y1
    
    # B. 전원 레일(Power Rail) 영역 정의 (상하단 15% 가정)
    # 실제 브레드보드 내에서의 좌표로 전원 연결 여부 판단
    power_rail_top_y = bb_y1 + (bb_h * 0.15)
    power_rail_bot_y = bb_y2 - (bb_h * 0.15)
    
    # C. 그래프(회로망) 생성 및 부품 추가
    circuit = CircuitGraph()
    
    # 탐지된 핀(Pin)들을 노드 생성의 힌트로 사용
    # (핀 객체 자체가 노드 위치를 의미하므로 먼저 등록)
    for pin in raw_pins:
        circuit.find_node(pin['center'])
        
    # 부품들을 회로망에 연결
    for comp in clean_bodies:
        if 'breadboard' in comp['name']: continue
        # 부품의 단자(Terminals)를 추정하여 회로 그래프에 추가
        circuit.add_component(comp['name'], comp['box'])
    
    # D. 전원 노드 식별 (영역 기반)
    power_nodes = set()
    for i, node_points in enumerate(circuit.nodes):
        # 노드 그룹 내 포인트들의 평균 Y값
        avg_y = sum(p[1] for p in node_points) / len(node_points)
        if avg_y < power_rail_top_y or avg_y > power_rail_bot_y:
            power_nodes.add(i)
            
    # E. 연결성 분석 실행
    circuit.analyze_connectivity(power_nodes)
    
    # ----------------------------------------------------
    # [NEW] Visualization
    # ----------------------------------------------------
    summary = {'total': 0, 'on': 0, 'off': 0, 'details': {}}
    
    # 매칭된 circuit component 정보를 시각화
    for comp_info in circuit.components:
        name = comp_info['name']
        box = comp_info['box']
        status = comp_info['status']
        
        # 이름 정규화
        norm_name = name
        label_name = name[:3].upper()
        if 'res' in name: norm_name = 'resistor'; label_name="RES"
        elif 'cap' in name: norm_name = 'capacitor'; label_name="CAP"
        elif 'wire' in name: label_name="WIRE"

        if 'wire' not in name:
            if norm_name not in summary['details']: summary['details'][norm_name] = {'count': 0}
            summary['details'][norm_name]['count'] += 1
        
        # 상태에 따른 색상 및 텍스트
        if status == 'CONNECTED':
            color = (0, 255, 0) # Green
            state_text = "OK"
            summary['on'] += 1
        elif status == 'SHORT':
            color = (0, 0, 255) # Red
            state_text = "SHORT!"
            summary['off'] += 1
        elif status == 'OPEN':
            color = (0, 165, 255) # Orange
            state_text = "OPEN?" # 연결 안됨
            summary['off'] += 1
        else:
            color = (128, 128, 128)
            state_text = "?"

        summary['total'] += 1
        
        # Draw Box
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Draw Label
        cv2.putText(img, f"{label_name}:{state_text}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # [Visual Debug] 부품의 단자(Terminal) 위치 표시 (작은 원)
        for t in comp_info['terminals']:
            cv2.circle(img, (int(t[0]), int(t[1])), 3, (255, 255, 0), -1)

    return img, summary

# ==========================================
# [Main UI]
# ==========================================
st.title("🧠 BrainBoard V54: Engineering Edition")
st.markdown("""
### 🛡️ 엔지니어링 정밀 검증 시스템
- **회로망(Netlist) 분석**: 단순 거리 측정이 아닌, 노드(Node) 기반 연결성 판단
- **단락(Short) / 단선(Open) 감지**: 잘못된 연결(Short)이나 끊어진 연결(Open)을 감지
- **동적 보드 인식**: 브레드보드 위치에 맞춰 전원부를 자동 보정
""")

@st.cache_resource
def load_models():
    real_models = []
    loaded_names = []
    try:
        for path in REAL_MODEL_PATHS:
            try:
                model = YOLO(path)
                real_models.append(model)
                loaded_names.append(path)
            except Exception:
                continue
        sym_model = YOLO(MODEL_SYM_PATH)
    except Exception as e:
        return [], None
    return real_models, sym_model

models_real, model_sym = load_models()

if not models_real:
    st.error("❌ 모델 파일(best.pt)을 찾을 수 없습니다. 경로를 확인해주세요.")
    st.stop()
else:
    st.sidebar.success(f"✅ 시스템 준비 완료\n- 활성 모델: {len(models_real)}개")

col1, col2 = st.columns(2)
ref_file = col1.file_uploader("1. 회로도(Schematic)", type=['jpg', 'png', 'jpeg'])
tgt_file = col2.file_uploader("2. 실물 사진(Real)", type=['jpg', 'png', 'jpeg'])

if ref_file and tgt_file:
    ref_image = Image.open(ref_file)
    tgt_image = Image.open(tgt_file)
    ref_cv = cv2.cvtColor(np.array(ref_image), cv2.COLOR_RGB2BGR)
    tgt_cv = cv2.cvtColor(np.array(tgt_image), cv2.COLOR_RGB2BGR)

    if st.button("🚀 정밀 회로 검증 시작"):
        with st.spinner("회로망 분석 및 부품 스펙 검증 중..."):
            
            # 1. 회로도 분석
            res_ref_img, ref_data = analyze_schematic(ref_cv.copy(), model_sym)
            
            # 2. 실물 분석 (Engineering Logic 적용)
            res_tgt_img, tgt_data = analyze_real_ensemble_engineering(tgt_cv.copy(), models_real)

            # 3. 결과 비교 및 리포트
            st.divider()
            col_res1, col_res2 = st.columns(2)
            
            with col_res1:
                st.image(cv2.cvtColor(res_ref_img, cv2.COLOR_BGR2RGB), caption="회로도 인식 결과", use_column_width=True)
                st.info(f"📄 회로도 부품 수: {ref_data['total']}개")
                
            with col_res2:
                st.image(cv2.cvtColor(res_tgt_img, cv2.COLOR_BGR2RGB), caption="실물 검증 결과 (Engineering Mode)", use_column_width=True)
                
                # 상태별 카운트 표시
                n_short = sum(1 for c in tgt_data['details'] if 'SHORT' in str(c)) # 단순 카운트용 로직 필요 시 수정
                st.info(f"📸 실물 인식: {tgt_data['total']}개 (정상: {tgt_data['on']}, 이상: {tgt_data['off']})")

            # 상세 진단 리포트
            st.subheader("📋 엔지니어링 진단 리포트")
            
            all_parts = set(ref_data['details'].keys()) | set(tgt_data['details'].keys())
            
            for part in all_parts:
                if part in ['text', 'hole', 'source', 'breadboard']: continue
                
                ref_c = ref_data['details'].get(part, 0)
                tgt_c = tgt_data['details'].get(part, {}).get('count', 0)
                
                if ref_c == tgt_c:
                    st.success(f"✅ **{part.upper()}**: 개수 일치 ({ref_c}개)")
                else:
                    st.error(f"⚠️ **{part.upper()}**: 개수 불일치 (회로도 {ref_c} vs 실물 {tgt_c})")
            
            if tgt_data['off'] > 0:
                st.warning("""
                **⚠️ 회로 이상 감지됨:**
                - **OPEN?**: 부품의 한쪽 다리가 연결되지 않았거나, 감지되지 않았습니다.
                - **SHORT!**: 부품의 양쪽 다리가 같은 라인(Node)에 연결되었습니다. (합선 위험)
                """)
            else:
                st.balloons()
                st.success("🎉 모든 부품이 회로적으로 올바르게 연결되었습니다!")
