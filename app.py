import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import math
from PIL import Image

# ==========================================
# [설정] BrainBoard V54: Topology & Sequence
# ==========================================
st.set_page_config(page_title="BrainBoard V54: Topology", layout="wide")

REAL_MODEL_PATHS = ['best.pt', 'best(2).pt', 'best(3).pt']
MODEL_SYM_PATH = 'symbol.pt'
LEG_EXTENSION_RANGE = 180

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

def get_center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

# [NEW] 부품 순서 추출 함수 (왼쪽 -> 오른쪽 순서)
def get_component_sequence(parts):
    # X좌표 기준으로 정렬
    sorted_parts = sorted(parts, key=lambda x: x['center'][0])
    # 이름만 추출 (예: ['source', 'resistor', 'capacitor'])
    sequence = [p['name'] for p in sorted_parts if p['name'] not in ['wire', 'text', 'breadboard']]
    return sequence, sorted_parts

# [NEW] 브레드보드 노드(Row) 계산 함수
# Y좌표가 비슷하고(같은 줄), X좌표가 가까우면 같은 노드로 간주
def check_breadboard_connection(comp1, comp2, threshold_y=20, threshold_x=100):
    c1 = comp1['center']
    c2 = comp2['center']
    
    # Y좌표 차이가 작아야 함 (같은 행)
    if abs(c1[1] - c2[1]) < threshold_y:
        # X좌표 거리도 적당히 가까워야 함 (너무 멀면 다른 구멍)
        if abs(c1[0] - c2[0]) < threshold_x:
            return True
    return False

# ==========================================
# [중복 제거]
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
            iou = calculate_iou(curr['box'], k['box'])
            dist = math.sqrt((curr['center'][0]-k['center'][0])**2 + (curr['center'][1]-k['center'][1])**2)
            if iou > iou_thresh or dist < dist_thresh: is_dup = True; break
        if not is_dup: final.append(curr)
    return final

# ==========================================
# [분석 1] 회로도
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
        
        base_name = name.split('_')[0].split(' ')[0]
        if base_name in ['vdc', 'vsource', 'battery', 'voltage', 'v']: base_name = 'source'
        if base_name in ['cap', 'c', 'capacitor']: base_name = 'capacitor'
        if base_name in ['res', 'r', 'resistor']: base_name = 'resistor'
        
        raw_parts.append({'name': base_name, 'box': coords, 'center': get_center(coords), 'conf': conf})

    clean_parts = solve_overlap_schematic_v48(raw_parts)

    # 전원 보정
    if clean_parts:
        has_source = any(p['name'] == 'source' for p in clean_parts)
        if not has_source:
            leftmost_part = min(clean_parts, key=lambda p: p['center'][0])
            leftmost_part['name'] = 'source'

    # 시각화
    summary = {'total': 0, 'details': {}, 'parts_list': clean_parts}
    for part in clean_parts:
        name = part['name']
        x1, y1, x2, y2 = map(int, part['box'])
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img, name, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        if name not in summary['details']: summary['details'][name] = 0
        summary['details'][name] += 1
        
    return img, summary

# ==========================================
# [분석 2] 실물 보드
# ==========================================
def analyze_real_ensemble(img, model_list):
    h, w, _ = img.shape
    raw_bodies = []
    
    # 앙상블 예측
    for model in model_list:
        res = model.predict(source=img, conf=0.10, verbose=False)
        for b in res[0].boxes:
            name = model.names[int(b.cls[0])].lower()
            coords = b.xyxy[0].tolist()
            conf = float(b.conf[0])
            
            if 'cap' in name and conf < 0.15: continue
            if 'res' in name and conf < 0.60: continue
            
            if 'breadboard' in name or 'hole' in name: continue
            if 'wire' in name: continue # 순수 부품 연결만 보기 위해 와이어 몸통은 제외

            raw_bodies.append({'name': name, 'box': coords, 'center': get_center(coords), 'conf': conf})

    clean_bodies = solve_overlap_real_v35(raw_bodies)

    # [NEW] 연결성(Connectivity) 분석 알고리즘
    # 각 부품이 "고립(Isolated)" 되었는지, 아니면 "연결(Connected)" 되었는지 판단
    for comp in clean_bodies:
        comp['is_connected'] = False
        comp['neighbors'] = []

    # O(N^2) 비교로 서로 가까운(같은 노드) 부품 찾기
    for i, c1 in enumerate(clean_bodies):
        for j, c2 in enumerate(clean_bodies):
            if i == j: continue
            # 두 부품이 브레드보드 상에서 연결되어 보이는지 확인
            if check_breadboard_connection(c1, c2):
                c1['is_connected'] = True
                c1['neighbors'].append(c2['name'])

    summary = {'total': 0, 'details': {}, 'parts_list': clean_bodies}
    
    for comp in clean_bodies:
        raw_name = comp['name']
        norm_name = raw_name
        if 'res' in raw_name: norm_name = 'resistor'
        elif 'cap' in raw_name: norm_name = 'capacitor'
        
        comp['name'] = norm_name # 이름 정규화

        if norm_name not in summary['details']: summary['details'][norm_name] = {'count': 0}
        summary['details'][norm_name]['count'] += 1

        # 시각화: 연결됐으면 초록색, 끊어졌으면(고립) 빨간색
        color = (0, 255, 0) if comp['is_connected'] else (0, 0, 255)
        status = "LINKED" if comp['is_connected'] else "OPEN"
        
        # 전원 부품은 무조건 ON 처리 (기준점)
        if 'source' in norm_name or 'batt' in norm_name:
            color = (0, 255, 0); status = "PWR"

        x1, y1, x2, y2 = map(int, comp['box'])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.putText(img, f"{norm_name[:3].upper()}:{status}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # 연결선 그리기 (시각적 확인)
        if comp['is_connected']:
            for neighbor in clean_bodies:
                if neighbor['name'] in comp['neighbors']:
                    cv2.line(img, (int(comp['center'][0]), int(comp['center'][1])), 
                             (int(neighbor['center'][0]), int(neighbor['center'][1])), (0, 255, 255), 2)

    return img, summary

# ==========================================
# [Main UI]
# ==========================================
st.title("🧠 BrainBoard V54: Topology Check")
st.markdown("### 🔍 부품 개수 + 연결 순서 + 회로 끊김(Open) 동시 분석")

@st.cache_resource
def load_models():
    real_models = []
    try:
        for path in REAL_MODEL_PATHS: real_models.append(YOLO(path))
    except: pass
    return real_models, YOLO(MODEL_SYM_PATH)

try:
    models_real, model_sym = load_models()
    if not models_real: st.stop()
    st.sidebar.success(f"✅ 모델 로드 완료")
except: st.stop()

col1, col2 = st.columns(2)
ref_file = col1.file_uploader("1. 회로도", type=['jpg', 'png', 'jpeg'])
tgt_file = col2.file_uploader("2. 실물 사진", type=['jpg', 'png', 'jpeg'])

if ref_file and tgt_file:
    ref_image = Image.open(ref_file)
    tgt_image = Image.open(tgt_file)
    ref_cv = cv2.cvtColor(np.array(ref_image), cv2.COLOR_RGB2BGR)
    tgt_cv = cv2.cvtColor(np.array(tgt_image), cv2.COLOR_RGB2BGR)

    if st.button("🚀 정밀 알고리즘 분석"):
        with st.spinner("회로의 위상(Topology)을 분석 중..."):
            
            res_ref_img, ref_data = analyze_schematic(ref_cv.copy(), model_sym)
            res_tgt_img, tgt_data = analyze_real_ensemble(tgt_cv.copy(), models_real)

            # 1. 부품 개수 비교
            issues = []
            all_parts = set(ref_data['details'].keys()) | set(tgt_data['details'].keys())
            counts_match = True
            
            st.subheader("1. 부품 개수 검증 (Counting)")
            for part in all_parts:
                if part in ['wire', 'breadboard', 'text', 'source']: continue
                ref_c = ref_data['details'].get(part, 0)
                tgt_c = tgt_data['details'].get(part, {}).get('count', 0)
                
                if ref_c != tgt_c:
                    st.error(f"⚠️ {part.capitalize()} 개수 불일치 (회로도:{ref_c} vs 실물:{tgt_c})")
                    counts_match = False
                else:
                    st.success(f"✅ {part.capitalize()} 개수 일치 ({ref_c}개)")

            # 2. 순서 비교 (Sequence Check)
            st.subheader("2. 배치 순서 검증 (Sequence)")
            ref_seq, _ = get_component_sequence(ref_data['parts_list'])
            tgt_seq, _ = get_component_sequence(tgt_data['parts_list'])
            
            # 비교를 위해 와이어 등 불필요한 것 제거한 순수 순서
            st.code(f"회로도 순서 (Left->Right): {ref_seq}")
            st.code(f"실물보드 순서 (Left->Right): {tgt_seq}")

            # 순서가 비슷한지 간단 체크 (완전히 같을 필요는 없지만, 구성이 달라지면 경고)
            # 가장 긴 공통 부분 수열(LCS) 같은 복잡한 것보다, 단순히 구성 요소 순서 비교
            if ref_seq == tgt_seq:
                st.success("🎉 부품 배치 순서가 회로도와 완벽하게 일치합니다!")
            else:
                st.warning("⚠️ 부품 배치 순서가 다릅니다. 위치를 확인해주세요.")

            # 3. 연결 상태 (Connectivity)
            st.subheader("3. 연결 상태 검증 (Connectivity)")
            disconnected_count = 0
            for comp in tgt_data['parts_list']:
                if not comp['is_connected'] and 'source' not in comp['name']:
                    disconnected_count += 1
            
            if disconnected_count == 0:
                st.success("🔌 모든 부품이 전기적으로 잘 연결되어 있습니다.")
            else:
                st.error(f"❌ {disconnected_count}개의 부품이 연결되지 않고 떠있습니다(Open). 사진의 빨간 박스를 확인하세요.")

            # 이미지 출력
            st.image(cv2.cvtColor(res_ref_img, cv2.COLOR_BGR2RGB), caption="회로도 분석", use_column_width=True)
            st.image(cv2.cvtColor(res_tgt_img, cv2.COLOR_BGR2RGB), caption="실물 연결 상태 분석 (노란 선=연결됨)", use_column_width=True)
