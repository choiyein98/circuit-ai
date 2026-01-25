import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import math
from PIL import Image
from collections import defaultdict
import itertools

# ==========================================
# [설정] BrainBoard V59: Circuit Theory Edition
# ==========================================
st.set_page_config(page_title="BrainBoard V59: Theory Check", layout="wide")

REAL_MODEL_PATHS = ['best.pt', 'best(2).pt', 'best(3).pt']
MODEL_SYM_PATH = 'symbol.pt'

# ==========================================
# [Helper Functions] 기하학 및 유틸
# ==========================================
def get_center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = ((box1[2]-box1[0])*(box1[3]-box1[1])) + ((box2[2]-box2[0])*(box2[3]-box2[1])) - inter
    return inter / union if union > 0 else 0

def get_x_overlap_ratio(box1, box2):
    x1_max = max(box1[0], box2[0]); x2_min = min(box1[2], box2[2])
    return max(0, x2_min - x1_max) / (box1[2] - box1[0])

# 중복 제거 (앙상블용)
def solve_overlap_real(parts):
    if not parts: return []
    parts.sort(key=lambda x: x.get('conf', 0), reverse=True)
    final = []
    for curr in parts:
        is_dup = False
        for k in final:
            iou = calculate_iou(curr['box'], k['box'])
            dist = math.sqrt((curr['center'][0]-k['center'][0])**2 + (curr['center'][1]-k['center'][1])**2)
            if 'leg' not in curr['name'] and (iou > 0.4 or dist < 60): 
                is_dup = True; break
        if not is_dup: final.append(curr)
    return final

# 관계 키 생성 (알파벳순 정렬)
def get_rel_key(n1, n2):
    return " <-> ".join(sorted([n1, n2]))

# ==========================================
# [Core Logic] 회로이론 기반 위상 추출
# ==========================================
def extract_circuit_topology(parts, connections):
    """
    부품 리스트와 연결 정보(공유 노드 수)를 바탕으로
    회로이론적 관계(직렬/병렬)를 정의합니다.
    """
    topology = []
    
    # 모든 부품 쌍에 대해 조사 (Combination)
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            p1 = parts[i]
            p2 = parts[j]
            
            p1_name = p1['name'].split('_')[0]
            p2_name = p2['name'].split('_')[0]
            
            # 이름 정규화
            if 'res' in p1_name: p1_name = 'resistor'
            if 'cap' in p1_name: p1_name = 'capacitor'
            if 'res' in p2_name: p2_name = 'resistor'
            if 'cap' in p2_name: p2_name = 'capacitor'
            
            # 와이어는 도선(Ideal Wire)이므로 부품 관계에서 제외 (노드로 흡수됨)
            if 'wire' in p1_name or 'wire' in p2_name: continue

            # 공유하는 노드 수 확인
            # connections는 {부품인덱스: {노드ID 집합}}
            nodes_1 = connections[i]
            nodes_2 = connections[j]
            shared_nodes = nodes_1.intersection(nodes_2)
            num_shared = len(shared_nodes)
            
            rel_type = None
            
            # [회로이론 정의 적용]
            if num_shared >= 2:
                # 두 노드를 모두 공유함 -> 병렬 (Parallel)
                rel_type = 'Parallel'
            elif num_shared == 1:
                # 한 노드만 공유함 -> 직렬 (Series) 가능성
                # 엄밀한 직렬: 해당 노드(KCL Node)에 이 두 부품 외에 다른 것이 없어야 함
                # 하지만 비전 인식 한계상 '연결됨(Series)' 정도로만 판단해도 충분
                rel_type = 'Series'
                
                # (심화) 전원과 연결된 경우 예외 처리
                if -1 in shared_nodes: # -1은 전원 노드
                    rel_type = 'Connected (Power)'
            
            if rel_type:
                topology.append({
                    'key': get_rel_key(p1_name, p2_name),
                    'type': rel_type,
                    'p1_idx': i, 'p2_idx': j
                })
    return topology

# ==========================================
# [분석 1] 회로도 (Schematic)
# ==========================================
def analyze_schematic(img, model):
    results = model.predict(source=img, save=False, conf=0.05, verbose=False)
    raw_parts = []
    for box in results[0].boxes:
        name = model.names[int(box.cls[0])].lower()
        coords = box.xyxy[0].tolist()
        base_name = name.split('_')[0].split(' ')[0]
        if base_name in ['vdc', 'vsource', 'battery', 'voltage', 'v']: base_name = 'source'
        if base_name in ['cap', 'c', 'capacitor']: base_name = 'capacitor'
        if base_name in ['res', 'r', 'resistor']: base_name = 'resistor'
        raw_parts.append({'name': base_name, 'box': coords, 'center': get_center(coords), 'conf': float(box.conf[0])})

    # 중복 제거
    parts = []
    raw_parts.sort(key=lambda x: x['conf'], reverse=True)
    for p in raw_parts:
        if not any(calculate_iou(p['box'], k['box']) > 0.1 for k in parts): parts.append(p)

    if parts and not any('source' in p['name'] for p in parts):
         leftmost = min(parts, key=lambda p: p['center'][0])
         leftmost['name'] = 'source'

    # 가상 연결 정보 생성 (기하학적 위치 기반)
    # 회로도는 좌표가 곧 위상(Topology)
    connections = defaultdict(set)
    
    # 1. 위아래 겹침 (병렬) -> 가상 노드 ID 부여 (100, 101)
    # 2. 좌우 인접 (직렬) -> 가상 노드 ID 부여 (200)
    
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            p1, p2 = parts[i], parts[j]
            overlap = get_x_overlap_ratio(p1['box'], p2['box'])
            
            if overlap > 0.3: # 병렬
                # 병렬이면 두 노드를 공유해야 함
                node_a = i * 100 + j # 임의의 노드 ID 생성
                node_b = i * 100 + j + 1
                connections[i].add(node_a); connections[i].add(node_b)
                connections[j].add(node_a); connections[j].add(node_b)
            
            elif abs(p1['center'][1] - p2['center'][1]) < 100 and abs(p1['center'][0] - p2['center'][0]) < 300: # 직렬
                node_common = i * 200 + j
                connections[i].add(node_common)
                connections[j].add(node_common)

    topology = extract_circuit_topology(parts, connections)
    
    # 시각화 (Clean View)
    for p in parts:
        x1, y1, x2, y2 = map(int, p['box'])
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img, p['name'], (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    return img, {'parts': parts, 'topology': topology}

# ==========================================
# [분석 2] 실물 보드 (Real Board)
# ==========================================
def analyze_real(img, model_list):
    h, w, _ = img.shape
    raw_bodies = []
    raw_legs = [] 

    # 1. 앙상블 탐지
    for model in model_list:
        res = model.predict(source=img, conf=0.10, verbose=False)
        for b in res[0].boxes:
            name = model.names[int(b.cls[0])].lower()
            coords = b.xyxy[0].tolist()
            conf = float(b.conf[0])
            
            if 'cap' in name and conf < 0.15: continue
            if 'res' in name and conf < 0.25: continue
            if 'breadboard' in name: continue
            
            center = get_center(coords)
            if any(x in name for x in ['pin', 'leg', 'lead']):
                raw_legs.append({'box': coords, 'center': center, 'name': 'leg'})
            else: 
                raw_bodies.append({'name': name, 'box': coords, 'center': center, 'conf': conf})

    parts = solve_overlap_real(raw_bodies)

    # 2. 전원(Source) 복구 로직 (KCL 소스 노드)
    TOP_RAIL = h * 0.20; BOTTOM_RAIL = h * 0.80
    has_source = False
    if any(p['name'] in ['source', 'battery', 'voltage'] for p in parts): has_source = True
    
    # 와이어/핀이 레일에 닿으면 전원 있는 것으로 간주
    if not has_source:
        for p in parts + raw_legs:
            if p['center'][1] < TOP_RAIL or p['center'][1] > BOTTOM_RAIL:
                if 'wire' in p['name'] or 'leg' in p['name']:
                    has_source = True; break
    
    if has_source and not any(p['name'] == 'source' for p in parts):
        parts.append({'name': 'source', 'box': [0,0,0,0], 'center': (0,0), 'conf': 1.0})

    # 3. 노드(Node) 식별 - 브레드보드 컬럼 클러스터링
    # 같은 세로줄(Column)에 있는 핀들은 '같은 전기적 노드'임
    grouped_legs = []
    for leg in raw_legs:
        assigned = False
        for group in grouped_legs:
            ref = group[0]
            # X오차 < 25px (같은 줄), Y오차 < 80px (같은 블록)
            if abs(leg['center'][0] - ref['center'][0]) < 25 and abs(leg['center'][1] - ref['center'][1]) < 80:
                group.append(leg); assigned = True; break
        if not assigned: grouped_legs.append([leg])

    # 4. 부품과 노드의 연결 (Incidence Matrix 개념)
    # connections[i] = {노드ID_1, 노드ID_2} (부품 i가 연결된 노드들)
    connections = defaultdict(set)
    
    for i, part in enumerate(parts):
        # 전원은 모든 레일 노드에 연결된 것으로 간주 (전역 노드 -1)
        if part['name'] == 'source':
            connections[i].add(-1) 
            continue
            
        # 부품 근처의 핀을 찾아서 해당 핀이 속한 노드 ID를 부여
        for nid, group in enumerate(grouped_legs):
            for leg in group:
                dist = math.sqrt((part['center'][0]-leg['center'][0])**2 + (part['center'][1]-leg['center'][1])**2)
                diag = math.sqrt((part['box'][2]-part['box'][0])**2 + (part['box'][3]-part['box'][1])**2)
                # 부품 몸통 근처에 핀이 있으면 연결
                if dist < max(60, diag * 1.0):
                    connections[i].add(nid)
                    
        # (예외처리) 전원 레일 근처에 있으면 전원 노드(-1) 추가
        if part['box'][2] > 0: # 가상부품 아님
             if part['center'][1] < TOP_RAIL or part['center'][1] > BOTTOM_RAIL:
                 connections[i].add(-1)

    topology = extract_circuit_topology(parts, connections)

    # 시각화 (Clean View)
    summary = {'parts': parts, 'topology': topology}
    for p in parts:
        norm_name = p['name']
        if 'res' in norm_name: norm_name = 'resistor'
        elif 'cap' in norm_name: norm_name = 'capacitor'
        if 'wire' in norm_name: continue # 와이어는 화면 표시 X

        if p['box'][2] > 0: 
            color = (0, 255, 0)
            if 'source' in norm_name: color = (0, 255, 255)
            x1, y1, x2, y2 = map(int, p['box'])
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            cv2.putText(img, norm_name[:3].upper(), (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return img, summary

# ==========================================
# [Main UI]
# ==========================================
st.title("🧠 BrainBoard V59: Circuit Theory Verifier")
st.markdown("### ⚡ 회로이론(KCL, Topology) 기반 정밀 검증")

@st.cache_resource
def load_models():
    reals = []
    try:
        for p in REAL_MODEL_PATHS: reals.append(YOLO(p))
    except: pass
    return reals, YOLO(MODEL_SYM_PATH)

try:
    models_real, model_sym = load_models()
    if not models_real: st.stop()
    st.sidebar.success("✅ 시스템 준비 완료")
except: st.stop()

col1, col2 = st.columns(2)
ref_file = col1.file_uploader("1. 회로도", type=['jpg', 'png', 'jpeg'])
tgt_file = col2.file_uploader("2. 실물 사진", type=['jpg', 'png', 'jpeg'])

if ref_file and tgt_file:
    ref_image = Image.open(ref_file)
    tgt_image = Image.open(tgt_file)
    ref_cv = cv2.cvtColor(np.array(ref_image), cv2.COLOR_RGB2BGR)
    tgt_cv = cv2.cvtColor(np.array(tgt_image), cv2.COLOR_RGB2BGR)

    if st.button("🚀 회로이론 분석 실행"):
        with st.spinner("회로망(Network) 해석 중..."):
            
            res_ref_img, ref_data = analyze_schematic(ref_cv.copy(), model_sym)
            res_tgt_img, tgt_data = analyze_real(tgt_cv.copy(), models_real)

            # 1. 부품 개수 (BOM Check)
            st.subheader("1. 부품 구성 확인 (BOM)")
            ref_counts = defaultdict(int)
            tgt_counts = defaultdict(int)
            for p in ref_data['parts']: ref_counts[p['name']] += 1
            for p in tgt_data['parts']: tgt_counts[p['name']] += 1
            
            all_keys = set(ref_counts.keys()) | set(tgt_counts.keys())
            for k in all_keys:
                if k == 'wire': continue
                r = ref_counts[k]; t = tgt_counts[k]
                if r != t: st.error(f"⚠️ {k} 개수 불일치 ({r} vs {t})")
                else: st.success(f"✅ {k} 개수 일치")

            # 2. 토폴로지 비교 (Topology Check)
            st.subheader("2. 회로 토폴로지 검증 (Circuit Theory)")
            
            # 비교 로직
            # 키(부품쌍)와 타입(직/병렬)을 Set으로 변환
            ref_set = set((item['key'], item['type']) for item in ref_data['topology'])
            tgt_set = set((item['key'], item['type']) for item in tgt_data['topology'])
            
            matches = []
            errors = []
            
            # 회로도에 있는 관계가 실물에 있는가?
            for r_item in ref_data['topology']:
                key = r_item['key']
                r_type = r_item['type']
                
                # 실물에서 같은 키 찾기
                found_type = None
                for t_item in tgt_data['topology']:
                    if t_item['key'] == key:
                        found_type = t_item['type']
                        break
                
                if found_type:
                    if found_type == r_type:
                        matches.append(f"✅ [Pass] {key} : {r_type} 연결 일치")
                    else:
                        # Connected(Power)는 Series의 일종으로 봐줌 (유연성)
                        if "Connected" in found_type and "Series" in r_type:
                             matches.append(f"✅ [Pass] {key} : 전원 연결 확인됨")
                        else:
                             errors.append(f"🚫 [Mismatch] {key} : 회로도는 {r_type}이나 실물은 {found_type}임")
                else:
                    errors.append(f"❌ [Open] {key} : 실물에서 연결 끊김")

            if not errors and len(matches) > 0:
                st.success("🎉 회로이론상 완벽하게 일치하는 회로입니다!")
                st.balloons()
            elif not matches and not errors:
                st.info("ℹ️ 감지된 부품 간 연결 관계가 없습니다.")

            for m in matches: st.caption(m)
            for e in errors: st.error(e)

            st.image(cv2.cvtColor(res_ref_img, cv2.COLOR_BGR2RGB), caption="회로도 분석", use_column_width=True)
            st.image(cv2.cvtColor(res_tgt_img, cv2.COLOR_BGR2RGB), caption="실물 분석 (Clean View)", use_column_width=True)
