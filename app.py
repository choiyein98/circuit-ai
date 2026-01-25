import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import math
from PIL import Image
from collections import defaultdict

# ==========================================
# [설정] BrainBoard V60: Stabilization
# ==========================================
st.set_page_config(page_title="BrainBoard V60: Stable", layout="wide")

REAL_MODEL_PATHS = ['best.pt', 'best(2).pt', 'best(3).pt']
MODEL_SYM_PATH = 'symbol.pt'

# ==========================================
# [Helper Functions]
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

# [핵심] 이름 정규화 함수 (모든 _body, _head 등을 통일)
def normalize_name(name):
    name = name.lower()
    if 'res' in name: return 'resistor'
    if 'cap' in name: return 'capacitor'
    if 'wire' in name: return 'wire'
    if any(x in name for x in ['source', 'batt', 'volt', 'vdc']): return 'source'
    if any(x in name for x in ['leg', 'pin', 'lead']): return 'leg'
    return name

def solve_overlap_real(parts):
    if not parts: return []
    parts.sort(key=lambda x: x.get('conf', 0), reverse=True)
    final = []
    for curr in parts:
        is_dup = False
        for k in final:
            iou = calculate_iou(curr['box'], k['box'])
            dist = math.sqrt((curr['center'][0]-k['center'][0])**2 + (curr['center'][1]-k['center'][1])**2)
            # 핀(leg)이 아니면 중복 제거
            if curr['name'] != 'leg' and (iou > 0.4 or dist < 60): 
                is_dup = True; break
        if not is_dup: final.append(curr)
    return final

def get_rel_key(n1, n2):
    return " <-> ".join(sorted([n1, n2]))

# ==========================================
# [Core Logic] 회로이론 위상 추출
# ==========================================
def extract_circuit_topology(parts, connections):
    topology = []
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            p1_name = parts[i]['name']
            p2_name = parts[j]['name']
            
            # 와이어 자체는 관계 분석에서 제외 (노드로 흡수됨)
            if p1_name == 'wire' or p2_name == 'wire': continue

            nodes_1 = connections[i]
            nodes_2 = connections[j]
            shared_nodes = nodes_1.intersection(nodes_2)
            num_shared = len(shared_nodes)
            
            rel_type = None
            if num_shared >= 2:
                rel_type = 'Parallel'
            elif num_shared == 1:
                rel_type = 'Series'
                if -1 in shared_nodes: rel_type = 'Connected (Power)'
            
            if rel_type:
                topology.append({
                    'key': get_rel_key(p1_name, p2_name),
                    'type': rel_type
                })
    return topology

# ==========================================
# [분석 1] 회로도 (Schematic)
# ==========================================
def analyze_schematic(img, model):
    results = model.predict(source=img, save=False, conf=0.05, verbose=False)
    raw_parts = []
    for box in results[0].boxes:
        # 1. 감지 및 이름 즉시 정규화
        raw_name = model.names[int(box.cls[0])]
        norm_name = normalize_name(raw_name)
        coords = box.xyxy[0].tolist()
        raw_parts.append({'name': norm_name, 'box': coords, 'center': get_center(coords), 'conf': float(box.conf[0])})

    # 2. 중복 제거
    parts = []
    raw_parts.sort(key=lambda x: x['conf'], reverse=True)
    for p in raw_parts:
        if not any(calculate_iou(p['box'], k['box']) > 0.1 for k in parts): parts.append(p)

    if parts and not any(p['name'] == 'source' for p in parts):
         leftmost = min(parts, key=lambda p: p['center'][0])
         leftmost['name'] = 'source'

    # 3. 위상 분석 (기하학적)
    connections = defaultdict(set)
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            p1, p2 = parts[i], parts[j]
            overlap = get_x_overlap_ratio(p1['box'], p2['box'])
            
            if overlap > 0.3: # 병렬
                node_a = i * 100 + j; node_b = i * 100 + j + 1
                connections[i].add(node_a); connections[i].add(node_b)
                connections[j].add(node_a); connections[j].add(node_b)
            elif abs(p1['center'][1] - p2['center'][1]) < 100 and abs(p1['center'][0] - p2['center'][0]) < 300: # 직렬
                node_common = i * 200 + j
                connections[i].add(node_common); connections[j].add(node_common)

    topology = extract_circuit_topology(parts, connections)
    
    # 시각화
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
    raw_objects = []

    # 1. 앙상블 탐지 & 이름 정규화
    for model in model_list:
        res = model.predict(source=img, conf=0.10, verbose=False)
        for b in res[0].boxes:
            raw_name = model.names[int(b.cls[0])]
            norm_name = normalize_name(raw_name) # 여기서 resistor_body -> resistor 로 통일됨
            coords = b.xyxy[0].tolist()
            conf = float(b.conf[0])
            
            # 필터링 (너무 낮은 신뢰도만 제거)
            if norm_name == 'capacitor' and conf < 0.15: continue
            if norm_name == 'resistor' and conf < 0.25: continue
            if 'breadboard' in raw_name or 'hole' in raw_name: continue
            
            raw_objects.append({'name': norm_name, 'box': coords, 'center': get_center(coords), 'conf': conf})

    # 2. 부품과 핀 분리
    parts_candidates = [p for p in raw_objects if p['name'] != 'leg']
    legs = [p for p in raw_objects if p['name'] == 'leg']

    # 3. 중복 제거
    parts = solve_overlap_real(parts_candidates)

    # 4. 전원(Source) 복구 로직 (와이어 이용)
    TOP_RAIL = h * 0.20; BOTTOM_RAIL = h * 0.80
    has_source = False
    
    # 이미 Source가 인식되었는지 확인
    if any(p['name'] == 'source' for p in parts): has_source = True
    
    # 인식 안됐으면 와이어/핀 위치로 추정
    if not has_source:
        for p in parts + legs:
            if p['center'][1] < TOP_RAIL or p['center'][1] > BOTTOM_RAIL:
                if p['name'] == 'wire' or p['name'] == 'leg':
                    has_source = True; break
    
    # Source 추가
    if has_source and not any(p['name'] == 'source' for p in parts):
        parts.append({'name': 'source', 'box': [0,0,0,0], 'center': (0,0), 'conf': 1.0})

    # 5. 노드 클러스터링 (세로줄)
    grouped_legs = []
    for leg in legs:
        assigned = False
        for group in grouped_legs:
            ref = group[0]
            if abs(leg['center'][0] - ref['center'][0]) < 25 and abs(leg['center'][1] - ref['center'][1]) < 80:
                group.append(leg); assigned = True; break
        if not assigned: grouped_legs.append([leg])

    # 6. 연결 매핑 (Incidence)
    connections = defaultdict(set)
    for i, part in enumerate(parts):
        if part['name'] == 'source':
            connections[i].add(-1)
            continue
            
        for nid, group in enumerate(grouped_legs):
            for leg in group:
                dist = math.sqrt((part['center'][0]-leg['center'][0])**2 + (part['center'][1]-leg['center'][1])**2)
                diag = math.sqrt((part['box'][2]-part['box'][0])**2 + (part['box'][3]-part['box'][1])**2)
                if dist < max(60, diag * 1.0):
                    connections[i].add(nid)
        
        # 가상 전원 레일 연결 확인
        if part['box'][2] > 0: # 가상부품 아님
             if part['center'][1] < TOP_RAIL or part['center'][1] > BOTTOM_RAIL:
                 connections[i].add(-1)

    topology = extract_circuit_topology(parts, connections)

    # 7. 시각화 (Clean View)
    summary = {'parts': parts, 'topology': topology}
    for p in parts:
        if p['name'] == 'wire': continue # 와이어는 계산엔 썼지만 화면엔 안 그림
        if p['box'][2] > 0: 
            color = (0, 255, 0)
            if p['name'] == 'source': color = (0, 255, 255)
            x1, y1, x2, y2 = map(int, p['box'])
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            # 깔끔하게 이름만
            label = p['name'][:3].upper()
            cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return img, summary

# ==========================================
# [Main UI]
# ==========================================
st.title("🧠 BrainBoard V60: Final Stabilized")
st.markdown("### ⚡ 회로 인식 + 이론 검증 (오류 수정 완료)")

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

    if st.button("🚀 정밀 분석 실행"):
        with st.spinner("데이터 정규화 및 회로 해석 중..."):
            
            res_ref_img, ref_data = analyze_schematic(ref_cv.copy(), model_sym)
            res_tgt_img, tgt_data = analyze_real(tgt_cv.copy(), models_real)

            # 1. 부품 개수 (BOM)
            st.subheader("1. 부품 구성 확인 (BOM)")
            ref_counts = defaultdict(int)
            tgt_counts = defaultdict(int)
            
            # 정규화된 이름으로 카운트
            for p in ref_data['parts']: ref_counts[p['name']] += 1
            for p in tgt_data['parts']: tgt_counts[p['name']] += 1
            
            all_keys = set(ref_counts.keys()) | set(tgt_counts.keys())
            bom_match = True
            
            # 테이블로 깔끔하게 비교
            bom_data = []
            for k in all_keys:
                if k == 'wire': continue
                r = ref_counts[k]; t = tgt_counts[k]
                status = "✅ 일치" if r == t else "❌ 불일치"
                bom_data.append({"부품명": k.upper(), "회로도": r, "실물": t, "상태": status})
                if r != t: bom_match = False
            
            st.table(bom_data)

            # 2. 토폴로지 비교
            st.subheader("2. 회로 이론 검증 (Theory Check)")
            
            # 키 생성 및 비교
            ref_topo = {item['key']: item['type'] for item in ref_data['topology']}
            tgt_topo = {item['key']: item['type'] for item in tgt_data['topology']}
            
            matches = []
            errors = []
            
            # 회로도 기준 검사
            for key, r_type in ref_topo.items():
                if key in tgt_topo:
                    t_type = tgt_topo[key]
                    if r_type == t_type:
                        matches.append(f"✅ {key} : {r_type} 연결 일치")
                    elif "Connected" in t_type and "Series" in r_type:
                        matches.append(f"✅ {key} : 전원부 연결 확인됨")
                    else:
                        errors.append(f"🚫 {key} : 회로도는 {r_type}이나, 실물은 {t_type}")
                else:
                    errors.append(f"❌ {key} : 실물에서 연결 끊김 (Open)")
            
            if not errors and len(matches) > 0 and bom_match:
                st.success("🎉 완벽합니다! 부품 구성과 회로 연결이 모두 일치합니다.")
                st.balloons()
            elif not matches and not errors:
                 st.info("ℹ️ 감지된 부품 간 연결이 없습니다. 부품이 서로 떨어져 있나요?")
            
            for m in matches: st.caption(m)
            for e in errors: st.error(e)

            st.image(cv2.cvtColor(res_ref_img, cv2.COLOR_BGR2RGB), caption="회로도 분석", use_column_width=True)
            st.image(cv2.cvtColor(res_tgt_img, cv2.COLOR_BGR2RGB), caption="실물 분석 (통합 버전)", use_column_width=True)
