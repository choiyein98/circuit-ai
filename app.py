import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import math
from PIL import Image
from collections import defaultdict
import gc

# ==========================================
# [설정] BrainBoard V64: The Final Integration
# ==========================================
st.set_page_config(page_title="BrainBoard V64: Final", layout="wide")

# [모델 경로] 가장 성능 좋은 모델 1개만 사용 (메모리 보호)
REAL_MODEL_PATH = 'best(3).pt' 
MODEL_SYM_PATH = 'symbol.pt'

# ==========================================
# [Helper Functions] 기본 도구들
# ==========================================
def resize_image_smart(image, max_size=1024):
    """메모리 폭발 방지를 위해 이미지 크기 자동 조절"""
    h, w = image.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image

def get_center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = ((box1[2]-box1[0])*(box1[3]-box1[1])) + ((box2[2]-box2[0])*(box2[3]-box2[1])) - inter
    return inter / union if union > 0 else 0

def get_x_overlap_ratio(box1, box2):
    """회로도에서 위아래로 겹쳤는지 확인 (병렬 판단용)"""
    x1_max = max(box1[0], box2[0]); x2_min = min(box1[2], box2[2])
    return max(0, x2_min - x1_max) / (box1[2] - box1[0])

def normalize_name(name):
    """복잡한 클래스 이름을 표준 이름으로 통일"""
    name = name.lower()
    if 'res' in name: return 'resistor'
    if 'cap' in name: return 'capacitor'
    if 'wire' in name: return 'wire'
    if any(x in name for x in ['source', 'batt', 'volt', 'vdc']): return 'source'
    if any(x in name for x in ['leg', 'pin', 'lead']): return 'leg'
    return name

def solve_overlap_real(parts):
    """중복된 박스 제거 (NMS)"""
    if not parts: return []
    parts.sort(key=lambda x: x.get('conf', 0), reverse=True)
    final = []
    for curr in parts:
        is_dup = False
        for k in final:
            iou = calculate_iou(curr['box'], k['box'])
            dist = math.sqrt((curr['center'][0]-k['center'][0])**2 + (curr['center'][1]-k['center'][1])**2)
            # 핀(leg)이 아니면 중복 제거 대상
            if curr['name'] != 'leg' and (iou > 0.4 or dist < 60): 
                is_dup = True; break
        if not is_dup: final.append(curr)
    return final

def get_rel_key(n1, n2):
    """관계 키 생성 (A-B와 B-A는 같음)"""
    return " <-> ".join(sorted([n1, n2]))

# ==========================================
# [Core Algorithm] 회로 위상(Topology) 추출기
# ==========================================
def extract_circuit_topology(parts, connections):
    """
    부품 간 공유하는 노드 개수를 세어서 직렬/병렬을 판단하는 핵심 뇌
    """
    topology = []
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            p1_name = parts[i]['name']
            p2_name = parts[j]['name']
            
            # 와이어는 연결 도구일 뿐, 부품 관계 비교에선 제외
            if p1_name == 'wire' or p2_name == 'wire': continue

            nodes_1 = connections[i]
            nodes_2 = connections[j]
            
            # 교집합 = 두 부품이 공유하는 노드들
            shared_nodes = nodes_1.intersection(nodes_2)
            num_shared = len(shared_nodes)
            
            rel_type = None
            if num_shared >= 2:
                rel_type = 'Parallel (병렬)' # 양쪽 다리 공유
            elif num_shared == 1:
                rel_type = 'Series (직렬)'   # 한쪽 다리 공유
                if -1 in shared_nodes: rel_type = 'Connected to Power (전원 연결)'
            
            if rel_type:
                topology.append({
                    'key': get_rel_key(p1_name, p2_name),
                    'type': rel_type,
                    'debug_nodes': shared_nodes
                })
    return topology

# ==========================================
# [분석 1] 회로도 (Schematic) 분석
# ==========================================
def analyze_schematic(img, model):
    img = resize_image_smart(img) # 리사이징
    results = model.predict(source=img, save=False, conf=0.05, verbose=False)
    
    # 1. 인식 및 정규화
    raw_parts = []
    for box in results[0].boxes:
        raw_name = model.names[int(box.cls[0])]
        norm_name = normalize_name(raw_name)
        coords = box.xyxy[0].tolist()
        raw_parts.append({'name': norm_name, 'box': coords, 'center': get_center(coords), 'conf': float(box.conf[0])})

    # 2. 중복 제거
    parts = []
    raw_parts.sort(key=lambda x: x['conf'], reverse=True)
    for p in raw_parts:
        if not any(calculate_iou(p['box'], k['box']) > 0.1 for k in parts): parts.append(p)

    # 3. 전원(Source)이 없으면 가장 왼쪽 부품을 전원으로 가정 (보정)
    if parts and not any(p['name'] == 'source' for p in parts):
         leftmost = min(parts, key=lambda p: p['center'][0])
         leftmost['name'] = 'source'

    # 4. 기하학적 연결 추론 (회로도는 그림 위치가 곧 연결)
    connections = defaultdict(set)
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            p1, p2 = parts[i], parts[j]
            overlap = get_x_overlap_ratio(p1['box'], p2['box'])
            
            # 위아래 겹침 -> 병렬 (가상 노드 2개 공유)
            if overlap > 0.3: 
                node_a = i * 100 + j; node_b = i * 100 + j + 1
                connections[i].add(node_a); connections[i].add(node_b)
                connections[j].add(node_a); connections[j].add(node_b)
            # 좌우 인접 -> 직렬 (가상 노드 1개 공유)
            elif abs(p1['center'][1] - p2['center'][1]) < 100 and abs(p1['center'][0] - p2['center'][0]) < 300:
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
# [분석 2] 실물 보드 (Real) 분석 - 노드 알고리즘 적용
# ==========================================
def analyze_real(img, model):
    img = resize_image_smart(img) # 리사이징
    h, w, _ = img.shape
    
    raw_objects = []
    res = model.predict(source=img, conf=0.10, verbose=False) # 인식률 10%까지 허용
    
    for b in res[0].boxes:
        raw_name = model.names[int(b.cls[0])]
        norm_name = normalize_name(raw_name)
        coords = b.xyxy[0].tolist()
        conf = float(b.conf[0])
        
        # 필터링 (너무 낮은 건 노이즈일 수 있으니 제거)
        if norm_name == 'capacitor' and conf < 0.15: continue
        if norm_name == 'resistor' and conf < 0.25: continue
        if 'breadboard' in raw_name or 'hole' in raw_name: continue
        
        raw_objects.append({'name': norm_name, 'box': coords, 'center': get_center(coords), 'conf': conf})

    # 1. 부품(Body)과 다리(Leg) 분리
    parts_candidates = [p for p in raw_objects if p['name'] != 'leg']
    legs = [p for p in raw_objects if p['name'] == 'leg']
    parts = solve_overlap_real(parts_candidates)

    # 2. 전원(Source) 유무 판단 (와이어 위치 기반 보정)
    TOP_RAIL = h * 0.20; BOTTOM_RAIL = h * 0.80
    has_source = False
    
    if any(p['name'] == 'source' for p in parts): has_source = True
    if not has_source:
        for p in parts + legs: # 와이어나 핀이 전원 레일에 있으면
            if p['center'][1] < TOP_RAIL or p['center'][1] > BOTTOM_RAIL:
                if p['name'] == 'wire' or p['name'] == 'leg':
                    has_source = True; break
    
    # Source가 인식 안 됐어도 있다고 가정 (비교를 위해)
    if has_source and not any(p['name'] == 'source' for p in parts):
        parts.append({'name': 'source', 'box': [0,0,0,0], 'center': (0,0), 'conf': 1.0})

    # 3. [핵심] 노드 클러스터링 (같은 세로줄 핀 찾기)
    grouped_legs = []
    for leg in legs:
        assigned = False
        for group in grouped_legs:
            ref = group[0]
            # X축 오차가 작고(같은 줄), Y축 오차가 적당하면(같은 블록)
            if abs(leg['center'][0] - ref['center'][0]) < 25 and abs(leg['center'][1] - ref['center'][1]) < 80:
                group.append(leg); assigned = True; break
        if not assigned: grouped_legs.append([leg])

    # 4. 부품과 노드 매핑 (Incidence Matrix 생성)
    connections = defaultdict(set)
    for i, part in enumerate(parts):
        # 전원은 글로벌 노드(-1)
        if part['name'] == 'source':
            connections[i].add(-1)
            continue
            
        # 부품 근처의 핀이 어느 노드(그룹)에 속했는지 확인
        for nid, group in enumerate(grouped_legs):
            for leg in group:
                dist = math.sqrt((part['center'][0]-leg['center'][0])**2 + (part['center'][1]-leg['center'][1])**2)
                diag = math.sqrt((part['box'][2]-part['box'][0])**2 + (part['box'][3]-part['box'][1])**2)
                # 부품 몸통 근처(대각선 길이 정도)에 핀이 있으면 연결된 것
                if dist < max(60, diag * 1.0):
                    connections[i].add(nid)
        
        # 전원 레일(VCC/GND) 직접 접촉 확인
        if part['box'][2] > 0: 
             if part['center'][1] < TOP_RAIL or part['center'][1] > BOTTOM_RAIL:
                 connections[i].add(-1)

    topology = extract_circuit_topology(parts, connections)

    # 5. 시각화
    summary = {'parts': parts, 'topology': topology}
    for p in parts:
        if p['name'] == 'wire': continue # 와이어는 계산엔 쓰지만 화면엔 안 그림 (깔끔하게)
        if p['box'][2] > 0: 
            color = (0, 255, 0)
            if p['name'] == 'source': color = (0, 255, 255)
            x1, y1, x2, y2 = map(int, p['box'])
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            label = p['name'][:3].upper()
            cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return img, summary

# ==========================================
# [Main UI] 스트림릿 인터페이스
# ==========================================
st.title("🧠 BrainBoard V64: Final System")
st.markdown("### ⚡ 부품 인식 + 노드(Node) 기반 회로 정밀 검증")

@st.cache_resource
def load_models():
    gc.collect() # 메모리 청소
    return YOLO(REAL_MODEL_PATH), YOLO(MODEL_SYM_PATH)

try:
    model_real, model_sym = load_models()
    st.sidebar.success("✅ 시스템 준비 완료")
except Exception as e:
    st.error(f"모델 로드 실패: {e}")
    st.stop()

col1, col2 = st.columns(2)
ref_file = col1.file_uploader("1. 회로도 (Schematic)", type=['jpg', 'png', 'jpeg'])
tgt_file = col2.file_uploader("2. 실물 사진 (Real Board)", type=['jpg', 'png', 'jpeg'])

if ref_file and tgt_file:
    ref_image = Image.open(ref_file)
    tgt_image = Image.open(tgt_file)
    ref_cv = cv2.cvtColor(np.array(ref_image), cv2.COLOR_RGB2BGR)
    tgt_cv = cv2.cvtColor(np.array(tgt_image), cv2.COLOR_RGB2BGR)

    if st.button("🚀 회로 분석 시작 (Analyze)"):
        gc.collect()
        with st.spinner("AI가 회로 노드(Node)를 추적 중입니다..."):
            
            res_ref_img, ref_data = analyze_schematic(ref_cv.copy(), model_sym)
            res_tgt_img, tgt_data = analyze_real(tgt_cv.copy(), model_real)

            # ------------------------------------------------
            # 1. 부품 구성 확인 (BOM Check)
            # ------------------------------------------------
            st.subheader("1. 부품 구성 확인 (BOM)")
            ref_counts = defaultdict(int)
            tgt_counts = defaultdict(int)
            for p in ref_data['parts']: ref_counts[p['name']] += 1
            for p in tgt_data['parts']: tgt_counts[p['name']] += 1
            
            all_keys = set(ref_counts.keys()) | set(tgt_counts.keys())
            bom_match = True
            bom_data = []
            
            for k in all_keys:
                if k == 'wire': continue # 와이어 개수는 무시 (연결 도구일 뿐)
                r = ref_counts[k]; t = tgt_counts[k]
                status = "✅ 일치" if r == t else "❌ 불일치"
                bom_data.append({"부품명": k.upper(), "회로도": r, "실물": t, "상태": status})
                if r != t: bom_match = False
            
            st.table(bom_data)

            # ------------------------------------------------
            # 2. 회로 위상 검증 (Topology Check)
            # ------------------------------------------------
            st.subheader("2. 회로 연결 검증 (Topology)")
            
            # 비교를 위해 딕셔너리로 변환
            ref_topo = {item['key']: item['type'] for item in ref_data['topology']}
            tgt_topo = {item['key']: item['type'] for item in tgt_data['topology']}
            
            matches = []
            errors = []
            
            for key, r_type in ref_topo.items():
                if key in tgt_topo:
                    t_type = tgt_topo[key]
                    # 'Parallel' 이나 'Series'가 일치하는지 확인
                    if r_type.split()[0] == t_type.split()[0]:
                        matches.append(f"✅ {key} : {t_type} - 정상 연결")
                    # 전원 연결은 Series의 일종으로 간주 (유연성)
                    elif "Power" in t_type and "Series" in r_type:
                        matches.append(f"✅ {key} : 전원부 연결 확인됨")
                    else:
                        errors.append(f"🚫 {key} : 회로도는 [{r_type}]인데 실물은 [{t_type}]입니다.")
                else:
                    errors.append(f"❌ {key} : 실물에서 연결이 끊겼습니다 (Open Circuit).")
            
            # 결과 출력
            if not errors and len(matches) > 0 and bom_match:
                st.success("🎉 완벽합니다! 부품 구성과 회로 연결이 정확합니다.")
                st.balloons()
            elif not matches and not errors:
                 st.info("ℹ️ 부품 간 연결 관계를 찾지 못했습니다. 부품이 너무 멀리 떨어져 있나요?")
            
            for m in matches: st.caption(m)
            for e in errors: st.error(e)

            st.image(cv2.cvtColor(res_ref_img, cv2.COLOR_BGR2RGB), caption="회로도 분석", use_column_width=True)
            st.image(cv2.cvtColor(res_tgt_img, cv2.COLOR_BGR2RGB), caption="실물 분석 (Node Logic Applied)", use_column_width=True)
            
            # 메모리 정리
            del res_ref_img, res_tgt_img
            gc.collect()
