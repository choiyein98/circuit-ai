import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import math
from PIL import Image
from collections import defaultdict

# ==========================================
# [설정] BrainBoard V56: Netlist & Role Check
# ==========================================
st.set_page_config(page_title="BrainBoard V56: Netlist", layout="wide")

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

def solve_overlap(parts, is_real=False):
    if not parts: return []
    parts.sort(key=lambda x: x.get('conf', 0), reverse=True)
    final = []
    for curr in parts:
        is_dup = False
        for k in final:
            iou = calculate_iou(curr['box'], k['box'])
            dist = math.sqrt((curr['center'][0]-k['center'][0])**2 + (curr['center'][1]-k['center'][1])**2)
            if is_real:
                if iou > 0.4 or dist < 60: is_dup = True; break
            else:
                if iou > 0.1: is_dup = True; break
        if not is_dup: final.append(curr)
    return final

# [NEW] 관계(Netlist) 텍스트 생성기
def generate_relation_key(name1, name2):
    # 이름을 알파벳 순으로 정렬해서 "Res-Cap"과 "Cap-Res"를 동일하게 취급
    names = sorted([name1, name2])
    return f"{names[0]} <-> {names[1]}"

# ==========================================
# [알고리즘 1] 회로도 넷리스트 추출
# ==========================================
def analyze_schematic_netlist(img, model):
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

    parts = solve_overlap(raw_parts, is_real=False)

    # 전원 보정
    if parts and not any('source' in p['name'] for p in parts):
         leftmost = min(parts, key=lambda p: p['center'][0])
         leftmost['name'] = 'source'

    connections = [] # [(부품1, 부품2, 관계유형)]
    
    # 기하학적 위치로 관계 추론
    # 1. 병렬 (위아래 겹침)
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            p1, p2 = parts[i], parts[j]
            overlap = get_x_overlap_ratio(p1['box'], p2['box'])
            
            if overlap > 0.3: # 위아래로 겹침
                connections.append({'p1': p1['name'], 'p2': p2['name'], 'type': 'Parallel'})
                # 시각화
                cv2.rectangle(img, (int(p1['box'][0]), int(p1['box'][1])), (int(p2['box'][2]), int(p2['box'][3])), (255, 0, 255), 2)
            
            # 2. 직렬 (바로 옆에 있음, Y축 비슷)
            elif abs(p1['center'][1] - p2['center'][1]) < 100:
                dist = abs(p1['center'][0] - p2['center'][0])
                if dist < 300: # 적당히 가까움
                    connections.append({'p1': p1['name'], 'p2': p2['name'], 'type': 'Series'})

    summary = {'parts': parts, 'connections': connections}
    
    # 부품 그리기
    for p in parts:
        x1, y1, x2, y2 = map(int, p['box'])
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img, p['name'], (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    return img, summary

# ==========================================
# [알고리즘 2] 실물 넷리스트 추출 (노드 기반)
# ==========================================
def analyze_real_netlist(img, model_list):
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
            
            # 필터링
            if 'cap' in name and conf < 0.15: continue
            if 'res' in name and conf < 0.30: continue # V49 기준 적용
            if 'breadboard' in name or 'hole' in name: continue
            
            center = get_center(coords)
            
            if any(x in name for x in ['pin', 'leg', 'lead']):
                raw_legs.append({'box': coords, 'center': center})
            elif 'wire' not in name: 
                raw_bodies.append({'name': name, 'box': coords, 'center': center, 'conf': conf})

    parts = solve_overlap(raw_bodies, is_real=True)

    # 2. 노드(Node) 클러스터링 (세로줄 그룹화)
    grouped_legs = []
    for leg in raw_legs:
        assigned = False
        for group in grouped_legs:
            ref = group[0] 
            # X축이 비슷하고(25px), Y축도 적당히(80px)
            if abs(leg['center'][0] - ref['center'][0]) < 25 and abs(leg['center'][1] - ref['center'][1]) < 80:
                group.append(leg); assigned = True; break
        if not assigned: grouped_legs.append([leg])

    # 3. 부품-노드 연결 매핑
    part_connections = defaultdict(set)
    for i, part in enumerate(parts):
        for nid, group in enumerate(grouped_legs):
            for leg in group:
                dist = math.sqrt((part['center'][0]-leg['center'][0])**2 + (part['center'][1]-leg['center'][1])**2)
                diag = math.sqrt((part['box'][2]-part['box'][0])**2 + (part['box'][3]-part['box'][1])**2)
                if dist < diag * 0.9: # 부품 근처에 있는 핀
                    part_connections[i].add(nid)

    # 4. 부품 간 관계(Netlist) 도출
    connections = []
    
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            nodes_i = part_connections[i]
            nodes_j = part_connections[j]
            shared_nodes = nodes_i.intersection(nodes_j)
            
            p1_name = parts[i]['name'].split('_')[0] # res_1 -> res
            p2_name = parts[j]['name'].split('_')[0]

            if len(shared_nodes) >= 2: # 노드 2개 공유 = 병렬
                connections.append({'p1': p1_name, 'p2': p2_name, 'type': 'Parallel'})
                # 병렬 시각화 (보라색 선)
                cv2.line(img, (int(parts[i]['center'][0]), int(parts[i]['center'][1])),
                         (int(parts[j]['center'][0]), int(parts[j]['center'][1])), (255, 0, 255), 3)
            
            elif len(shared_nodes) == 1: # 노드 1개 공유 = 직렬
                connections.append({'p1': p1_name, 'p2': p2_name, 'type': 'Series'})
                # 직렬 시각화 (청록색 선)
                cv2.line(img, (int(parts[i]['center'][0]), int(parts[i]['center'][1])),
                         (int(parts[j]['center'][0]), int(parts[j]['center'][1])), (255, 255, 0), 2)

    # 부품 그리기 & 이름 정규화
    summary = {'parts': parts, 'connections': connections}
    for p in parts:
        norm_name = p['name']
        if 'res' in norm_name: norm_name = 'resistor'
        elif 'cap' in norm_name: norm_name = 'capacitor'
        p['name'] = norm_name # 이름 업데이트

        color = (0, 255, 0) # 기본 녹색
        if 'source' in norm_name: color = (0, 255, 255) # 전원은 노란색

        x1, y1, x2, y2 = map(int, p['box'])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.putText(img, norm_name[:3].upper(), (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return img, summary

# ==========================================
# [Main UI]
# ==========================================
st.title("🧠 BrainBoard V56: Netlist Validator")
st.markdown("### 🔍 부품의 역할과 연결 관계(Netlist) 정밀 검증")

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

    if st.button("🚀 Netlist 분석 실행"):
        with st.spinner("회로 넷리스트 추출 및 비교 중..."):
            
            res_ref_img, ref_data = analyze_schematic_netlist(ref_cv.copy(), model_sym)
            res_tgt_img, tgt_data = analyze_real_netlist(tgt_cv.copy(), models_real)

            # ------------------------------------------------
            # 1. 부품 목록 비교 (Bill of Materials)
            # ------------------------------------------------
            st.subheader("1. 부품 목록 (BOM)")
            ref_counts = defaultdict(int)
            tgt_counts = defaultdict(int)
            for p in ref_data['parts']: ref_counts[p['name']] += 1
            for p in tgt_data['parts']: tgt_counts[p['name']] += 1
            
            all_keys = set(ref_counts.keys()) | set(tgt_counts.keys())
            bom_match = True
            for k in all_keys:
                if k == 'wire': continue
                r = ref_counts[k]; t = tgt_counts[k]
                if r != t:
                    st.error(f"⚠️ {k} 개수 불일치 ({r} vs {t})")
                    bom_match = False
                else:
                    st.success(f"✅ {k} 개수 일치 ({r})")

            # ------------------------------------------------
            # 2. 연결 관계(Netlist) 비교 (핵심!)
            # ------------------------------------------------
            st.subheader("2. 연결 관계 및 역할 검증 (Netlist Check)")
            
            # 회로도 관계 리스트 만들기
            ref_relations = set()
            for c in ref_data['connections']:
                key = generate_relation_key(c['p1'], c['p2'])
                ref_relations.add((key, c['type']))
            
            # 실물 관계 리스트 만들기
            tgt_relations = set()
            for c in tgt_data['connections']:
                key = generate_relation_key(c['p1'], c['p2'])
                tgt_relations.add((key, c['type']))

            # 비교 로직
            matches = []
            missings = []
            
            # 회로도에 있는게 실물에 있는가?
            for rel in ref_relations:
                key, type_ = rel
                # 실물에서 키가 같은게 있는지 확인 (타입은 다를 수도 있으니 키로 먼저 검색)
                found = False
                for t_rel in tgt_relations:
                    if t_rel[0] == key:
                        found = True
                        if t_rel[1] == type_:
                            matches.append(f"✅ [일치] {key} : {type_} 연결됨")
                        else:
                            missings.append(f"⚠️ [오류] {key} : 회로도는 {type_}인데 실물은 {t_rel[1]}임")
                        break
                if not found:
                    missings.append(f"❌ [끊김] {key} : 실물에서 연결되지 않음")

            if not missings and len(matches) > 0:
                st.success("🎉 모든 부품의 연결 관계와 역할이 완벽하게 일치합니다!")
                st.balloons()
            elif not matches and not missings:
                 st.info("ℹ️ 감지된 연결 관계가 없습니다. 부품이 너무 멀리 떨어져 있나요?")
            
            for m in matches: st.caption(m)
            for m in missings: st.error(m)

            # 이미지 출력
            st.image(cv2.cvtColor(res_ref_img, cv2.COLOR_BGR2RGB), caption="회로도 Netlist", use_column_width=True)
            st.image(cv2.cvtColor(res_tgt_img, cv2.COLOR_BGR2RGB), caption="실물 Netlist (보라색=병렬, 청록색=직렬)", use_column_width=True)
