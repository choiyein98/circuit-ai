import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import math
from PIL import Image
from collections import defaultdict

# ==========================================
# [설정] BrainBoard V59: Clean UI & OFF Detection
# ==========================================
st.set_page_config(page_title="BrainBoard V59", layout="wide")

REAL_MODEL_PATHS = ['best.pt', 'best(2).pt', 'best(3).pt']
MODEL_SYM_PATH = 'symbol.pt'
LEG_EXTENSION_RANGE = 180

# ==========================================
# [Class] 회로 연결 분석기 (텍스트 분석용)
# ==========================================
class CircuitAnalyzer:
    def __init__(self, components, distance_threshold=60):
        self.components = components
        self.threshold = distance_threshold
        self.nodes = [] 
        self.netlist = {} 

    def _get_legs(self, box):
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        if w > h: return [(x1, (y1+y2)/2), (x2, (y1+y2)/2)]
        else: return [((x1+x2)/2, y1), ((x1+x2)/2, y2)]

    def _find_node_id(self, leg_point):
        lx, ly = leg_point
        for node_id, points in enumerate(self.nodes):
            for px, py in points:
                if math.sqrt((lx-px)**2 + (ly-py)**2) < self.threshold:
                    self.nodes[node_id].append(leg_point)
                    return node_id
        new_id = len(self.nodes)
        self.nodes.append([leg_point])
        return new_id

    def build_graph(self):
        for i, comp in enumerate(self.components):
            comp_id = f"{comp['name']}_{i}"
            legs = self._get_legs(comp['box'])
            connected_nodes = set()
            for leg in legs:
                connected_nodes.add(self._find_node_id(leg))
            self.netlist[comp_id] = connected_nodes

    def get_connections(self):
        connections = []
        comp_ids = list(self.netlist.keys())
        for i in range(len(comp_ids)):
            for j in range(i + 1, len(comp_ids)):
                id_a = comp_ids[i]; id_b = comp_ids[j]
                nodes_a = self.netlist[id_a]; nodes_b = self.netlist[id_b]
                shared = len(nodes_a.intersection(nodes_b))
                
                name_a = id_a.split('_')[0]; name_b = id_b.split('_')[0]
                
                if shared == 2:
                    connections.append({'p1': name_a, 'p2': name_b, 'type': 'Parallel'})
                elif shared == 1:
                    connections.append({'p1': name_a, 'p2': name_b, 'type': 'Series'})
        return connections

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

def generate_relation_key(name1, name2):
    names = sorted([name1, name2])
    return f"{names[0]}-{names[1]}"

# ==========================================
# [중복 제거]
# ==========================================
def solve_overlap_schematic_v48(parts):
    if not parts: return []
    parts.sort(key=lambda x: x['conf'], reverse=True)
    final = []
    for curr in parts:
        is_dup = False
        for k in final:
            iou = calculate_iou(curr['box'], k['box'])
            dist = math.sqrt((curr['center'][0]-k['center'][0])**2 + (curr['center'][1]-k['center'][1])**2)
            if iou > 0.1 or dist < 80: is_dup = True; break
        if not is_dup: final.append(curr)
    return final

def solve_overlap_real_v35(parts):
    if not parts: return []
    parts.sort(key=lambda x: x.get('conf', 0), reverse=True)
    final = []
    for curr in parts:
        is_dup = False
        for k in final:
            iou = calculate_iou(curr['box'], k['box'])
            dist = math.sqrt((curr['center'][0]-k['center'][0])**2 + (curr['center'][1]-k['center'][1])**2)
            if iou > 0.4 or dist < 60: is_dup = True; break
        if not is_dup: final.append(curr)
    return final

# ==========================================
# [분석 1] 회로도 (V48 로직)
# ==========================================
def analyze_schematic(img, model):
    results = model.predict(source=img, save=False, conf=0.05, verbose=False)
    raw_parts = []
    for box in results[0].boxes:
        name = model.names[int(box.cls[0])].lower()
        conf = float(box.conf[0])
        coords = box.xyxy[0].tolist()
        base_name = name.split('_')[0].split(' ')[0]
        if base_name in ['vdc', 'vsource', 'battery', 'voltage', 'v']: base_name = 'source'
        if base_name in ['cap', 'c', 'capacitor']: base_name = 'capacitor'
        if base_name in ['res', 'r', 'resistor']: base_name = 'resistor'
        raw_parts.append({'name': base_name, 'box': coords, 'center': get_center(coords), 'conf': conf})

    parts = solve_overlap_schematic_v48(raw_parts)
    if parts and not any(p['name'] == 'source' for p in parts):
        leftmost = min(parts, key=lambda p: p['center'][0])
        leftmost['name'] = 'source'

    analyzer = CircuitAnalyzer(parts, distance_threshold=120) 
    analyzer.build_graph()
    connections = analyzer.get_connections()

    summary = {'parts': parts, 'connections': connections, 'counts': defaultdict(int)}
    for p in parts:
        summary['counts'][p['name']] += 1
        color = (255, 0, 0) if p['name'] == 'source' else (0, 0, 255)
        x1, y1, x2, y2 = map(int, p['box'])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, p['name'], (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
    return img, summary

# ==========================================
# [분석 2] 실물 (V35 ON/OFF 로직 + 앙상블)
# ==========================================
def analyze_real_ensemble(img, model_list):
    h, w, _ = img.shape
    raw_bodies = []
    raw_pins = [] 
    
    # 1. 앙상블 탐지
    for model in model_list:
        res = model.predict(source=img, conf=0.10, verbose=False)
        for box in res[0].boxes:
            name = model.names[int(box.cls[0])].lower()
            conf = float(box.conf[0])
            coords = box.xyxy[0].tolist()
            
            if 'cap' in name and conf < 0.15: continue
            elif 'res' in name and conf < 0.60: continue
            elif 'wire' in name and conf < 0.15: continue
            elif conf < 0.25: continue

            if any(x in name for x in ['pin', 'leg', 'lead']) and 'wire' not in name:
                raw_pins.append({'center': get_center(coords), 'box': coords})
            elif 'breadboard' not in name:
                raw_bodies.append({'name': name, 'box': coords, 'center': get_center(coords), 'conf': conf, 'is_on': False}) # 초기값 OFF

    parts = solve_overlap_real_v35(raw_bodies)
    
    # 2. [ON/OFF 판별] V35의 "전원 전파(Propagation)" 로직 복원
    # 전원 레일(상/하단 45%)에 닿은 핀이나 와이어 찾기
    power_active = False
    for p in raw_pins:
        if p['center'][1] < h * 0.45 or p['center'][1] > h * 0.55: # 브레드보드 전원부 대략적 위치
            power_active = True; break
            
    if not power_active:
        # 핀이 없으면 와이어라도 찾음
        for b in parts:
            if 'wire' in b['name'] and (b['center'][1] < h * 0.45 or b['center'][1] > h * 0.55):
                power_active = True; break

    if power_active:
        # 1차: 전원부에 직접 닿은 부품 켜기
        for comp in parts:
            cy = comp['center'][1]
            if cy < h*0.48 or cy > h*0.52: 
                comp['is_on'] = True

        # 2차: 전원 연결된 부품과 가까운 부품 전파 (3회 반복)
        for _ in range(3): 
            for comp in parts:
                if comp['is_on']: continue 
                cx, cy = comp['center']
                
                # 핀을 통해 연결 확인
                for p in raw_pins:
                    px, py = p['center']
                    # 핀이 전원부 영역에 있거나
                    if py < h*0.48 or py > h*0.52:
                         dist = math.sqrt((cx - px)**2 + (cy - py)**2)
                         if dist < LEG_EXTENSION_RANGE:
                             comp['is_on'] = True; break
                if comp['is_on']: continue

                # 이미 켜진 다른 부품과 가까우면 연결 (직렬 연결 가정)
                for other in parts:
                    if not other['is_on']: continue
                    ocx, ocy = other['center']
                    dist = math.sqrt((cx - ocx)**2 + (cy - ocy)**2)
                    if dist < LEG_EXTENSION_RANGE * 1.5:
                        comp['is_on'] = True; break

    # 3. 넷리스트 분석 (텍스트용)
    analyzer = CircuitAnalyzer(parts, distance_threshold=60)
    analyzer.build_graph()
    connections = analyzer.get_connections()

    summary = {'parts': parts, 'connections': connections, 'counts': defaultdict(int)}

    # 4. 시각화 (선 없음! 오직 박스 색상으로만 표시)
    for p in parts:
        norm_name = p['name']
        if 'res' in norm_name: norm_name = 'resistor'
        elif 'cap' in norm_name: norm_name = 'capacitor'
        if 'wire' not in norm_name: summary['counts'][norm_name] += 1
        
        is_on = p['is_on']
        
        # [핵심] ON=초록, OFF=빨강
        if is_on:
            color = (0, 255, 0) # Green
            label = f"{norm_name[:3].upper()}: ON"
        else:
            color = (0, 0, 255) # Red (BGR)
            label = f"{norm_name[:3].upper()}: OFF"

        x1, y1, x2, y2 = map(int, p['box'])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        
        # 라벨 배경 박스
        (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - 25), (x1 + w_text, y1), color, -1)
        cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return img, summary

# ==========================================
# [Main UI]
# ==========================================
st.title("🧠 BrainBoard V59: Visual Fix")

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
    st.sidebar.success(f"✅ 모델 로드 완료 ({len(models_real)} Ens)")
except: st.stop()

col1, col2 = st.columns(2)
ref_file = col1.file_uploader("1. 회로도", type=['jpg', 'png', 'jpeg'])
tgt_file = col2.file_uploader("2. 실물 사진", type=['jpg', 'png', 'jpeg'])

if ref_file and tgt_file:
    ref_image = Image.open(ref_file)
    tgt_image = Image.open(tgt_file)
    ref_cv = cv2.cvtColor(np.array(ref_image), cv2.COLOR_RGB2BGR)
    tgt_cv = cv2.cvtColor(np.array(tgt_image), cv2.COLOR_RGB2BGR)

    if st.button("🚀 분석 실행"):
        with st.spinner("회로 구조 분석 및 비교 중..."):
            
            res_ref_img, ref_data = analyze_schematic(ref_cv.copy(), model_sym)
            res_tgt_img, tgt_data = analyze_real_ensemble(tgt_cv.copy(), models_real)

            # ---------------------------------------------
            # [1] 부품 개수 비교 (BOM)
            # ---------------------------------------------
            st.subheader("1. 부품 구성 확인")
            all_parts = set(ref_data['counts'].keys()) | set(tgt_data['counts'].keys())
            
            for k in all_parts:
                if k in ['wire', 'breadboard', 'text', 'hole', 'source']: continue 
                r = ref_data['counts'][k]
                t = tgt_data['counts'][k]
                if r != t:
                    st.error(f"⚠️ {k.capitalize()} 개수 불일치! (회로도 {r}개 vs 실물 {t}개)")
                else:
                    st.success(f"✅ {k.capitalize()} 개수 일치 ({r}개)")

            # ---------------------------------------------
            # [2] 연결 오류 (텍스트로만 표시)
            # ---------------------------------------------
            st.subheader("2. 연결 오류 리포트")
            
            ref_rels = {generate_relation_key(c['p1'], c['p2']): c['type'] for c in ref_data['connections']}
            tgt_rels = {generate_relation_key(c['p1'], c['p2']): c['type'] for c in tgt_data['connections']}
            
            error_found = False
            
            # 회로도에 있는데 실물에 없는 경우
            for key, ref_type in ref_rels.items():
                if key not in tgt_rels:
                    p1, p2 = key.split('-')
                    st.error(f"❌ [연결 끊김] '{p1}'와(과) '{p2}'가 연결되지 않았습니다. (회로도: {ref_type})")
                    error_found = True
            
            # 실물에만 있는 경우 (잘못된 연결)
            for c in tgt_data['connections']:
                key = generate_relation_key(c['p1'], c['p2'])
                if key not in ref_rels:
                    st.error(f"❓ [잘못된 연결] '{c['p1']}'와(과) '{c['p2']}'가 엉뚱하게 연결되었습니다.")
                    error_found = True

            # 타입 불일치
            for key, ref_type in ref_rels.items():
                if key in tgt_rels:
                    if ref_type != tgt_rels[key]:
                        st.warning(f"⚠️ [연결 방식 다름] '{key}': 회로도는 {ref_type}인데 실물은 {tgt_rels[key]}입니다.")
                        error_found = True

            if not error_found:
                st.info("✨ 연결 관계에 특별한 문제가 발견되지 않았습니다.")

            # ---------------------------------------------
            # [3] 결과 이미지 출력 (회로도 포함!)
            # ---------------------------------------------
            st.divider()
            st.image(cv2.cvtColor(res_ref_img, cv2.COLOR_BGR2RGB), caption="[1] 회로도 분석 결과", use_column_width=True)
            st.image(cv2.cvtColor(res_tgt_img, cv2.COLOR_BGR2RGB), caption="[2] 실물 분석 결과 (빨간박스: 연결안됨 / 초록박스: 정상)", use_column_width=True)
