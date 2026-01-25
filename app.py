import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import math
from PIL import Image
from collections import defaultdict

# ==========================================
# [설정] BrainBoard V58: Error Visualizer
# ==========================================
st.set_page_config(page_title="BrainBoard V58", layout="wide")

REAL_MODEL_PATHS = ['best.pt', 'best(2).pt', 'best(3).pt']
MODEL_SYM_PATH = 'symbol.pt'
LEG_EXTENSION_RANGE = 180

# ==========================================
# [Class] 회로 연결 분석기
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
                
                # 좌표 정보도 함께 저장 (시각화용)
                # components 리스트에서 인덱스(i, j)로 좌표 찾기
                pos_a = self.components[i]['center']
                pos_b = self.components[j]['center']

                if shared == 2:
                    connections.append({'p1': name_a, 'p2': name_b, 'type': 'Parallel', 'pos1': pos_a, 'pos2': pos_b})
                elif shared == 1:
                    connections.append({'p1': name_a, 'p2': name_b, 'type': 'Series', 'pos1': pos_a, 'pos2': pos_b})
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
    return f"{names[0]}-{names[1]}" # e.g., "capacitor-resistor"

# 시각화용: 두 점 사이 점선 그리기
def draw_dotted_line(img, pt1, pt2, color, thickness=2, gap=10):
    dist = np.linalg.norm(np.array(pt1) - np.array(pt2))
    pts = []
    for i in np.arange(0, dist, gap):
        r = i / dist
        x = int((pt1[0] * (1 - r) + pt2[0] * r))
        y = int((pt1[1] * (1 - r) + pt2[1] * r))
        pts.append((x, y))
    
    for i in range(0, len(pts) - 1, 2):
        cv2.line(img, pts[i], pts[i+1], color, thickness)

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
# [분석 1] 회로도 (V48)
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
        min(parts, key=lambda p: p['center'][0])['name'] = 'source'

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
# [분석 2] 실물 (V35 앙상블)
# ==========================================
def analyze_real_ensemble(img, model_list):
    raw_bodies = []
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

            if 'breadboard' not in name and 'hole' not in name and 'pin' not in name and 'leg' not in name:
                raw_bodies.append({'name': name, 'box': coords, 'center': get_center(coords), 'conf': conf})

    parts = solve_overlap_real_v35(raw_bodies)
    
    analyzer = CircuitAnalyzer(parts, distance_threshold=60)
    analyzer.build_graph()
    connections = analyzer.get_connections()

    summary = {'parts': parts, 'connections': connections, 'counts': defaultdict(int)}
    for p in parts:
        norm_name = p['name']
        if 'res' in norm_name: norm_name = 'resistor'
        elif 'cap' in norm_name: norm_name = 'capacitor'
        if 'wire' not in norm_name: summary['counts'][norm_name] += 1
        
        # 기본 박스 그리기
        x1, y1, x2, y2 = map(int, p['box'])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, norm_name, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return img, summary

# ==========================================
# [Main UI]
# ==========================================
st.title("🧠 BrainBoard V58: Error Visualizer")

@st.cache_resource
def load_models():
    reals = []; sym = None
    try:
        for p in REAL_MODEL_PATHS: reals.append(YOLO(p))
        sym = YOLO(MODEL_SYM_PATH)
    except: pass
    return reals, sym

try:
    models_real, model_sym = load_models()
    if not models_real: st.stop()
    st.sidebar.success(f"✅ Ready ({len(models_real)} Ens)")
except: st.stop()

col1, col2 = st.columns(2)
ref_file = col1.file_uploader("1. 회로도", type=['jpg', 'png', 'jpeg'])
tgt_file = col2.file_uploader("2. 실물 사진", type=['jpg', 'png', 'jpeg'])

if ref_file and tgt_file:
    ref_image = Image.open(ref_file)
    tgt_image = Image.open(tgt_file)
    ref_cv = cv2.cvtColor(np.array(ref_image), cv2.COLOR_RGB2BGR)
    tgt_cv = cv2.cvtColor(np.array(tgt_image), cv2.COLOR_RGB2BGR)

    if st.button("🚀 분석 및 시각화"):
        with st.spinner("회로 불일치 추적 중..."):
            res_ref_img, ref_data = analyze_schematic(ref_cv.copy(), model_sym)
            res_tgt_img, tgt_data = analyze_real_ensemble(tgt_cv.copy(), models_real)

            # ---------------------------------------------
            # [비교 로직 및 시각화]
            # ---------------------------------------------
            ref_rels = {generate_relation_key(c['p1'], c['p2']): c['type'] for c in ref_data['connections']}
            tgt_rels = {generate_relation_key(c['p1'], c['p2']): c['type'] for c in tgt_data['connections']}
            
            error_messages = []
            
            # 1. 회로도에 있는데 실물에 없는 경우 (Missing)
            for key, ref_type in ref_rels.items():
                if key not in tgt_rels:
                    p1_name, p2_name = key.split('-')
                    error_messages.append(f"❌ [불일치] '{p1_name}'와(과) '{p2_name}' 사이의 연결이 끊어져 있습니다 (회로도: {ref_type})")
                    
                    # 시각화: 실물 이미지에 '있어야 할 곳'을 파란 점선으로 표시
                    # (해당 부품들 중 가장 가까운 쌍을 찾아 연결)
                    cands_p1 = [p for p in tgt_data['parts'] if p1_name in p['name']]
                    cands_p2 = [p for p in tgt_data['parts'] if p2_name in p['name']]
                    
                    if cands_p1 and cands_p2:
                        # 가장 가까운 쌍 찾기
                        min_dist = float('inf'); best_pair = None
                        for cp1 in cands_p1:
                            for cp2 in cands_p2:
                                d = math.sqrt((cp1['center'][0]-cp2['center'][0])**2 + (cp1['center'][1]-cp2['center'][1])**2)
                                if d < min_dist: min_dist = d; best_pair = (cp1, cp2)
                        
                        if best_pair:
                            pt1 = (int(best_pair[0]['center'][0]), int(best_pair[0]['center'][1]))
                            pt2 = (int(best_pair[1]['center'][0]), int(best_pair[1]['center'][1]))
                            draw_dotted_line(res_tgt_img, pt1, pt2, (255, 0, 0), thickness=3) # 파란 점선

            # 2. 실물에만 있는 엉뚱한 연결 (Wrong)
            for c in tgt_data['connections']:
                key = generate_relation_key(c['p1'], c['p2'])
                if key not in ref_rels:
                    error_messages.append(f"❓ [불일치] '{c['p1']}'와(과) '{c['p2']}'가 잘못 연결되었습니다 (회로도에 없음)")
                    
                    # 시각화: 실물 이미지에 잘못된 연결을 빨간 실선으로 표시
                    pt1 = (int(c['pos1'][0]), int(c['pos1'][1]))
                    pt2 = (int(c['pos2'][0]), int(c['pos2'][1]))
                    cv2.line(res_tgt_img, pt1, pt2, (0, 0, 255), 4) # 빨간 실선

            # 3. 연결은 있는데 타입이 다른 경우 (Diff Type)
            for key, ref_type in ref_rels.items():
                if key in tgt_rels:
                    if ref_type != tgt_rels[key]:
                        error_messages.append(f"⚠️ [불일치] '{key}' 연결이 다릅니다 (회로도: {ref_type}, 실물: {tgt_rels[key]})")

            # 결과 출력
            st.divider()
            if not error_messages:
                st.success("🎉 회로 연결이 완벽하게 일치합니다!")
            else:
                st.subheader("🚨 불일치 분석 결과")
                for msg in error_messages:
                    st.error(msg)
            
            st.image(cv2.cvtColor(res_tgt_img, cv2.COLOR_BGR2RGB), caption="실물 분석 결과 (🔴빨간선: 잘못됨 / 🔵파란점선: 누락됨)", use_column_width=True)
