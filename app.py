import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import math
from PIL import Image
from collections import defaultdict

# ==========================================
# [설정] BrainBoard V57: Netlist Validator
# ==========================================
st.set_page_config(page_title="BrainBoard V57", layout="wide")

# 실물 모델 3개 (앙상블)
REAL_MODEL_PATHS = ['best.pt', 'best(2).pt', 'best(3).pt']
MODEL_SYM_PATH = 'symbol.pt'
LEG_EXTENSION_RANGE = 180

# ==========================================
# [Class] 회로 연결 분석기 (Graph Builder)
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
        if w > h: # 가로형
            return [(x1, (y1+y2)/2), (x2, (y1+y2)/2)]
        else: # 세로형
            return [((x1+x2)/2, y1), ((x1+x2)/2, y2)]

    def _find_node_id(self, leg_point):
        lx, ly = leg_point
        for node_id, points in enumerate(self.nodes):
            for px, py in points:
                # 같은 노드로 묶는 거리 기준
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
        """
        비교를 위해 표준화된 연결 리스트 반환
        Format: [{'p1': 'resistor', 'p2': 'capacitor', 'type': 'Series'}, ...]
        """
        connections = []
        comp_ids = list(self.netlist.keys())
        
        for i in range(len(comp_ids)):
            for j in range(i + 1, len(comp_ids)):
                id_a = comp_ids[i]
                id_b = comp_ids[j]
                
                nodes_a = self.netlist[id_a]
                nodes_b = self.netlist[id_b]
                
                shared = len(nodes_a.intersection(nodes_b))
                
                name_a = id_a.split('_')[0]
                name_b = id_b.split('_')[0]
                
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

def generate_relation_key(c):
    # 비교를 위해 "Resistor-Capacitor"와 "Capacitor-Resistor"를 같게 처리
    names = sorted([c['p1'], c['p2']])
    return f"{names[0]} - {names[1]}"

# ==========================================
# [중복 제거] V48(회로도) & V35(실물)
# ==========================================
def solve_overlap_schematic_v48(parts):
    if not parts: return []
    parts.sort(key=lambda x: x['conf'], reverse=True) # 점수순
    final = []
    for curr in parts:
        is_dup = False
        for k in final:
            iou = calculate_iou(curr['box'], k['box'])
            dist = math.sqrt((curr['center'][0]-k['center'][0])**2 + (curr['center'][1]-k['center'][1])**2)
            if iou > 0.1 or dist < 80: # V48: 관대함
                is_dup = True; break
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
            # V35: 정교함 (IoU 0.4, Dist 60)
            if iou > 0.4: is_dup = True; break
            if dist < 60: is_dup = True; break
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

    # 연결 분석 실행 (회로도는 선이 길어서 threshold를 크게 잡음)
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
# [분석 2] 실물 (V35 + 앙상블)
# ==========================================
def analyze_real_ensemble(img, model_list):
    raw_bodies = []
    raw_pins = [] 
    
    for model in model_list:
        res = model.predict(source=img, conf=0.10, verbose=False)
        for box in res[0].boxes:
            name = model.names[int(box.cls[0])].lower()
            conf = float(box.conf[0])
            coords = box.xyxy[0].tolist()
            
            # V35 감도
            if 'cap' in name: min_c = 0.15
            elif 'res' in name: min_c = 0.60
            elif 'wire' in name: min_c = 0.15
            else: min_c = 0.25
            if conf < min_c: continue

            if any(x in name for x in ['pin', 'leg', 'lead']) and 'wire' not in name:
                raw_pins.append({'center': get_center(coords), 'box': coords})
            elif 'breadboard' not in name:
                raw_bodies.append({'name': name, 'box': coords, 'center': get_center(coords), 'conf': conf})

    parts = solve_overlap_real_v35(raw_bodies)
    
    # 연결 분석 실행 (실물은 핀/와이어 고려 필요하므로, 부품 자체 좌표로 근사 계산)
    # 정교한 분석을 위해 부품+핀 정보를 모두 활용하면 좋지만, 
    # 여기서는 부품 간 거리(터미널 스트립)를 기반으로 V54 로직 적용
    analyzer = CircuitAnalyzer(parts, distance_threshold=60)
    analyzer.build_graph()
    connections = analyzer.get_connections()

    summary = {'parts': parts, 'connections': connections, 'counts': defaultdict(int)}

    for p in parts:
        norm_name = p['name']
        if 'res' in norm_name: norm_name = 'resistor'
        elif 'cap' in norm_name: norm_name = 'capacitor'
        
        if 'wire' not in norm_name:
            summary['counts'][norm_name] += 1
            
        color = (0, 255, 0)
        x1, y1, x2, y2 = map(int, p['box'])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.putText(img, norm_name, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return img, summary

# ==========================================
# [Main UI]
# ==========================================
st.title("🧠 BrainBoard V57: Netlist Validator")

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
    st.sidebar.success(f"✅ 시스템 준비 완료 ({len(models_real)} Ens)")
except: st.stop()

col1, col2 = st.columns(2)
ref_file = col1.file_uploader("1. 회로도", type=['jpg', 'png', 'jpeg'])
tgt_file = col2.file_uploader("2. 실물 사진", type=['jpg', 'png', 'jpeg'])

if ref_file and tgt_file:
    ref_image = Image.open(ref_file)
    tgt_image = Image.open(tgt_file)
    ref_cv = cv2.cvtColor(np.array(ref_image), cv2.COLOR_RGB2BGR)
    tgt_cv = cv2.cvtColor(np.array(tgt_image), cv2.COLOR_RGB2BGR)

    if st.button("🚀 정밀 분석 및 비교"):
        with st.spinner("회로 구조를 비교 분석 중입니다..."):
            
            res_ref_img, ref_data = analyze_schematic(ref_cv.copy(), model_sym)
            res_tgt_img, tgt_data = analyze_real_ensemble(tgt_cv.copy(), models_real)

            # ----------------------------------------
            # [1] 부품 개수 비교 (BOM)
            # ----------------------------------------
            st.subheader("1. 부품 개수 비교")
            all_parts = set(ref_data['counts'].keys()) | set(tgt_data['counts'].keys())
            bom_ok = True
            
            for k in all_parts:
                if k in ['wire', 'breadboard', 'text', 'hole', 'source']: continue # source 제외
                r = ref_data['counts'][k]
                t = tgt_data['counts'][k]
                
                if r != t:
                    st.error(f"⚠️ {k.capitalize()} 개수 불일치! (회로도 {r}개 vs 실물 {t}개)")
                    bom_ok = False
                else:
                    st.success(f"✅ {k.capitalize()} 개수 일치 ({r}개)")

            # ----------------------------------------
            # [2] 연결 관계 비교 (Netlist)
            # ----------------------------------------
            st.subheader("2. 연결 관계 검증 (오류 지적)")
            
            # 비교를 위해 (키, 타입) 형태로 변환
            ref_rels = {}
            for c in ref_data['connections']:
                key = generate_relation_key(c)
                ref_rels[key] = c['type']
                
            tgt_rels = {}
            for c in tgt_data['connections']:
                key = generate_relation_key(c)
                tgt_rels[key] = c['type']
            
            # 매칭 로직
            match_list = []
            error_list = []
            
            # 회로도에 있는 연결이 실물에 있는가?
            for key, ref_type in ref_rels.items():
                if key in tgt_rels:
                    tgt_type = tgt_rels[key]
                    if ref_type == tgt_type:
                        match_list.append(f"✅ [일치] {key} : {ref_type} 연결 확인")
                    else:
                        error_list.append(f"⚠️ [타입 불일치] {key} : 회로도는 {ref_type}인데, 실물은 {tgt_type}입니다.")
                else:
                    error_list.append(f"❌ [연결 누락] {key} : 회로도엔 있는데 실물에서 끊어져 있습니다.")
            
            # 실물에만 있는 엉뚱한 연결?
            for key in tgt_rels:
                if key not in ref_rels:
                    error_list.append(f"❓ [미확인 연결] {key} : 회로도에 없는 연결이 실물에서 발견됨 (쇼트 의심)")

            if not error_list and len(match_list) > 0:
                st.balloons()
                st.success("🎉 회로 연결이 완벽하게 일치합니다!")
            elif not match_list and not error_list:
                st.info("ℹ️ 감지된 연결 관계가 없습니다.")
            
            for e in error_list: st.error(e)
            with st.expander("일치하는 연결 보기"):
                for m in match_list: st.caption(m)

            st.image(cv2.cvtColor(res_ref_img, cv2.COLOR_BGR2RGB), caption="회로도 분석 (V48)", use_column_width=True)
            st.image(cv2.cvtColor(res_tgt_img, cv2.COLOR_BGR2RGB), caption="실물 분석 (V35+Ensemble)", use_column_width=True)
