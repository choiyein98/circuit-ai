import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import math
from PIL import Image

# ==========================================
# [설정] BrainBoard V54: Connectivity Added
# ==========================================
st.set_page_config(page_title="BrainBoard V54", layout="wide")

# 실물 모델 3개를 리스트로 정의 (앙상블용)
REAL_MODEL_PATHS = ['best.pt', 'best(2).pt', 'best(3).pt']
MODEL_SYM_PATH = 'symbol.pt'
LEG_EXTENSION_RANGE = 180

# ==========================================
# [Class] 회로 연결 분석기 (직렬/병렬 판독)
# ==========================================
class CircuitAnalyzer:
    def __init__(self, components, distance_threshold=60):
        """
        components: [{'name': 'resistor', 'box': [x1, y1, x2, y2]}, ...]
        distance_threshold: 같은 노드로 인식할 최대 거리 (픽셀)
        """
        self.components = components
        self.threshold = distance_threshold
        self.nodes = [] # 노드 리스트 (좌표 그룹)
        self.netlist = {} # 부품별 연결된 노드 ID Set

    # 부품의 양쪽 다리(Leg) 좌표 추정
    def _get_legs(self, box):
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        # 가로로 긴 부품 -> 좌우 끝이 다리
        if w > h:
            leg1 = (x1, (y1 + y2) / 2)
            leg2 = (x2, (y1 + y2) / 2)
        # 세로로 긴 부품 -> 상하 끝이 다리
        else:
            leg1 = ((x1 + x2) / 2, y1)
            leg2 = ((x1 + x2) / 2, y2)
        return [leg1, leg2]

    # 좌표를 노드 ID로 매핑 (가까우면 같은 노드)
    def _find_node_id(self, leg_point):
        lx, ly = leg_point
        for node_id, points in enumerate(self.nodes):
            for px, py in points:
                dist = math.sqrt((lx - px)**2 + (ly - py)**2)
                if dist < self.threshold:
                    self.nodes[node_id].append(leg_point)
                    return node_id
        
        # 새로운 노드 생성
        new_id = len(self.nodes)
        self.nodes.append([leg_point])
        return new_id

    # 회로망 구축
    def build_graph(self):
        for i, comp in enumerate(self.components):
            # 이름에 인덱스를 붙여 고유 ID 생성 (예: resistor_0)
            comp_id = f"{comp['name']}_{i}"
            legs = self._get_legs(comp['box'])
            
            connected_nodes = set()
            for leg in legs:
                node_id = self._find_node_id(leg)
                connected_nodes.add(node_id)
            
            self.netlist[comp_id] = connected_nodes
            
    # 관계 분석 (직렬 vs 병렬)
    def analyze_relationship(self):
        results = []
        comp_ids = list(self.netlist.keys())
        
        # 모든 부품 쌍 비교
        for i in range(len(comp_ids)):
            for j in range(i + 1, len(comp_ids)):
                id_a = comp_ids[i]
                id_b = comp_ids[j]
                
                nodes_a = self.netlist[id_a]
                nodes_b = self.netlist[id_b]
                
                # 교집합(공유 노드) 개수
                shared_nodes = nodes_a.intersection(nodes_b)
                count = len(shared_nodes)
                
                name_a = id_a.split('_')[0].upper()
                name_b = id_b.split('_')[0].upper()
                
                if count == 2:
                    results.append(f"🔗 **[병렬 (Parallel)]** {name_a} ∥ {name_b}")
                elif count == 1:
                    results.append(f"➖ **[직렬 (Series)]** {name_a} ─ {name_b}")
        
        return results

# ==========================================
# [Helper Functions] 공통 함수
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
# [중복 제거 1] V48 스타일 (회로도용)
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
                is_duplicate = True
                break
        if not is_duplicate:
            final_parts.append(current)
    return final_parts

# ==========================================
# [중복 제거 2] V35 스타일 (실물용)
# ==========================================
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

        if not is_dup:
            final.append(curr)
            
    return final

# ==========================================
# [분석 1] 회로도 (V48 로직)
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
        if name not in summary['details']: summary['details'][name] = 0
        summary['details'][name] += 1
        
    return img, summary

# ==========================================
# [분석 2] 실물 보드 (V35 + 앙상블 + 연결분석)
# ==========================================
def analyze_real_ensemble(img, model_list):
    h, w, _ = img.shape
    
    raw_bodies = []
    raw_pins = [] 
    
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
            elif 'breadboard' in name:
                continue
            else:
                raw_bodies.append({'name': name, 'box': coords, 'center': center, 'conf': conf, 'is_on': False})

    clean_bodies = solve_overlap_real_v35(raw_bodies, dist_thresh=60, iou_thresh=0.4)
    
    # [연결 상태(ON/OFF) 로직]
    power_active = False
    for b in clean_bodies:
        if 'wire' in b['name'] and b['center'][1] < h * 0.45:
            power_active = True; break
    if not power_active:
        for p in raw_pins:
            if p['center'][1] < h * 0.45:
                power_active = True; break

    if power_active:
        for comp in clean_bodies:
            cy = comp['center'][1]
            if cy < h*0.48 or cy > h*0.52: 
                comp['is_on'] = True

        for _ in range(3): 
            for comp in clean_bodies:
                if comp['is_on']: continue 
                cx, cy = comp['center']
                
                for p in raw_pins:
                    px, py = p['center']
                    if py < h*0.48 or py > h*0.52:
                         dist = math.sqrt((cx - px)**2 + (cy - py)**2)
                         if dist < LEG_EXTENSION_RANGE:
                             comp['is_on'] = True; break

                if comp['is_on']: continue

                for other in clean_bodies:
                    if not other['is_on']: continue
                    ocx, ocy = other['center']
                    dist = math.sqrt((cx - ocx)**2 + (cy - ocy)**2)
                    if dist < LEG_EXTENSION_RANGE * 1.5:
                        comp['is_on'] = True; break

    summary = {'total': 0, 'on': 0, 'off': 0, 'details': {}, 'connections': []}
    
    # [회로 연결 분석 실행]
    # 감지된 부품 정보를 추출하여 CircuitAnalyzer에 전달
    detected_comps = []
    for comp in clean_bodies:
        detected_comps.append({
            'name': comp['name'],
            'box': comp['box']
        })
    
    # 분석기 가동 (거리 임계값 60px)
    analyzer = CircuitAnalyzer(detected_comps, distance_threshold=60)
    analyzer.build_graph()
    summary['connections'] = analyzer.analyze_relationship()

    # 시각화 및 카운트
    for comp in clean_bodies:
        is_on = comp['is_on']
        raw_name = comp['name']
        
        norm_name = raw_name
        label_name = "" 
        if 'res' in raw_name: 
            norm_name = 'resistor'; label_name = "RES"
        elif 'cap' in raw_name: 
            norm_name = 'capacitor'; label_name = "CAP"
        elif 'wire' in raw_name:
            label_name = "WIRE"
        else:
            label_name = raw_name[:3].upper()
        
        if 'wire' not in raw_name:
            if norm_name not in summary['details']: summary['details'][norm_name] = {'count': 0}
            summary['details'][norm_name]['count'] += 1

        if is_on:
            color = (0, 255, 0)
            status = "ON"
            summary['on'] += 1
        else:
            color = (0, 0, 255)
            status = "OFF"
            summary['off'] += 1
        
        summary['total'] += 1
        
        display_text = f"{label_name}: {status}"
        x1, y1, x2, y2 = map(int, comp['box'])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.putText(img, display_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
    return img, summary

# ==========================================
# [Main UI]
# ==========================================
st.title("🧠 BrainBoard V54")

@st.cache_resource
def load_models():
    real_models = []
    try:
        for path in REAL_MODEL_PATHS:
            real_models.append(YOLO(path))
    except Exception:
        pass 
        
    sym_model = YOLO(MODEL_SYM_PATH)
    return real_models, sym_model

try:
    models_real, model_sym = load_models()
    if not models_real:
        st.error("❌ 실물 모델 파일(best.pt 등)을 찾을 수 없습니다.")
        st.stop()
    st.sidebar.success(f"✅ 모델 로드 성공! ({len(models_real)}개 앙상블)")
    
except Exception as e:
    st.error(f"모델 로드 중 오류 발생: {e}")
    st.stop()

col1, col2 = st.columns(2)
ref_file = col1.file_uploader("1. 회로도 업로드", type=['jpg', 'png', 'jpeg'])
tgt_file = col2.file_uploader("2. 실물 사진 업로드", type=['jpg', 'png', 'jpeg'])

if ref_file and tgt_file:
    ref_image = Image.open(ref_file)
    tgt_image = Image.open(tgt_file)
    ref_cv = cv2.cvtColor(np.array(ref_image), cv2.COLOR_RGB2BGR)
    tgt_cv = cv2.cvtColor(np.array(tgt_image), cv2.COLOR_RGB2BGR)

    if st.button("🚀 정밀 분석 실행"):
        with st.spinner("AI가 회로 구조를 분석하고 있습니다..."):
            
            res_ref_img, ref_data = analyze_schematic(ref_cv.copy(), model_sym)
            res_tgt_img, tgt_data = analyze_real_ensemble(tgt_cv.copy(), models_real)

            issues = []
            all_parts = set(ref_data['details'].keys()) | set(tgt_data['details'].keys())
            counts_match = True
            
            for part in all_parts:
                if part in ['wire', 'breadboard', 'text', 'hole', 'source']: continue
                
                ref_c = ref_data['details'].get(part, 0)
                tgt_c = tgt_data['details'].get(part, {}).get('count', 0)
                
                if ref_c != tgt_c:
                    issues.append(f"⚠️ {part.capitalize()} 개수 불일치 (회로도:{ref_c}개 vs 실물:{tgt_c}개)")
                    counts_match = False
                else:
                    issues.append(f"✅ {part.capitalize()} 개수 일치 ({ref_c}개)")

            st.divider()
            
            if counts_match:
                st.success("🎉 회로 구성(부품 개수)이 완벽합니다!")
            else:
                st.warning("⚠️ 부품 개수에 차이가 있습니다.")
            
            for i in issues:
                if "✅" in i: st.caption(i)
                else: st.error(i)

            # [회로 연결 분석 결과 표시]
            st.divider()
            st.subheader("🔌 회로 연결 구조 분석 (Connectivity)")
            
            if tgt_data['connections']:
                for conn in tgt_data['connections']:
                    st.write(conn)
            else:
                st.info("연결된 부품이 감지되지 않았습니다. (부품 간 거리가 너무 멀거나 감지가 안 됨)")

            st.image(cv2.cvtColor(res_ref_img, cv2.COLOR_BGR2RGB), caption="회로도 분석", use_column_width=True)
            st.image(cv2.cvtColor(res_tgt_img, cv2.COLOR_BGR2RGB), caption="실물 분석 (앙상블)", use_column_width=True)
