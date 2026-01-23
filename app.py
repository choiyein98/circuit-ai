import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import math
from PIL import Image

# ==========================================
# [설정 및 상수]
# ==========================================
st.set_page_config(page_title="BrainBoard V44", layout="wide")

MODEL_REAL_PATH = 'best.pt'        # 실제 보드용 모델 경로
MODEL_SYM_PATH = 'symbol.pt'       # 회로도용 모델 경로 (변경됨)
PROXIMITY_THRESHOLD = 60
IOU_THRESHOLD = 0.3

CONFIDENCE_MAP = {
    'led': 0.60, 'capacitor': 0.45, 'voltage': 0.25,
    'source': 0.25, 'resistor': 0.50, 'wire': 0.35, 'default': 0.35
}

# [NEW] 60x10 브레드보드 논리 모델 클래스
class Breadboard60x10:
    def __init__(self):
        self.width = 60    # 가로 60칸 (0~59)
        self.height = 10   # 세로 10칸 (0~9)
        self.split_index = 5 # 0~4(Top)와 5~9(Bottom) 분리 기준

    def get_node_id(self, x, y):
        """좌표(x,y)를 입력받아 전기적 노드 ID 반환"""
        if not (0 <= x < self.width and 0 <= y < self.height):
            return "OUT" # 보드 밖
        
        # 세로줄(x)이 같으면 연결, 단 중앙 분리대(y=5) 기준 상하 분리
        if y < self.split_index:
            return f"Node_{x}_Top"
        else:
            return f"Node_{x}_Bottom"

    def check_is_short(self, pin1, pin2):
        """두 핀이 같은 노드에 연결되어 합선인지 확인 (True=합선)"""
        node1 = self.get_node_id(*pin1)
        node2 = self.get_node_id(*pin2)
        
        # 둘 중 하나라도 보드 밖이면 판단 유보(False)
        if node1 == "OUT" or node2 == "OUT": 
            return False
            
        # 노드 ID가 같으면 합선
        return (node1 == node2)

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

def non_max_suppression(boxes, iou_thresh):
    if not boxes: return []
    kept = []
    for curr in boxes:
        is_dup = False
        for k in kept:
            if calculate_iou(curr['box'], k['box']) > iou_thresh: is_dup = True; break
        if not is_dup: kept.append(curr)
    return kept

def get_center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

def is_near_box(point, box, margin=25):
    px, py = point
    return (box[0]-margin) < px < (box[2]+margin) and (box[1]-margin) < py < (box[3]+margin)

def is_valid_size(box, img_w, img_h):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    if (w * h) < (img_w * img_h * 0.001): return False
    return True

def is_valid_resistor_size(box, img_w, img_h):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    if (w * h) > (img_w * img_h * 0.05): return False
    return True

def is_intersecting(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    return max(0, xB - xA) * max(0, yB - yA) > 0

def solve_overlap(parts, distance_threshold=60):
    if not parts: return []
    if 'conf' in parts[0]:
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
        if not is_duplicate:
            final_parts.append(current)
    return final_parts

# [NEW] 픽셀 좌표 -> 60x10 그리드 좌표 변환
def map_pixel_to_grid(px, py, img_w, img_h):
    # 사진에서 브레드보드가 차지하는 영역 가정 (여백 조정 필요)
    margin_x = img_w * 0.05  # 좌측 여백 5%
    margin_y = img_h * 0.15  # 상단 여백 15%
    board_w = img_w * 0.90   # 보드 가로폭 90%
    board_h = img_h * 0.70   # 보드 세로폭 70%
    
    rel_x = (px - margin_x) / board_w
    rel_y = (py - margin_y) / board_h
    
    grid_x = int(rel_x * 60)
    grid_y = int(rel_y * 10)
    
    # 좌표 클램핑 (0~59, 0~9)
    grid_x = max(0, min(59, grid_x))
    grid_y = max(0, min(9, grid_y))
    
    return (grid_x, grid_y)

# ==========================================
# [분석 함수]
# ==========================================
def analyze_schematic(img, model):
    results = model.predict(source=img, save=False, conf=0.15, verbose=False)
    boxes = results[0].boxes
    raw_parts = []
    
    for box in boxes:
        cls_id = int(box.cls[0])
        name = model.names[cls_id].lower()
        conf = float(box.conf[0])
        coords = box.xyxy[0].tolist()
        center = get_center(coords)
        
        base_name = name.split('_')[0].split(' ')[0]
        if base_name in ['vdc', 'vsource', 'battery', 'voltage']: base_name = 'source'
        if base_name in ['cap', 'c', 'capacitor']: base_name = 'capacitor'
        if base_name in ['res', 'r', 'resistor']: base_name = 'resistor'
        
        raw_parts.append({'name': base_name, 'box': coords, 'center': center, 'conf': conf})

    clean_parts = solve_overlap(raw_parts)

    if clean_parts:
        leftmost_part = min(clean_parts, key=lambda p: p['center'][0])
        if leftmost_part['name'] != 'source':
            leftmost_part['name'] = 'source'

    summary = {'total': 0, 'details': {}}
    for part in clean_parts:
        name = part['name']
        x1, y1, x2, y2 = map(int, part['box'])
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img, name, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        summary['total'] += 1
        summary['details'][name] = summary['details'].get(name, 0) + 1
        
    return img, summary

def analyze_real(img, model):
    height, width, _ = img.shape
    results = model.predict(source=img, save=False, conf=0.1, verbose=False)
    boxes = results[0].boxes

    # [NEW] 브레드보드 논리 객체 생성
    logic_board = Breadboard60x10()

    raw_objects = {'body': [], 'leg': [], 'hole': [], 'plus': [], 'minus': []}
    
    for box in boxes:
        cls_id = int(box.cls[0])
        name = model.names[cls_id].lower()
        conf = float(box.conf[0])
        
        threshold = CONFIDENCE_MAP.get('default')
        for key in CONFIDENCE_MAP:
            if key in name: threshold = CONFIDENCE_MAP[key]; break
        
        if conf < threshold: continue
        if name in ['breadboard', 'text']: continue
        
        coords = box.xyxy[0].tolist()
        if not is_valid_size(coords, width, height): continue
        if 'resistor' in name or 'capacitor' in name:
             if not is_valid_resistor_size(coords, width, height): continue

        item = {'name': name, 'box': coords, 'center': get_center(coords), 'conf': conf}
        
        if any(x in name for x in ['pin', 'leg', 'lead']): raw_objects['leg'].append(item)
        elif 'hole' in name: raw_objects['hole'].append(item)
        elif any(x in name for x in ['plus', 'positive', 'vcc', '5v']): raw_objects['plus'].append(item)
        elif any(x in name for x in ['minus', 'negative', 'gnd']): raw_objects['minus'].append(item)
        else: raw_objects['body'].append(item)

    clean_bodies = solve_overlap(raw_objects['body'], distance_threshold=60)
    objects = {'body': clean_bodies, 'leg': non_max_suppression(raw_objects['leg'], IOU_THRESHOLD)}

    # Virtual Rails
    virtual_rails = {'plus': [], 'minus': []}
    virtual_rails['plus'].append({'box': [0, 0, width, height*0.15], 'type': 'VCC (Top)'})
    virtual_rails['minus'].append({'box': [0, height*0.70, width, height], 'type': 'GND (Bottom)'})
    virtual_rails['plus'].append({'box': [0, 0, width*0.15, height], 'type': 'VCC (Left)'})

    # Draw Rails
    for r in virtual_rails['minus']:
        x1, y1, x2, y2 = map(int, r['box'])
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 200, 0), 2)
    for r in virtual_rails['plus']:
        x1, y1, x2, y2 = map(int, r['box'])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)

    components = []
    detected_power_wires = 0
    short_circuit_count = 0 # [NEW] 합선 카운트

    # Body-Leg Association
    for body in objects['body']:
        bx, by = body['center']
        dists = []
        for leg in objects['leg']:
            lx, ly = leg['center']
            d = math.sqrt((bx-lx)**2 + (by-ly)**2)
            dists.append((d, leg))
        dists.sort(key=lambda x: x[0])
        
        bw, bh = body['box'][2]-body['box'][0], body['box'][3]-body['box'][1]
        max_d = math.sqrt(bw**2 + bh**2) * 2.5
        my_legs = [leg for d, leg in dists[:2] if d < max_d]
        
        # Fallback for missing legs
        if 'wire' in body['name']: pass
        else:
            if len(my_legs) < 2:
                x1, y1, x2, y2 = body['box']
                if bh > bw:
                    f1={'center':((x1+x2)/2,y1),'box':[x1,y1,x2,y1],'name':'virt'}
                    f2={'center':((x1+x2)/2,y2),'box':[x1,y2,x2,y2],'name':'virt'}
                else:
                    f1={'center':(x1,(y1+y2)/2),'box':[x1,y1,x1,y2],'name':'virt'}
                    f2={'center':(x2,(y1+y2)/2),'box':[x2,y1,x2,y2],'name':'virt'}
                if len(my_legs)==0: my_legs=[f1,f2]
                elif len(my_legs)==1: my_legs.append(f2)
        
        # [NEW] 합선(Short Circuit) 검사
        is_shorted = False
        if len(my_legs) >= 2:
            # 픽셀 좌표를 그리드 좌표로 변환
            p1 = map_pixel_to_grid(*my_legs[0]['center'], width, height)
            p2 = map_pixel_to_grid(*my_legs[1]['center'], width, height)
            
            # 논리 모델로 체크
            if logic_board.check_is_short(p1, p2):
                is_shorted = True
                short_circuit_count += 1

        components.append({'body': body, 'legs': my_legs, 'is_active': False, 'is_shorted': is_shorted})

    # Connectivity Check
    for comp in components:
        # 합선된 부품은 전원이 연결되어도 무조건 비정상 처리 (추후 시각화에서 처리)
        is_power_connected = False
        for leg in comp['legs']:
            leg['is_terminated'] = False
            leg['node'] = None
            for vcc in virtual_rails['plus']:
                if is_near_box(leg['center'], vcc['box'], margin=PROXIMITY_THRESHOLD):
                    leg['node'] = "VCC"; leg['is_terminated'] = True; is_power_connected = True; break
            if not leg['is_terminated']:
                for gnd in virtual_rails['minus']:
                    if is_near_box(leg['center'], gnd['box'], margin=PROXIMITY_THRESHOLD):
                        leg['node'] = "GND"; leg['is_terminated'] = True; is_power_connected = True; break
            if not leg['is_terminated']:
                for other in components:
                    if other == comp: continue
                    for o_leg in other['legs']:
                        d = math.sqrt((leg['center'][0]-o_leg['center'][0])**2 + (leg['center'][1]-o_leg['center'][1])**2)
                        if d < PROXIMITY_THRESHOLD:
                            leg['is_terminated'] = True
                            cv2.line(img, (int(leg['center'][0]), int(leg['center'][1])),
                                     (int(o_leg['center'][0]), int(o_leg['center'][1])), (0, 255, 255), 3)
                            break
                    if leg['is_terminated']: break
        
        if 'wire' in comp['body']['name']:
            wire_box = comp['body']['box']
            for rail in virtual_rails['plus'] + virtual_rails['minus']:
                if is_intersecting(wire_box, rail['box']):
                    is_power_connected = True; break
        
        if is_power_connected:
            comp['is_active'] = True
            if 'wire' in comp['body']['name']: detected_power_wires += 1

    # Propagation
    active_nodes = set(["VCC"]); changed = True
    while changed:
        changed = False
        for comp in components:
            if comp['is_active']: continue
            connected = False
            other_nodes = []
            for leg in comp['legs']:
                if leg['node'] in active_nodes: connected = True
                elif leg['node'] and leg['node'] != "GND": other_nodes.append(leg['node'])
            if not connected:
                for leg in comp['legs']:
                    for ac in [c for c in components if c['is_active']]:
                        for al in ac['legs']:
                            d = math.sqrt((leg['center'][0]-al['center'][0])**2 + (leg['center'][1]-al['center'][1])**2)
                            if d < PROXIMITY_THRESHOLD: connected = True; break
                        if connected: break
                    if connected: break
            if connected:
                comp['is_active'] = True; changed = True
                for n in other_nodes:
                    if n not in active_nodes: active_nodes.add(n)

    # Floating Check & Short Check Override
    for comp in components:
        # 합선된 부품은 강제로 OFF 처리
        if comp['is_shorted']:
            comp['is_active'] = False
            continue

        if 'wire' in comp['body']['name'] and comp['is_active']: continue
        if comp['is_active']:
            all_legs_connected = True
            for leg in comp['legs']:
                if not leg.get('is_terminated', False): all_legs_connected = False; break
            if 'wire' in comp['body']['name'] and len(comp['legs']) < 2: all_legs_connected = False
            if not all_legs_connected: comp['is_active'] = False

    summary = {'total': 0, 'on': 0, 'off': 0, 'short': short_circuit_count, 'details': {}}
    if detected_power_wires > 0: summary['details']['source'] = {'count': 1}

    # Visualization
    for comp in components:
        name = comp['body']['name']
        x1, y1, x2, y2 = map(int, comp['body']['box'])
        
        # [NEW] 상태별 색상 및 텍스트 설정
        if comp['is_shorted']:
            color = (0, 0, 255) # 빨간색 (합선)
            status = "SHORT!"
            # 합선 시각화 (두 다리 연결)
            if len(comp['legs']) >= 2:
                lx1, ly1 = map(int, comp['legs'][0]['center'])
                lx2, ly2 = map(int, comp['legs'][1]['center'])
                cv2.line(img, (lx1, ly1), (lx2, ly2), (0, 0, 255), 4) # 두꺼운 빨간선
        elif comp['is_active']:
            color = (0, 255, 0) # 초록색 (ON)
            status = "ON"
        else:
            color = (0, 0, 255) # 빨간색 (OFF)
            status = "OFF"

        thickness = 3
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        text_pos = (x1, y1 - 10) if y1 > 25 else (x1, y2 + 25)
        cv2.putText(img, status, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        if 'wire' in name:
             for leg in comp['legs']:
                lx, ly = map(int, leg['center'])
                cv2.rectangle(img, (lx-10, ly-10), (lx+10, ly+10), color, 2)
        
        summary['total'] += 1
        if comp['is_active']: summary['on'] += 1
        else: summary['off'] += 1

        base_name = name.split('_')[0].split(' ')[0]
        if base_name in ['voltage', 'source', 'battery']: base_name = 'source'
        if base_name in ['cap', 'c', 'capacitor']: base_name = 'capacitor'
        if base_name in ['res', 'r', 'resistor']: base_name = 'resistor'
        
        if base_name not in summary['details']: summary['details'][base_name] = {'count': 0}
        summary['details'][base_name]['count'] += 1

    return img, summary

# ==========================================
# [WEB APP UI] Streamlit Main Code
# ==========================================
st.title("🧠 BrainBoard V44: AI Circuit Verifier")
st.markdown("### PSpice 회로도와 실제 브레드보드(60x10) 사진을 업로드하세요.")

@st.cache_resource
def load_models():
    return YOLO(MODEL_REAL_PATH), YOLO(MODEL_SYM_PATH)

try:
    model_real, model_sym = load_models()
    st.sidebar.success("✅ AI 모델 로드 완료!")
except Exception as e:
    st.error(f"모델 파일을 찾을 수 없습니다: {e}")
    st.stop()

col1, col2 = st.columns(2)
ref_file = col1.file_uploader("1. 회로도(Schematic) 업로드", type=['jpg', 'png', 'jpeg'])
tgt_file = col2.file_uploader("2. 실물(Real Board) 업로드", type=['jpg', 'png', 'jpeg'])

if ref_file and tgt_file:
    ref_image = Image.open(ref_file)
    tgt_image = Image.open(tgt_file)
    ref_cv = cv2.cvtColor(np.array(ref_image), cv2.COLOR_RGB2BGR)
    tgt_cv = cv2.cvtColor(np.array(tgt_image), cv2.COLOR_RGB2BGR)

    if st.button("🚀 회로 검증 시작 (Analyze)"):
        with st.spinner("AI가 회로를 분석 중입니다..."):
            res_ref_img, ref_data = analyze_schematic(ref_cv.copy(), model_sym)
            res_tgt_img, tgt_data = analyze_real(tgt_cv.copy(), model_real)

            issues = []
            all_parts = set(ref_data['details'].keys()) | set(tgt_data['details'].keys())
            counts_match = True
            
            for part in all_parts:
                if part in ['wire', 'breadboard', 'text']: continue
                ref_c = ref_data['details'].get(part, 0)
                tgt_c = tgt_data['details'].get(part, {}).get('count', 0)
                if ref_c != tgt_c:
                    issues.append(f"{part.capitalize()} 개수 불일치 (회로도:{ref_c} vs 실물:{tgt_c})")
                    counts_match = False

            # [NEW] 합선 확인 로직 추가
            short_count = tgt_data.get('short', 0)
            if short_count > 0:
                issues.append(f"🚨 위험: 합선(Short) 발견! ({short_count}개 부품이 같은 줄에 연결됨)")

            # 연결 확인 (OFF 부품 확인)
            # 합선된 부품은 OFF 카운트에도 포함될 수 있으므로 중복 주의 (UI 표시용)
            off_count = tgt_data['off']
            if off_count > 0:
                 issues.append(f"연결 끊김/비정상 부품 발견 ({off_count}개 OFF 또는 SHORT)")

            connection_ok = (off_count == 0) and (short_count == 0)

            st.divider()
            if counts_match and connection_ok:
                st.success("🎉 Perfect! 모든 부품 개수와 연결이 정확합니다.")
            else:
                st.error("❌ 오류가 발견되었습니다.")
                for i in issues:
                    st.write(f"- {i}")

            st.image(cv2.cvtColor(res_ref_img, cv2.COLOR_BGR2RGB), caption="PSpice 회로도 분석", use_column_width=True)
            st.image(cv2.cvtColor(res_tgt_img, cv2.COLOR_BGR2RGB), caption="실물 보드 분석 (V44 with Short Check)", use_column_width=True)
