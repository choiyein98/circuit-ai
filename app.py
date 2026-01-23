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

MODEL_REAL_PATH = 'best.pt'
MODEL_SYM_PATH = 'circuit_model_v3.pt'
PROXIMITY_THRESHOLD = 60
IOU_THRESHOLD = 0.3

CONFIDENCE_MAP = {
    'led': 0.60, 'capacitor': 0.45, 'voltage': 0.25,
    'source': 0.25, 'resistor': 0.50, 'wire': 0.35, 'default': 0.35
}

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
    w = x2 - x1
    h = y2 - y1
    area = w * h
    img_area = img_w * img_h
    if area < img_area * 0.001: return False
    return True

def is_valid_resistor_size(box, img_w, img_h):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    area = w * h
    img_area = img_w * img_h
    if area > img_area * 0.05: return False
    return True

def is_intersecting(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
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
                is_duplicate = True
                break
        if not is_duplicate:
            final_parts.append(current)
    return final_parts

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
        if name not in summary['details']: summary['details'][name] = 0
        summary['details'][name] += 1
        
    return img, summary

def analyze_real(img, model):
    height, width, _ = img.shape
    results = model.predict(source=img, save=False, conf=0.1, verbose=False)
    boxes = results[0].boxes

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
        components.append({'body': body, 'legs': my_legs, 'is_active': False})

    # Connectivity Check
    for comp in components:
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

    # Floating Check
    for comp in components:
        if 'wire' in comp['body']['name'] and comp['is_active']: continue
        if comp['is_active']:
            all_legs_connected = True
            for leg in comp['legs']:
                if not leg.get('is_terminated', False): all_legs_connected = False; break
            if 'wire' in comp['body']['name'] and len(comp['legs']) < 2: all_legs_connected = False
            if not all_legs_connected: comp['is_active'] = False

    summary = {'total': 0, 'on': 0, 'off': 0, 'details': {}}
    if detected_power_wires > 0: summary['details']['source'] = {'count': 1}

    for comp in components:
        name = comp['body']['name']
        color = (0, 255, 0) if comp['is_active'] else (0, 0, 255)
        status = "ON" if comp['is_active'] else "OFF"
        thickness = 3
        x1, y1, x2, y2 = map(int, comp['body']['box'])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        text_pos = (x1, y1 - 10) if y1 > 25 else (x1, y2 + 25)
        cv2.putText(img, status, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        if 'wire' in name:
             for leg in comp['legs']:
                lx, ly = map(int, leg['center'])
                cv2.rectangle(img, (lx-10, ly-10), (lx+10, ly+10), color, 2)
        summary['total'] += 1
        
        # [수정완료] if-else 분리
        if comp['is_active']:
            summary['on'] += 1
        else:
            summary['off'] += 1

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
st.markdown("### PSpice 회로도와 실제 브레드보드 사진을 업로드하세요.")

# 사이드바에서 모델 로드
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
    # 이미지 읽기
    ref_image = Image.open(ref_file)
    tgt_image = Image.open(tgt_file)
    ref_cv = cv2.cvtColor(np.array(ref_image), cv2.COLOR_RGB2BGR)
    tgt_cv = cv2.cvtColor(np.array(tgt_image), cv2.COLOR_RGB2BGR)

    if st.button("🚀 회로 검증 시작 (Analyze)"):
        with st.spinner("AI가 회로를 분석 중입니다..."):
            # 분석 실행
            res_ref_img, ref_data = analyze_schematic(ref_cv.copy(), model_sym)
            res_tgt_img, tgt_data = analyze_real(tgt_cv.copy(), model_real)

            # 결과 비교 로직
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

            connection_ok = (tgt_data['off'] == 0)
            if not connection_ok:
                issues.append(f"연결 끊김 발견 ({tgt_data['off']}개 부품 OFF)")

            # 결과 화면 출력
            st.divider()
            if counts_match and connection_ok:
                st.success("🎉 Perfect! 모든 부품 개수와 연결이 정확합니다.")
            else:
                st.error("❌ 오류가 발견되었습니다.")
                for i in issues:
                    st.write(f"- {i}")

            # 이미지 출력 (RGB 변환 필요)
            st.image(cv2.cvtColor(res_ref_img, cv2.COLOR_BGR2RGB), caption="PSpice 회로도 분석", use_column_width=True)
            st.image(cv2.cvtColor(res_tgt_img, cv2.COLOR_BGR2RGB), caption="실물 보드 분석 (V44)", use_column_width=True)
