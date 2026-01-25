import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import math
from PIL import Image

# ==========================================
# [설정] V49: "놓치는 부품 없게 하라" (Recall 우선)
# ==========================================
st.set_page_config(page_title="BrainBoard V49: Recall Boost", layout="wide")

MODEL_REAL_PATH = 'best(3).pt'  # 실물 모델
MODEL_SYM_PATH = 'symbol.pt'    # 회로도 모델

PROXIMITY_THRESHOLD = 100
IOU_THRESHOLD = 0.3

# [핵심 수정 1] 저항 기준을 다시 낮춤 (0.65 -> 0.30)
# 이제 조금만 저항 같아 보여도 무조건 잡습니다.
CONFIDENCE_MAP = {
    'led': 0.50,
    'capacitor': 0.40,
    'voltage': 0.25,
    'source': 0.25,
    'resistor': 0.30,  # [하향 조정] 숨은 저항 찾기 위해 문턱 낮춤
    'wire': 0.25,
    'default': 0.25
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

def is_near_box(point, box, margin=50):
    px, py = point
    return (box[0]-margin) < px < (box[2]+margin) and (box[1]-margin) < py < (box[3]+margin)

# [핵심 수정 2] 크기 필터 더 완화 (작은 저항도 통과)
def is_valid_size(box, img_w, img_h):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    area = w * h
    img_area = img_w * img_h
    # 0.05% 크기만 되어도 통과 (아주 작은 부품 허용)
    if area < img_area * 0.0005: return False 
    return True

# 비율 필터 삭제 (다리가 긴 저항을 와이어로 착각해서 지우는 현상 방지)

def is_intersecting(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    return max(0, xB - xA) * max(0, yB - yA) > 0

def solve_overlap(parts, distance_threshold=80):
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
# [분석 1] 회로도 (Schematic) - 현상 유지
# ==========================================
def analyze_schematic(img, model):
    # 회로도는 잘 되므로 5% 유지
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
        if base_name in ['vdc', 'vsource', 'battery', 'voltage']: base_name = 'source'
        if base_name in ['cap', 'c', 'capacitor']: base_name = 'capacitor'
        if base_name in ['res', 'r', 'resistor']: base_name = 'resistor'
        
        raw_parts.append({'name': base_name, 'box': coords, 'center': center, 'conf': conf})

    clean_parts = solve_overlap(raw_parts)

    # 전원 보정
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
# [분석 2] 실물 보드 (Real) - 숨은 저항 찾기
# ==========================================
def analyze_real(img, model):
    height, width, _ = img.shape
    # conf=0.1 : 10% 확률이라도 있으면 일단 후보로 올림
    results = model.predict(source=img, save=False, conf=0.1, verbose=False)
    boxes = results[0].boxes

    raw_objects = {'body': [], 'leg': [], 'plus': [], 'minus': []}
    
    for box in boxes:
        cls_id = int(box.cls[0])
        name = model.names[cls_id].lower()
        conf = float(box.conf[0])
        
        # 1. 신뢰도 필터 (여기서 걸러짐)
        threshold = CONFIDENCE_MAP.get('default')
        for key in CONFIDENCE_MAP:
            if key in name: threshold = CONFIDENCE_MAP[key]; break
        
        if conf < threshold: continue
        if name in ['breadboard', 'text', 'hole']: continue
        
        coords = box.xyxy[0].tolist()
        
        # 2. 크기 필터 (너무 작은 것만 제거)
        if not is_valid_size(coords, width, height): continue
        
        # [삭제] 비율 필터 제거 (길쭉한 저항도 통과시키기 위함)

        item = {'name': name, 'box': coords, 'center': get_center(coords), 'conf': conf}
        
        if any(x in name for x in ['pin', 'leg', 'lead']): raw_objects['leg'].append(item)
        elif any(x in name for x in ['plus', 'positive', 'vcc', '5v']): raw_objects['plus'].append(item)
        elif any(x in name for x in ['minus', 'negative', 'gnd']): raw_objects['minus'].append(item)
        else: raw_objects['body'].append(item)

    clean_bodies = solve_overlap(raw_objects['body'], distance_threshold=80)

    # 시각화
    virtual_rails = {'plus': [], 'minus': []}
    virtual_rails['plus'].append({'box': [0, 0, width, height*0.20], 'type': 'VCC'})
    virtual_rails['minus'].append({'box': [0, height*0.80, width, height], 'type': 'GND'})
    
    for r in virtual_rails['plus']:
        cv2.rectangle(img, (0, 0), (width, int(height*0.20)), (0, 255, 255), 1)
    for r in virtual_rails['minus']:
        cv2.rectangle(img, (0, int(height*0.80)), (width, height), (255, 200, 0), 1)

    components = []
    for body in clean_bodies:
        components.append({'body': body, 'is_active': True})

    summary = {'total': 0, 'on': 0, 'off': 0, 'details': {}}
    
    # 와이어가 2개 이상이면 전원 연결된 것으로 간주
    wire_count = sum(1 for c in components if 'wire' in c['body']['name'])
    if wire_count >= 2:
        summary['details']['source'] = {'count': 1}

    for comp in components:
        name = comp['body']['name']
        color = (0, 255, 0)
        status = f"{name} {comp['body']['conf']:.2f}"
        
        x1, y1, x2, y2 = map(int, comp['body']['box'])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        
        label_size, baseline = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        y_label = max(y1, label_size[1] + 10)
        cv2.rectangle(img, (x1, y_label - label_size[1] - 10), (x1 + label_size[0], y_label + baseline - 10), color, -1)
        cv2.putText(img, status, (x1, y_label - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        summary['total'] += 1
        summary['on'] += 1

        base_name = name.split('_')[0].split(' ')[0]
        if base_name in ['voltage', 'source', 'battery']: base_name = 'source'
        if base_name in ['cap', 'c', 'capacitor']: base_name = 'capacitor'
        if base_name in ['res', 'r', 'resistor']: base_name = 'resistor'
        
        if base_name not in summary['details']: summary['details'][base_name] = {'count': 0}
        summary['details'][base_name]['count'] += 1
        
    return img, summary

# ==========================================
# [Main UI]
# ==========================================
st.title("🧠 BrainBoard V49: Final Fix")
st.markdown("### 🔍 V49 업데이트: 숨은 저항 찾기 (기준 완화)")

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

    if st.button("🚀 분석 시작 (Analyze)"):
        with st.spinner("AI가 부품을 정밀 탐색 중..."):
            
            res_ref_img, ref_data = analyze_schematic(ref_cv.copy(), model_sym)
            res_tgt_img, tgt_data = analyze_real(tgt_cv.copy(), model_real)

            issues = []
            all_parts = set(ref_data['details'].keys()) | set(tgt_data['details'].keys())
            counts_match = True
            
            for part in all_parts:
                if part in ['wire', 'breadboard', 'text', 'hole']: continue
                
                ref_c = ref_data['details'].get(part, 0)
                tgt_c = tgt_data['details'].get(part, {}).get('count', 0)
                
                if ref_c != tgt_c:
                    issues.append(f"⚠️ {part.capitalize()} 개수 불일치 (회로도:{ref_c}개 vs 실물:{tgt_c}개)")
                    counts_match = False
                else:
                    issues.append(f"✅ {part.capitalize()} 개수 일치 ({ref_c}개)")

            st.divider()
            
            if counts_match:
                st.success("🎉 회로 구성이 완벽합니다!")
            else:
                st.warning("⚠️ 회로 구성에 차이가 있습니다.")
            
            for i in issues:
                if "✅" in i: st.caption(i)
                else: st.error(i)

            st.image(cv2.cvtColor(res_ref_img, cv2.COLOR_BGR2RGB), caption="PSpice 회로도 분석", use_column_width=True)
            st.image(cv2.cvtColor(res_tgt_img, cv2.COLOR_BGR2RGB), caption="실물 보드 분석 (숨은 저항 탐색)", use_column_width=True)
