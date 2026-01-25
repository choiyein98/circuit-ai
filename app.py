import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import math
from PIL import Image
from collections import defaultdict  # 이 부분이 빠지면 에러 납니다!
import os

# ==========================================
# [1. 설정 및 모델 로드]
# ==========================================
st.set_page_config(page_title="BrainBoard V56: Stable Netlist", layout="wide")

REAL_MODEL_PATHS = ['best.pt', 'best(2).pt', 'best(3).pt']
MODEL_SYM_PATH = 'symbol.pt'

@st.cache_resource
def load_models():
    reals = []
    for p in REAL_MODEL_PATHS:
        if os.path.exists(p): reals.append(YOLO(p))
    sym = YOLO(MODEL_SYM_PATH) if os.path.exists(MODEL_SYM_PATH) else None
    return reals, sym

models_real, model_sym = load_models()

# ==========================================
# [2. Helper Functions]
# ==========================================
def get_center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

def calculate_iou(box1, box2):
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2]-box1[0])*(box1[3]-box1[1])
    area2 = (box2[2]-box2[0])*(box2[3]-box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

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

# ==========================================
# [3. 실물 분석 (흐름 검증 로직 통합)]
# ==========================================
def analyze_real_netlist(img, model_list):
    h, w, _ = img.shape
    raw_bodies = []
    raw_legs = [] 

    for model in model_list:
        # 인식을 위해 conf를 0.1로 낮춤 (인식 안 되는 문제 해결)
        res = model.predict(source=img, conf=0.1, imgsz=640, verbose=False)
        for b in res[0].boxes:
            name = model.names[int(b.cls[0])].lower()
            coords = b.xyxy[0].tolist()
            conf = float(b.conf[0])
            center = get_center(coords)
            
            if any(x in name for x in ['pin', 'leg', 'lead']):
                raw_legs.append({'box': coords, 'center': center})
            elif 'breadboard' not in name and 'hole' not in name: 
                raw_bodies.append({'name': name, 'box': coords, 'center': center, 'conf': conf})

    parts = solve_overlap(raw_bodies, is_real=True)

    # 노드 클러스터링 (세로줄 인식)
    grouped_legs = []
    for leg in raw_legs:
        assigned = False
        for group in grouped_legs:
            ref = group[0] 
            if abs(leg['center'][0] - ref['center'][0]) < 30: # 노드 범위 확장
                group.append(leg); assigned = True; break
        if not assigned: grouped_legs.append([leg])

    # 부품-노드 연결 매핑 및 역할 정의
    part_connections = defaultdict(set)
    for i, part in enumerate(parts):
        p_name = part['name'].lower()
        if 'res' in p_name: part['role'] = 'resistor'
        elif 'cap' in p_name: part['role'] = 'capacitor'
        elif any(x in p_name for x in ['source', 'volt', 'batt']): part['role'] = 'source'
        else: part['role'] = p_name

        for nid, group in enumerate(grouped_legs):
            for leg in group:
                dist = math.sqrt((part['center'][0]-leg['center'][0])**2 + (part['center'][1]-leg['center'][1])**2)
                if dist < 180: # 다리 탐색 범위 확장 (인식 복구)
                    part_connections[i].add(nid)

    connections = []
    flow_errors = []
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            p1, p2 = parts[i], parts[j]
            shared = part_connections[i].intersection(part_connections[j])
            if shared:
                rel = 'Parallel' if len(shared) >= 2 else 'Series'
                connections.append({'p1': p1['role'], 'p2': p2['role'], 'type': rel})
                # [흐름 검증] Source와 Capacitor가 직접 만나면 에러
                if (p1['role'] == 'source' and p2['role'] == 'capacitor') or \
                   (p2['role'] == 'source' and p1['role'] == 'capacitor'):
                    flow_errors.append(f"❌ 흐름 오류: Capacitor가 저항 없이 전원에 직접 연결됨!")

    # 시각화
    for p in parts:
        x1, y1, x2, y2 = map(int, p['box'])
        color = (0, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.putText(img, p['role'].upper(), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return img, {'parts': parts, 'connections': connections, 'errors': flow_errors}

# ==========================================
# [4. UI 및 실행]
# ==========================================
st.title("🧠 BrainBoard V56: Final Stable Ensemble")

if not models_real or not model_sym:
    st.error("❌ 모델 파일을 찾을 수 없습니다. (best.pt, symbol.pt 확인)"); st.stop()

col1, col2 = st.columns(2)
ref_file = col1.file_uploader("1. 회로도 업로드", type=['jpg', 'png', 'jpeg'])
tgt_file = col2.file_uploader("2. 실물 사진 업로드", type=['jpg', 'png', 'jpeg'])

if ref_file and tgt_file:
    if st.button("🚀 정밀 분석 시작"):
        tgt_img = cv2.cvtColor(np.array(Image.open(tgt_file)), cv2.COLOR_RGB2BGR)
        res_tgt_img, tgt_data = analyze_real_netlist(tgt_img, models_real)

        st.divider()
        if tgt_data['errors']:
            for err in tgt_data['errors']: st.error(err)
        else:
            st.success("✅ 배선 흐름이 정상입니다.")

        st.image(cv2.cvtColor(res_tgt_img, cv2.COLOR_BGR2RGB), caption="실물 분석 결과 (흐름 검증 포함)", use_column_width=True)
