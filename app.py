import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import math
from PIL import Image
from collections import defaultdict
import os

# ==========================================
# [1. 설정 및 모델 로딩]
# ==========================================
st.set_page_config(page_title="BrainBoard V59: Final Fix", layout="wide")

REAL_MODEL_PATHS = ['best.pt', 'best(2).pt', 'best(3).pt']
MODEL_SYM_PATH = 'symbol.pt'

@st.cache_resource
def load_all_models():
    reals = []
    for p in REAL_MODEL_PATHS:
        if os.path.exists(p): reals.append(YOLO(p))
    sym = YOLO(MODEL_SYM_PATH) if os.path.exists(MODEL_SYM_PATH) else None
    return reals, sym

models_real, model_sym = load_all_models()

# ==========================================
# [2. 안정적인 유틸리티]
# ==========================================
def get_safe_center(box):
    if not box or len(box) < 4: return (0, 0)
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

def normalize_name(raw_name):
    name = raw_name.lower().strip()
    if any(x in name for x in ['res', 'r']): return 'RESISTOR'
    if any(x in name for x in ['cap', 'c']): return 'CAPACITOR'
    if any(x in name for x in ['v', 'volt', 'batt', 'source', 'vdc']): return 'SOURCE'
    return 'OTHER'

# ==========================================
# [3. 알고리즘 1: 회로도(정답지) 추출 - 에러 방지형]
# ==========================================
def analyze_schematic_gold(img, model):
    if model is None: return img, set()
    
    res = model.predict(source=img, conf=0.15, imgsz=640, verbose=False)
    parts = []
    if not res or not res[0].boxes: return img, set()

    for b in res[0].boxes:
        cls_id = int(b.cls[0])
        name = normalize_name(model.names[cls_id])
        if name != 'OTHER':
            coords = b.xyxy[0].tolist()
            parts.append({'name': name, 'box': coords, 'center': get_safe_center(coords)})
    
    gold_netlist = set()
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            p1, p2 = parts[i], parts[j]
            dist = math.sqrt((p1['center'][0]-p2['center'][0])**2 + (p1['center'][1]-p2['center'][1])**2)
            if dist < 300: # 회로도상 연결 간격
                key = "-".join(sorted([p1['name'], p2['name']]))
                gold_netlist.add(key)
    
    # 시각화
    for p in parts:
        x1, y1, x2, y2 = map(int, p['box'])
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img, p['name'], (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
    return img, gold_netlist

# ==========================================
# [4. 알고리즘 2: 실물 분석 및 정밀 대조]
# ==========================================
def analyze_real_verify(img, model_list, gold_netlist):
    h, w, _ = img.shape
    raw_bodies, raw_legs = [], []
    bb_box = [w*0.05, h*0.1, w*0.95, h*0.9] # 브레드보드 기본값

    for m in model_list:
        res = m.predict(source=img, conf=0.12, imgsz=640, verbose=False)
        if not res or not res[0].boxes: continue
        
        for b in res[0].boxes:
            cls_name = m.names[int(b.cls[0])].lower()
            coords = b.xyxy[0].tolist()
            center = get_safe_center(coords)
            
            if 'breadboard' in cls_name: bb_box = coords
            elif any(x in cls_name for x in ['pin', 'leg', 'lead']):
                raw_legs.append({'center': center})
            elif 'wire' not in cls_name:
                raw_bodies.append({'name': normalize_name(cls_name), 'box': coords, 'center': center})

    # 노드 추적 (안정성 강화)
    part_to_nodes = defaultdict(set)
    bb_w = max(1, bb_box[2] - bb_box[0])
    for i, p in enumerate(raw_bodies):
        node_id = int(((p['center'][0] - bb_box[0]) / bb_w) * 63)
        part_to_nodes[i].add(max(1, min(63, node_id)))

    # 실물 넷리스트 생성 및 대조
    current_netlist = set()
    errors = []
    for i in range(len(raw_bodies)):
        for j in range(i + 1, len(raw_bodies)):
            # 두 부품이 같은 노드를 공유하는지 확인
            if part_to_nodes[i].intersection(part_to_nodes[j]):
                key = "-".join(sorted([raw_bodies[i]['name'], raw_bodies[j]['name']]))
                current_netlist.add(key)

    # 1. 누락 검사
    for ref_conn in gold_netlist:
        if ref_conn not in current_netlist:
            errors.append(f"❌ 연결 누락: 회로도에는 있는 {ref_conn} 연결이 실물에는 없습니다.")
    
    # 2. 오결선(흐름) 검사
    for node, p_indices in part_to_nodes.items():
        roles = [raw_bodies[idx]['name'] for idx in p_indices]
        if 'SOURCE' in roles and 'CAPACITOR' in roles:
            errors.append(f"⚠️ 오결선: 커패시터가 저항을 거치지 않고 전원 마디(N{node})에 직접 연결되었습니다!")

    # 실물 시각화
    for p in raw_bodies:
        x1, y1, x2, y2 = map(int, p['box'])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(img, p['name'], (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return img, errors

# ==========================================
# [5. UI 실행부]
# ==========================================
st.title("🧠 BrainBoard V59: Zero-Error Matcher")

if not models_real or model_sym is None:
    st.error("⚠️ 모델 로드 실패. 파일명을 확인하세요 (best.pt, symbol.pt 등)"); st.stop()

col1, col2 = st.columns(2)
ref_file = col1.file_uploader("1. 회로도 업로드", type=['jpg', 'png', 'jpeg'])
tgt_file = col2.file_uploader("2. 실물 사진 업로드", type=['jpg', 'png', 'jpeg'])

if ref_file and tgt_file:
    if st.button("🚀 정밀 대조 분석 시작"):
        try:
            ref_cv = cv2.cvtColor(np.array(Image.open(ref_file)), cv2.COLOR_RGB2BGR)
            tgt_cv = cv2.cvtColor(np.array(Image.open(tgt_file)), cv2.COLOR_RGB2BGR)

            # 분석 실행
            res_ref_img, gold_netlist = analyze_schematic_gold(ref_cv, model_sym)
            res_tgt_img, errors = analyze_real_verify(tgt_cv, models_real, gold_netlist)

            st.divider()
            if not errors: 
                st.success("🎉 회로도 설계와 실물 배선이 완벽하게 일치합니다!")
            else:
                for e in errors: st.error(e)

            st.image(cv2.cvtColor(res_ref_img, cv2.COLOR_BGR2RGB), caption="회로도 분석 (정답 추출)")
            st.image(cv2.cvtColor(res_tgt_img, cv2.COLOR_BGR2RGB), caption="실물 분석 (배선 오류 검증)")
        except Exception as e:
            st.error(f"❌ 분석 중 치명적 오류 발생: {e}")
