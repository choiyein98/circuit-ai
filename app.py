import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import math
from PIL import Image
from collections import defaultdict
import os

# ==========================================
# [1. 설정 및 모델 로드]
# ==========================================
st.set_page_config(page_title="BrainBoard V58: Final Stable", layout="wide")

REAL_MODEL_PATHS = ['best.pt', 'best(2).pt', 'best(3).pt']
MODEL_SYM_PATH = 'symbol.pt'

@st.cache_resource
def load_all_models():
    reals = []
    for p in REAL_MODEL_PATHS:
        if os.path.exists(p): reals.append(YOLO(p))
    # symbol.pt 파일 존재 여부를 반드시 체크합니다.
    sym = YOLO(MODEL_SYM_PATH) if os.path.exists(MODEL_SYM_PATH) else None
    return reals, sym

models_real, model_sym = load_all_models()

# ==========================================
# [2. Helper Functions]
# ==========================================
def normalize_name(raw_name):
    name = raw_name.lower().strip()
    if any(x in name for x in ['res', 'r']): return 'RESISTOR'
    if any(x in name for x in ['cap', 'c']): return 'CAPACITOR'
    if any(x in name for x in ['v', 'volt', 'batt', 'source', 'vdc']): return 'SOURCE'
    return 'OTHER'

def get_relation_key(p1, p2):
    return "-".join(sorted([p1, p2]))

# ==========================================
# [3. 알고리즘 1: 회로도(정답지) 추출]
# ==========================================
def analyze_schematic_gold(img, model):
    if model is None: return img, set() # 모델이 없으면 빈 세트 반환
    
    res = model.predict(source=img, conf=0.15, imgsz=640, verbose=False)
    parts = []
    for b in res[0].boxes:
        name = normalize_name(model.names[int(b.cls[0])])
        if name != 'OTHER':
            coords = b.xyxy[0].tolist()
            parts.append({'name': name, 'box': coords, 'center': ((coords[0]+coords[2])/2, (coords[1]+coords[3])/2)})
    
    gold_netlist = set()
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            p1, p2 = parts[i], parts[j]
            dist = math.sqrt((p1['center'][0]-p2['center'][0])**2 + (p1['center'][1]-p2['center'][1])**2)
            if dist < 300: # 회로도 내 연결 거리 기준
                gold_netlist.add(get_relation_key(p1['name'], p2['name']))
    
    # 시각화 (정답 확인용)
    for p in parts:
        x1, y1, x2, y2 = map(int, p['box'])
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img, p['name'], (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
    return img, gold_netlist

# ==========================================
# [4. 알고리즘 2: 실물 분석 및 대조]
# ==========================================
def analyze_real_verify(img, model_list, gold_netlist):
    h, w, _ = img.shape
    raw_bodies, raw_legs = [], []
    bb_box = [w*0.1, h*0.2, w*0.9, h*0.8]

    for m in model_list:
        res = m.predict(source=img, conf=0.1, imgsz=640, verbose=False)
        for b in res[0].boxes:
            name_idx = int(b.cls[0])
            name = m.names[name_idx].lower()
            coords = b.xyxy[0].tolist()
            center = ((coords[0]+coords[2])/2, (coords[1]+coords[3])/2)
            
            if 'breadboard' in name: bb_box = coords
            elif any(x in name for x in ['pin', 'leg', 'lead']):
                raw_legs.append({'center': center})
            elif 'wire' not in name:
                raw_bodies.append({'name': normalize_name(name), 'box': coords, 'center': center})

    # 마디(Node) 추적
    part_to_nodes = defaultdict(set)
    for i, p in enumerate(raw_bodies):
        node_id = int(((p['center'][0] - bb_box[0]) / max(1, bb_box[2] - bb_box[0])) * 60)
        part_to_nodes[i].add(node_id)

    # 실물 넷리스트 생성
    current_netlist = set()
    errors = []
    for i in range(len(raw_bodies)):
        for j in range(i + 1, len(raw_bodies)):
            if part_to_nodes[i].intersection(part_to_nodes[j]):
                current_netlist.add(get_relation_key(raw_bodies[i]['name'], raw_bodies[j]['name']))

    # 정답지 대조
    for ref in gold_netlist:
        if ref not in current_netlist:
            errors.append(f"❌ 배선 누락: 회로도에는 있는 {ref} 연결이 실물에는 없습니다.")
    
    # [사용자 지적 사항] 흐름 역전 방지
    for node, p_indices in part_to_nodes.items():
        roles = [raw_bodies[idx]['name'] for idx in p_indices]
        if 'SOURCE' in roles and 'CAPACITOR' in roles:
            errors.append(f"⚠️ 오결선: 커패시터가 저항을 거치지 않고 전원에 직접 연결되었습니다!")

    # 실물 시각화
    for p in raw_bodies:
        x1, y1, x2, y2 = map(int, p['box'])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(img, p['name'], (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return img, errors

# ==========================================
# [5. 메인 UI]
# ==========================================
st.title("🧠 BrainBoard V58: Full Circuit Matcher")

if model_sym is None:
    st.error("⚠️ symbol.pt 파일을 찾을 수 없어 회로도 분석이 불가능합니다."); st.stop()

col1, col2 = st.columns(2)
ref_file = col1.file_uploader("1. 회로도(정답지)", type=['jpg', 'png', 'jpeg'])
tgt_file = col2.file_uploader("2. 실물 사진(검증 대상)", type=['jpg', 'png', 'jpeg'])

if ref_file and tgt_file:
    if st.button("🚀 정밀 대조 분석 시작"):
        try:
            ref_cv = cv2.cvtColor(np.array(Image.open(ref_file)), cv2.COLOR_RGB2BGR)
            tgt_cv = cv2.cvtColor(np.array(Image.open(tgt_file)), cv2.COLOR_RGB2BGR)

            # 1. 회로도 정답지 추출
            res_ref_img, gold_netlist = analyze_schematic_gold(ref_cv, model_sym)
            # 2. 실물 검증
            res_tgt_img, errors = analyze_real_verify(tgt_cv, models_real, gold_netlist)

            st.divider()
            if not errors: st.success("🎉 축하합니다! 회로도와 실물 배선이 완벽히 일치합니다.")
            else:
                for e in errors: st.error(e)

            st.image(cv2.cvtColor(res_ref_img, cv2.COLOR_BGR2RGB), caption="회로도 분석 (정답 마디 추출)")
            st.image(cv2.cvtColor(res_tgt_img, cv2.COLOR_BGR2RGB), caption="실물 분석 (배선 오류 검증)")
        except Exception as e:
            st.error(f"❌ 분석 중 에러 발생: {e}")
