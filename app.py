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
st.set_page_config(page_title="BrainBoard V57: Netlist Matcher", layout="wide")

REAL_MODEL_PATHS = ['best.pt', 'best(2).pt', 'best(3).pt']
MODEL_SYM_PATH = 'symbol.pt'

@st.cache_resource
def load_all_models():
    reals = [YOLO(p) for p in REAL_PATHS if os.path.exists(p)] if 'REAL_PATHS' in locals() else [YOLO(p) for p in REAL_MODEL_PATHS if os.path.exists(p)]
    sym = YOLO(MODEL_SYM_PATH) if os.path.exists(MODEL_SYM_PATH) else None
    return reals, sym

models_real, model_sym = load_all_models()

# ==========================================
# [2. 이름 표준화 및 관계 생성기]
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
# [3. 알고리즘 1: 회로도 정답지(Golden Netlist) 추출]
# ==========================================
def analyze_schematic_gold(img, model):
    res = model.predict(source=img, conf=0.15, imgsz=640, verbose=False)
    parts = []
    for b in res[0].boxes:
        name = normalize_name(model.names[int(b.cls[0])])
        if name != 'OTHER':
            coords = b.xyxy[0].tolist()
            parts.append({'name': name, 'box': coords, 'center': ((coords[0]+coords[2])/2, (coords[1]+coords[3])/2)})
    
    # 기하학적 위치로 직/병렬 관계 정답 도출
    gold_netlist = set()
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            p1, p2 = parts[i], parts[j]
            dist = math.sqrt((p1['center'][0]-p2['center'][0])**2 + (p1['center'][1]-p2['center'][1])**2)
            if dist < 300: # 선으로 연결된 거리라면 정답 셋에 추가
                gold_netlist.add(get_relation_key(p1['name'], p2['name']))
    return img, gold_netlist

# ==========================================
# [4. 알고리즘 2: 실물 배선(Physical Netlist) 추출 및 대조]
# ==========================================
def analyze_real_verify(img, model_list, gold_netlist):
    h, w, _ = img.shape
    raw_bodies, raw_legs = [], []
    bb_box = [w*0.1, h*0.2, w*0.9, h*0.8]

    for m in model_list:
        res = m.predict(source=img, conf=0.15, imgsz=640, verbose=False)
        for b in res[0].boxes:
            name = m.names[int(b.cls[0])].lower()
            coords = b.xyxy[0].tolist()
            if 'breadboard' in name: bb_box = coords
            if any(x in name for x in ['pin', 'leg', 'lead']):
                raw_legs.append({'center': ((coords[0]+coords[2])/2, (coords[1]+coords[3])/2)})
            elif 'wire' not in name:
                raw_bodies.append({'name': normalize_name(name), 'box': coords, 'center': ((coords[0]+coords[2])/2, (coords[1]+coords[3])/2)})

    # 중복 제거 및 노드(세로줄) 할당
    parts = [] # 중복 제거 로직(생략) 후의 결과
    # ... (기존 solve_overlap 로직 적용) ...
    
    # [마디 추적] 부품-노드 매핑
    part_to_nodes = defaultdict(set)
    for i, p in enumerate(raw_bodies): # 편의상 raw_bodies 사용
        node_id = int(((p['center'][0] - bb_box[0]) / (bb_box[2] - bb_box[0])) * 60)
        part_to_nodes[i].add(node_id)

    # [대조] 회로도 정답지와 실물 배선 비교
    current_netlist = set()
    errors = []
    for i in range(len(raw_bodies)):
        for j in range(i + 1, len(raw_bodies)):
            if part_to_nodes[i].intersection(part_to_nodes[j]):
                current_netlist.add(get_relation_key(raw_bodies[i]['name'], raw_bodies[j]['name']))

    # 결과 판정
    for ref in gold_netlist:
        if ref not in current_netlist:
            errors.append(f"❌ 배선 누락: 회로도의 {ref} 연결이 실물에서는 끊어져 있습니다.")
    
    # 사용자 지적: 커패시터 흐름 역전(Source 직접 연결) 체크
    for node, p_idx_list in part_to_nodes.items():
        roles = [raw_bodies[idx]['name'] for idx in p_idx_list]
        if 'SOURCE' in roles and 'CAPACITOR' in roles:
            errors.append(f"⚠️ 오결선: 커패시터가 저항을 거치지 않고 전원 마디에 직접 연결됨!")

    return img, errors

# ==========================================
# [5. UI 실행]
# ==========================================
st.title("🧠 BrainBoard V57: Golden Netlist Checker")
st.info("💡 회로도 설계(정답)와 실물 배선을 1:1로 대조하여 오결선을 찾아냅니다.")

col1, col2 = st.columns(2)
ref_file = col1.file_uploader("1. 회로도(정답지)", type=['jpg', 'png', 'jpeg'])
tgt_file = col2.file_uploader("2. 실물 사진(검증 대상)", type=['jpg', 'png', 'jpeg'])

if ref_file and tgt_file:
    if st.button("🚀 정밀 배선 대조 시작"):
        ref_img = cv2.cvtColor(np.array(Image.open(ref_file)), cv2.COLOR_RGB2BGR)
        tgt_img = cv2.cvtColor(np.array(Image.open(tgt_file)), cv2.COLOR_RGB2BGR)

        # 1. 회로도에서 정답(Golden Netlist) 추출
        _, gold_netlist = analyze_schematic_gold(ref_img, model_sym)
        # 2. 실물 배선 검증
        res_img, errors = analyze_real_verify(tgt_img, models_real, gold_netlist)

        st.divider()
        if not errors: st.success("✅ 축하합니다! 회로도와 배선이 완벽하게 일치합니다.")
        else:
            for e in errors: st.error(e)
        st.image(cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB), use_column_width=True)
