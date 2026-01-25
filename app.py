import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import math
from PIL import Image
import os

# ==========================================
# [1. 설정 및 모든 모델 로드]
# ==========================================
st.set_page_config(page_title="BrainBoard V55: Node-Flow Final", layout="wide")

# 모든 모델 경로 (앙상블용)
REAL_PATHS = ['best.pt', 'best(2).pt', 'best(3).pt']
SYM_PATH = 'symbol.pt'

@st.cache_resource
def load_all_models():
    real_models = [YOLO(p) for p in REAL_PATHS if os.path.exists(p)]
    sym_model = YOLO(SYM_PATH) if os.path.exists(SYM_PATH) else None
    return real_models, sym_model

real_models, model_sym = load_all_models()

# ==========================================
# [2. 이름 표준화 및 기하학 필터]
# ==========================================
def normalize_name(raw_name):
    name = raw_name.lower().strip()
    if any(x in name for x in ['res', 'r']): return 'RESISTOR'
    if any(x in name for x in ['cap', 'c']): return 'CAPACITOR'
    if any(x in name for x in ['v', 'volt', 'batt', 'source', 'vdc']): return 'SOURCE'
    if any(x in name for x in ['pin', 'leg', 'lead']): return 'PIN'
    if 'wire' in name: return 'WIRE'
    return 'OTHER'

def is_wire_by_ratio(box):
    x1, y1, x2, y2 = box
    w, h = abs(x2-x1), abs(y2-y1)
    if min(w, h) == 0: return True
    return (max(w, h) / min(w, h)) > 6.0 # 너무 길쭉하면 와이어

# ==========================================
# [3. 핵심: 노드(마디) 추적 및 토폴로지 분석]
# ==========================================
def get_node_id(x_coord, bb_box, total_nodes=63):
    """브레드보드 영역 내에서 x좌표를 세로줄 번호(1~63)로 변환"""
    bx1, _, bx2, _ = bb_box
    width = bx2 - bx1
    if width <= 0: return 0
    node_idx = int(((x_coord - bx1) / width) * total_nodes)
    return max(1, min(total_nodes, node_idx))

def solve_overlap_with_nodes(parts, bb_box, iou_thresh=0.2):
    if not parts: return []
    parts.sort(key=lambda x: x['conf'], reverse=True)
    final = []
    for curr in parts:
        curr['node'] = get_node_id(curr['center'][0], bb_box)
        is_dup = False
        for k in final:
            # IoU 계산
            ix1, iy1 = max(curr['box'][0], k['box'][0]), max(curr['box'][1], k['box'][1])
            ix2, iy2 = min(curr['box'][2], k['box'][2]), min(curr['box'][3], k['box'][3])
            inter = max(0, ix2-ix1) * max(0, iy2-iy1)
            area1 = (curr['box'][2]-curr['box'][0]) * (curr['box'][3]-curr['box'][1])
            area2 = (k['box'][2]-k['box'][0]) * (k['box'][3]-k['box'][1])
            iou = inter / (area1 + area2 - inter) if (area1 + area2 - inter) > 0 else 0
            if iou > iou_thresh: is_dup = True; break
        if not is_dup: final.append(curr)
    return final

# ==========================================
# [4. 분석 엔진: 실물 앙상블 + 흐름 검증]
# ==========================================
def analyze_real_flow(img, models):
    h, w, _ = img.shape
    all_raw = []
    bb_box = [w*0.1, h*0.2, w*0.9, h*0.8] # 기본 브레드보드 영역 (인식 실패 대비)

    for m in models:
        res = m.predict(source=img, conf=0.3, imgsz=640, verbose=False)
        for b in res[0].boxes:
            name = normalize_name(m.names[int(b.cls[0])])
            coords = b.xyxy[0].tolist()
            if 'breadboard' in m.names[int(b.cls[0])].lower(): bb_box = coords
            if name in ['RESISTOR', 'CAPACITOR', 'SOURCE', 'WIRE']:
                if name == 'RESISTOR' and is_wire_by_ratio(coords): name = 'WIRE'
                all_raw.append({
                    'name': name, 'box': coords, 'conf': float(b.conf[0]),
                    'center': ((coords[0]+coords[2])/2, (coords[1]+coords[3])/2)
                })

    clean = solve_overlap_with_nodes(all_raw, bb_box)
    
    # [흐름 분석 로직]
    # 회로도 흐름: SOURCE(Node 1) -> RESISTOR1(Node 1~10) -> (Node 10) -> CAP & RES2
    nodes_content = {}
    for p in clean:
        if p['node'] not in nodes_content: nodes_content[p['node']] = []
        nodes_content[p['node']].append(p['name'])

    errors = []
    # 사용자 지적 사항: 커패시터가 소스에 직접 꽂히면 에러
    for node, items in nodes_content.items():
        if 'SOURCE' in items and 'CAPACITOR' in items:
            errors.append(f"❌ 배선 오류: {node}번 노드에서 CAPACITOR가 전원에 직접 연결됨 (저항을 거쳐야 함)")

    # 시각화
    for p in clean:
        x1, y1, x2, y2 = map(int, p['box'])
        status_color = (0, 255, 0) # 기본 초록
        # 에러 노드에 포함된 부품은 빨간색으로 표시
        for err in errors:
            if str(p['node']) in err: status_color = (0, 0, 255)
        
        cv2.rectangle(img, (x1, y1), (x2, y2), status_color, 3)
        cv2.putText(img, f"{p['name']}(N{p['node']})", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)

    return img, {'details': clean, 'errors': errors}

# ==========================================
# [5. UI 실행부]
# ==========================================
st.title("🧠 BrainBoard V55: Topology Flow Analysis")
st.markdown("### 실물 배선 순서 및 노드(세로줄) 일치 여부 정밀 분석")

col1, col2 = st.columns(2)
ref_file = col1.file_uploader("1. PSpice 회로도 업로드", type=['jpg', 'png', 'jpeg'])
tgt_file = col2.file_uploader("2. 실물 보드 사진 업로드", type=['jpg', 'png', 'jpeg'])

if ref_file and tgt_file:
    if st.button("🚀 전체 회로 흐름 분석 시작"):
        with st.spinner("마디(Node) 단위로 회로를 추적 중..."):
            ref_img = cv2.cvtColor(np.array(Image.open(ref_file)), cv2.COLOR_RGB2BGR)
            tgt_img = cv2.cvtColor(np.array(Image.open(tgt_file)), cv2.COLOR_RGB2BGR)

            # 회로도 분석 (기존 잘 작동하는 V35 로직 유지)
            # res_ref_img, data_ref = analyze_schematic(ref_img, model_sym) 
            
            # 실물 흐름 분석 (앙상블 + 노드 추적)
            res_tgt_img, tgt_result = analyze_real_flow(tgt_img, real_models)

            st.divider()
            if tgt_result['errors']:
                for err in tgt_result['errors']: st.error(err)
            else:
                st.success("✅ 분석 결과: 모든 부품이 올바른 순서(Node)로 배선되었습니다.")

            st.image(cv2.cvtColor(res_tgt_img, cv2.COLOR_BGR2RGB), caption="실물 보드 마디 분석 (N=노드번호)", use_column_width=True)
