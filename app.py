import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import math
from PIL import Image
from collections import defaultdict
import os

# ==========================================
# [1. 설정 및 모든 모델 로드]
# ==========================================
st.set_page_config(page_title="BrainBoard V60: Zero Error", layout="wide")

REAL_MODEL_PATHS = ['best.pt', 'best(2).pt', 'best(3).pt']
MODEL_SYM_PATH = 'symbol.pt'

@st.cache_resource
def load_all_models():
    reals = []
    for p in REAL_MODEL_PATHS:
        if os.path.exists(p): reals.append(YOLO(p))
    # symbol.pt가 없으면 에러를 내지 않고 None을 반환하여 사이트 뻗음을 방지
    sym = YOLO(MODEL_SYM_PATH) if os.path.exists(MODEL_SYM_PATH) else None
    return reals, sym

models_real, model_sym = load_all_models()

# ==========================================
# [2. 안전한 보조 함수]
# ==========================================
def normalize_name(raw_name):
    name = raw_name.lower().strip()
    if any(x in name for x in ['res', 'r']): return 'RESISTOR'
    if any(x in name for x in ['cap', 'c']): return 'CAPACITOR'
    if any(x in name for x in ['v', 'volt', 'batt', 'source', 'vdc']): return 'SOURCE'
    return 'OTHER'

def get_safe_center(box):
    if not box or len(box) < 4: return (0, 0)
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

# ==========================================
# [3. 알고리즘 1: 회로도(정답지) 추출 - 성능 복구]
# ==========================================
def analyze_schematic_gold(img, model):
    if model is None: return img, set()
    
    res = model.predict(source=img, conf=0.15, imgsz=640, verbose=False)
    parts = []
    # 리스트 인덱스 에러 방지용 체크
    if not res or len(res) == 0 or not res[0].boxes: return img, set()

    for b in res[0].boxes:
        name = normalize_name(model.names[int(b.cls[0])])
        if name != 'OTHER':
            coords = b.xyxy[0].tolist()
            parts.append({'name': name, 'box': coords, 'center': get_safe_center(coords)})
    
    gold_netlist = set()
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            p1, p2 = parts[i], parts[j]
            dist = math.sqrt((p1['center'][0]-p2['center'][0])**2 + (p1['center'][1]-p2['center'][1])**2)
            if dist < 300: 
                key = "-".join(sorted([p1['name'], p2['name']]))
                gold_netlist.add(key)
    
    # 시각화
    for p in parts:
        x1, y1, x2, y2 = map(int, p['box'])
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img, p['name'], (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
    return img, gold_netlist

# ==========================================
# [4. 알고리즘 2: 실물 앙상블 분석 및 대조]
# ==========================================
def analyze_real_ensemble_verify(img, model_list, gold_netlist):
    h, w, _ = img.shape
    all_raw_results = []
    bb_box = [w*0.05, h*0.1, w*0.95, h*0.9]

    # 모든 실물 모델의 지혜를 모음
    for m in model_list:
        res = m.predict(source=img, conf=0.45, imgsz=640, verbose=False) # 신뢰도 높여서 노이즈 차단
        if not res or len(res) == 0 or not res[0].boxes: continue
        
        for b in res[0].boxes:
            cls_name = m.names[int(b.cls[0])].lower()
            coords = b.xyxy[0].tolist()
            if 'breadboard' in cls_name: bb_box = coords; continue
            
            norm_name = normalize_name(cls_name)
            if norm_name != 'OTHER':
                all_raw_results.append({
                    'name': norm_name, 'box': coords, 'conf': float(b.conf[0]),
                    'center': get_safe_center(coords)
                })

    # 중복 박스 독하게 제거 (IoU 0.1 기준 - 조금만 겹쳐도 하나로 합침)
    all_raw_results.sort(key=lambda x: x['conf'], reverse=True)
    clean_bodies = []
    for curr in all_raw_results:
        is_dup = False
        for k in clean_bodies:
            # IoU 계산
            ix1, iy1 = max(curr['box'][0], k['box'][0]), max(curr['box'][1], k['box'][1])
            ix2, iy2 = min(curr['box'][2], k['box'][2]), min(curr['box'][3], k['box'][3])
            inter = max(0, ix2-ix1) * max(0, iy2-iy1)
            area1 = (curr['box'][2]-curr['box'][0])*(curr['box'][3]-curr['box'][1])
            area2 = (k['box'][2]-k['box'][0])*(k['box'][3]-k['box'][1])
            iou = inter / (area1 + area2 - inter) if (area1 + area2 - inter) > 0 else 0
            
            if iou > 0.1: is_dup = True; break
        if not is_dup: clean_bodies.append(curr)

    # 노드(세로줄) 추적 및 비교
    part_to_nodes = defaultdict(set)
    bb_w = max(1, bb_box[2] - bb_box[0])
    for i, p in enumerate(clean_bodies):
        node_id = int(((p['center'][0] - bb_box[0]) / bb_w) * 63)
        part_to_nodes[i].add(max(1, min(63, node_id)))

    current_netlist = set()
    errors = []
    for i in range(len(clean_bodies)):
        for j in range(i + 1, len(clean_bodies)):
            if part_to_nodes[i].intersection(part_to_nodes[j]):
                key = "-".join(sorted([clean_bodies[i]['name'], clean_bodies[j]['name']]))
                current_netlist.add(key)

    # 정답지 대조 및 흐름 검증
    for ref_conn in gold_netlist:
        if ref_conn not in current_netlist:
            errors.append(f"❌ 배선 누락: 회로도에는 있는 {ref_conn} 연결이 실물에는 없습니다.")
    
    for node, p_indices in part_to_nodes.items():
        roles = [clean_bodies[idx]['name'] for idx in p_indices]
        if 'SOURCE' in roles and 'CAPACITOR' in roles:
            errors.append(f"⚠️ 오결선: 커패시터가 저항 없이 전원 마디(N{node})에 직접 연결되었습니다!")

    # 실물 시각화
    for p in clean_bodies:
        x1, y1, x2, y2 = map(int, p['box'])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(img, p['name'], (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return img, errors

# ==========================================
# [5. UI 실행]
# ==========================================
st.title("🧠 BrainBoard V60: Zero Error Ensemble")

if model_sym is None:
    st.error("⚠️ symbol.pt 파일을 찾을 수 없어 회로도 분석이 불가능합니다."); st.stop()

col1, col2 = st.columns(2)
ref_file = col1.file_uploader("1. 회로도(정답지) 업로드", type=['jpg', 'png', 'jpeg'])
tgt_file = col2.file_uploader("2. 실물 사진(검증 대상) 업로드", type=['jpg', 'png', 'jpeg'])

if ref_file and tgt_file:
    if st.button("🚀 정밀 대조 분석 시작"):
        try:
            ref_cv = cv2.cvtColor(np.array(Image.open(ref_file)), cv2.COLOR_RGB2BGR)
            tgt_cv = cv2.cvtColor(np.array(Image.open(tgt_file)), cv2.COLOR_RGB2BGR)

            res_ref_img, gold_netlist = analyze_schematic_gold(ref_cv, model_sym)
            res_tgt_img, errors = analyze_real_ensemble_verify(tgt_cv, models_real, gold_netlist)

            st.divider()
            if not errors and len(gold_netlist) > 0: st.success("🎉 회로도 설계와 실물 배선이 완벽히 일치합니다!")
            else:
                for e in errors: st.error(e)

            st.image(cv2.cvtColor(res_ref_img, cv2.COLOR_BGR2RGB), caption="회로도 분석 (정답 추출)")
            st.image(cv2.cvtColor(res_tgt_img, cv2.COLOR_BGR2RGB), caption="실물 앙상블 분석 (오결선 검증)")
        except Exception as e:
            st.error(f"❌ 분석 중 치명적 오류 발생: {e}")
