import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import math
from PIL import Image

# ==========================================
# [설정] BrainBoard V58: Body-Count / Pin-Connect
# ==========================================
st.set_page_config(page_title="BrainBoard V58", layout="wide")

# [모델 설정]
REAL_MODEL_PATHS = ['best.pt', 'best(2).pt', 'best(3).pt']
MODEL_SYM_PATH = 'symbol.pt'
LEG_EXTENSION_RANGE = 180
SHORT_CIRCUIT_IOU = 0.6

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

def get_center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

def dist(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

# ==========================================
# [Logic 1] 회로도 분석 (V48)
# ==========================================
def solve_overlap_schematic(parts):
    if not parts: return []
    parts.sort(key=lambda x: x['conf'], reverse=True)
    final = []
    for curr in parts:
        is_dup = False
        for k in final:
            if calculate_iou(curr['box'], k['box']) > 0.1: is_dup = True; break
            if dist(curr['center'], k['center']) < 80: is_dup = True; break
        if not is_dup: final.append(curr)
    return final

def analyze_schematic(img, model):
    results = model.predict(source=img, save=False, conf=0.05, verbose=False)
    raw = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        name = model.names[cls_id].lower()
        coords = box.xyxy[0].tolist()
        
        base = name.split('_')[0].split(' ')[0]
        if base in ['vdc', 'vsource', 'battery', 'voltage', 'v']: base = 'source'
        if base in ['cap', 'c', 'capacitor']: base = 'capacitor'
        if base in ['res', 'r', 'resistor']: base = 'resistor'
        
        raw.append({'name': base, 'box': coords, 'center': get_center(coords), 'conf': float(box.conf[0])})

    clean = solve_overlap_schematic(raw)
    
    # 소스(전원)가 없으면 가장 왼쪽 부품을 소스로 가정
    if clean and not any(p['name'] == 'source' for p in clean):
        min(clean, key=lambda p: p['center'][0])['name'] = 'source'

    # -----------------------------------------------------------
    # [NEW] 구조(순서) 검증 로직 추가: 소스와 가장 가까운 부품 찾기
    # -----------------------------------------------------------
    first_conn_type = "unknown"
    sources = [p for p in clean if p['name'] == 'source']
    
    if sources:
        src_center = sources[0]['center']
        min_dist = 99999
        for part in clean:
            if part['name'] == 'source': continue
            d = dist(src_center, part['center'])
            if d < min_dist:
                min_dist = d
                first_conn_type = part['name']
    # -----------------------------------------------------------

    summary = {'details': {}, 'first_component': first_conn_type} # 결과에 추가
    
    for part in clean:
        name = part['name']
        x1, y1, x2, y2 = map(int, part['box'])
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img, name, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        summary['details'][name] = summary['details'].get(name, 0) + 1
        
    return img, summary

# ==========================================
# [Logic 2] 실물 분석 (Body Count / Pin Connect)
# ==========================================
def solve_overlap_real(parts):
    if not parts: return []
    parts.sort(key=lambda x: x.get('conf', 0), reverse=True)
    final = []
    for curr in parts:
        is_dup = False
        for k in final:
            if calculate_iou(curr['box'], k['box']) > 0.4: is_dup = True; break
            if dist(curr['center'], k['center']) < 60: is_dup = True; break
        if not is_dup: final.append(curr)
    return final

def analyze_real_v58(img, model_list):
    h, w, _ = img.shape
    raw_bodies = [] # 저항, 커패시터, 와이어 (카운팅 및 시각화용)
    raw_pins = []   # 핀, 다리 (연결 확인용)

    # 1. 앙상블 탐지
    for model in model_list:
        res = model.predict(source=img, conf=0.10, verbose=False)
        for box in res[0].boxes:
            name = model.names[int(box.cls[0])].lower()
            coords = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            center = get_center(coords)

            # [Rule] 민감도 설정
            if 'cap' in name: thresh = 0.15
            elif 'res' in name: thresh = 0.20 
            elif 'wire' in name: thresh = 0.15
            else: thresh = 0.25
            
            if conf < thresh: continue

            # [Rule] 핀(Pin)과 바디(Body) 분리
            if any(x in name for x in ['pin', 'leg', 'lead']) and 'wire' not in name:
                raw_pins.append({'center': center, 'box': coords, 'is_active': False})
            elif 'breadboard' in name:
                continue
            else:
                raw_bodies.append({
                    'name': name, 'box': coords, 'center': center, 'conf': conf,
                    'is_on': False, 'is_short': False
                })

    # 2. 중복 제거
    clean_bodies = solve_overlap_real(raw_bodies)
    
    # 3. [Connectivity Logic] 핀 기반 연결 확인
    power_top = h * 0.15
    power_bot = h * 0.85 
    
    # (Step A) 1차 활성화: 전원부에 직접 닿은 핀 & 와이어 찾기
    for p in raw_pins:
        py = p['center'][1]
        if py < h * 0.25 or py > h * 0.75: 
            p['is_active'] = True

    for b in clean_bodies:
        y1, y2 = b['box'][1], b['box'][3]
        if y1 < h * 0.25 or y2 > h * 0.75:
            b['is_on'] = True

    # (Step B) 전파 (Propagation)
    for _ in range(3):
        for b in clean_bodies:
            if b['is_on']: continue
            for p in raw_pins:
                if p['is_active']:
                    if dist(b['center'], p['center']) < LEG_EXTENSION_RANGE:
                        b['is_on'] = True
                        break
        
        for b in clean_bodies:
            if b['is_on']:
                for p in raw_pins:
                    if not p['is_active']:
                        if dist(b['center'], p['center']) < LEG_EXTENSION_RANGE:
                            p['is_active'] = True

        for b1 in clean_bodies:
            if b1['is_on']:
                for b2 in clean_bodies:
                    if not b2['is_on']:
                        limit = LEG_EXTENSION_RANGE * 1.5 if 'wire' in b1['name'] else LEG_EXTENSION_RANGE
                        if dist(b1['center'], b2['center']) < limit:
                            b2['is_on'] = True

    # 4. [Safety Logic] 쇼트 감지
    for i, c1 in enumerate(clean_bodies):
        if 'wire' in c1['name']: continue
        for j, c2 in enumerate(clean_bodies):
            if i >= j or 'wire' in c2['name']: continue
            if calculate_iou(c1['box'], c2['box']) > SHORT_CIRCUIT_IOU:
                c1['is_short'] = True
                c2['is_short'] = True

    # -----------------------------------------------------------
    # [NEW] 구조(순서) 검증 로직 추가: 전원 레일과 가장 가까운 부품 찾기
    # -----------------------------------------------------------
    real_first_type = "unknown"
    min_rail_dist = 99999
    
    for b in clean_bodies:
        # 상단 끝(0) 또는 하단 끝(h)과의 수직 거리 중 최소값
        cy = b['center'][1]
        dist_to_rail = min(cy, h - cy) 
        
        # 전원부 영역(25%) 안에 있으면서 가장 가까운 부품 선정
        if dist_to_rail < h * 0.25:
            if dist_to_rail < min_rail_dist:
                min_rail_dist = dist_to_rail
                real_first_type = b['name']
    
    # 이름 정규화 (표시용)
    if 'res' in real_first_type: real_first_type = 'resistor'
    elif 'cap' in real_first_type: real_first_type = 'capacitor'
    # -----------------------------------------------------------

    # 5. 결과 집계 및 그리기
    summary = {'total': 0, 'on': 0, 'off': 0, 'short': 0, 
               'details': {}, 'first_component': real_first_type} # 결과에 추가
    
    for comp in clean_bodies:
        raw_name = comp['name']
        norm_name = raw_name
        label = raw_name[:3].upper()
        
        if 'res' in raw_name: norm_name = 'resistor'; label="RES"
        elif 'cap' in raw_name: norm_name = 'capacitor'; label="CAP"
        elif 'wire' in raw_name: label="WIRE"
        
        if 'wire' not in raw_name:
            if norm_name not in summary['details']: summary['details'][norm_name] = {'count': 0}
            summary['details'][norm_name]['count'] += 1

        if comp['is_short']:
            color = (0, 0, 255)
            text = "SHORT!"
            summary['short'] += 1
            summary['off'] += 1
        elif comp['is_on']:
            color = (0, 255, 0)
            text = "ON"
            summary['on'] += 1
        else:
            color = (0, 0, 255)
            text = "OFF"
            summary['off'] += 1
            
        summary['total'] += 1
        
        x1, y1, x2, y2 = map(int, comp['box'])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.putText(img, f"{label}:{text}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
    return img, summary

# ==========================================
# [UI] Main
# ==========================================
st.title("🧠 BrainBoard V58: Body-Count & Pin-Connect & Topology")
st.markdown("""
### 🎯 핵심 로직
1.  **수량 확인 (Quantity)**: 부품의 개수 확인
2.  **연결 확인 (Connectivity)**: 전원 연결 여부 확인
3.  **구조 확인 (Topology)**: **[NEW]** 전원에 처음 연결되는 부품의 종류가 일치하는지 확인 (순서 검증)
""")

@st.cache_resource
def load_models():
    real = []
    try:
        for p in REAL_MODEL_PATHS:
            try: real.append(YOLO(p))
            except: pass
        sym = YOLO(MODEL_SYM_PATH)
    except: return [], None
    return real, sym

models_real, model_sym = load_models()

if not models_real:
    st.error("❌ 모델 로드 실패")
    st.stop()

c1, c2 = st.columns(2)
f1 = c1.file_uploader("1. 회로도", type=['jpg','png','jpeg'])
f2 = c2.file_uploader("2. 실물 사진", type=['jpg','png','jpeg'])

if f1 and f2:
    im1 = cv2.cvtColor(np.array(Image.open(f1)), cv2.COLOR_RGB2BGR)
    im2 = cv2.cvtColor(np.array(Image.open(f2)), cv2.COLOR_RGB2BGR)

    if st.button("🚀 정밀 검증 실행"):
        r_img, r_dat = analyze_schematic(im1.copy(), model_sym)
        t_img, t_dat = analyze_real_v58(im2.copy(), models_real)
        
        st.divider()
        st.subheader("📊 검증 리포트")
        
        # ---------------------------------------------------------
        # [NEW] 순서(Topology) 검증 결과 출력
        # ---------------------------------------------------------
        s_first = r_dat.get('first_component', 'unknown')
        r_first = t_dat.get('first_component', 'unknown')
        topology_match = True
        
        st.markdown(f"**🔍 입력 단(첫 부품) 분석**: 회로도=`{s_first.upper()}` vs 실물=`{r_first.upper()}`")
        
        if s_first != 'unknown' and r_first != 'unknown':
            if s_first != r_first:
                topology_match = False
                st.error(f"🚫 **구조 불일치**: 회로 연결 순서가 다릅니다!")
                st.write(f"- 회로도는 **{s_first.upper()}**가 전원에 먼저 연결되지만,")
                st.write(f"- 실물은 **{r_first.upper()}**가 전원에 먼저 연결되었습니다.")
            else:
                st.info(f"✅ **구조 일치**: 전원 입력 부품이 **{s_first.upper()}**로 동일합니다.")
        # ---------------------------------------------------------

        # 수량 비교
        keys = set(r_dat['details'].keys()) | set(t_dat['details'].keys())
        all_match = True
        
        for k in keys:
            if k in ['source', 'text']: continue
            v1 = r_dat['details'].get(k, 0)
            v2 = t_dat['details'].get(k, {}).get('count', 0)
            
            if v1 == v2:
                st.success(f"✅ {k.upper()}: 수량 일치 ({v1}개)")
            else:
                all_match = False
                st.error(f"⚠️ {k.upper()}: 수량 불일치 (회로도 {v1} vs 실물 {v2})")
                
        # 연결 상태
        if t_dat['short'] > 0:
            st.error(f"🚨 **합선 경고**: {t_dat['short']}개의 부품이 겹쳐 있습니다.")
        elif t_dat['off'] > 0:
            st.warning(f"⚠️ **연결 끊김**: {t_dat['off']}개의 부품이 전원과 연결되지 않았습니다.")
        elif all_match and topology_match: # [NEW] topology_match 조건 추가
            st.balloons()
            st.success("🎉 수량, 연결, 그리고 회로 순서까지 모두 완벽합니다!")
            
        col1, col2 = st.columns(2)
        col1.image(cv2.cvtColor(r_img, cv2.COLOR_BGR2RGB), caption=f"회로도 (입력: {s_first})", use_column_width=True)
        col2.image(cv2.cvtColor(t_img, cv2.COLOR_BGR2RGB), caption=f"실물 검증 (입력: {r_first})", use_column_width=True)
