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

    summary = {'details': {}}
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

            # [Rule] 민감도 설정 (저항은 0.20까지 낮춤 - 사용자 요청)
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
    # 핀은 중복 제거를 하지 않거나 약하게 하여 연결 포인트 확보
    
    # 3. [Connectivity Logic] 핀 기반 연결 확인
    # 전원 레일 정의 (상하단 15%)
    power_top = h * 0.15
    power_bot = h * 0.85 # 하단 15% 지점 (좌표상 h*0.85 이상)
    # *참고: 브레드보드가 꽉 차게 찍히면 상단/하단 끝부분이 전원부임. 
    # V35 로직(중앙 기준 위아래)을 보완하여 Y좌표 절대값으로 1차 필터링
    
    # (Step A) 1차 활성화: 전원부에 직접 닿은 핀 & 와이어 찾기
    for p in raw_pins:
        py = p['center'][1]
        # 상단 전원부 or 하단 전원부
        if py < h * 0.25 or py > h * 0.75: 
            p['is_active'] = True

    for b in clean_bodies:
        # 와이어나 부품 자체가 전원부에 걸쳐있는 경우 (Box 기준)
        y1, y2 = b['box'][1], b['box'][3]
        if y1 < h * 0.25 or y2 > h * 0.75:
            b['is_on'] = True

    # (Step B) 전파 (Propagation): Active Pin <-> Body <-> Active Pin
    # 3번 반복하여 연결을 확산시킵니다.
    for _ in range(3):
        # 1. 핀 -> 바디 (핀이 활성화되면, 그 핀과 가까운 바디도 켜짐)
        for b in clean_bodies:
            if b['is_on']: continue
            for p in raw_pins:
                if p['is_active']:
                    if dist(b['center'], p['center']) < LEG_EXTENSION_RANGE:
                        b['is_on'] = True
                        break
        
        # 2. 바디 -> 핀 (바디가 켜지면, 그 바디와 가까운 핀들도 활성화됨 - 릴레이)
        for b in clean_bodies:
            if b['is_on']:
                for p in raw_pins:
                    if not p['is_active']:
                        if dist(b['center'], p['center']) < LEG_EXTENSION_RANGE:
                            p['is_active'] = True

        # 3. 바디 -> 바디 (와이어 등을 통한 직접 연결)
        for b1 in clean_bodies:
            if b1['is_on']:
                for b2 in clean_bodies:
                    if not b2['is_on']:
                        # 와이어는 연결 범위가 더 넓음
                        limit = LEG_EXTENSION_RANGE * 1.5 if 'wire' in b1['name'] else LEG_EXTENSION_RANGE
                        if dist(b1['center'], b2['center']) < limit:
                            b2['is_on'] = True

    # 4. [Safety Logic] 쇼트 감지 (바디끼리 겹침)
    for i, c1 in enumerate(clean_bodies):
        if 'wire' in c1['name']: continue
        for j, c2 in enumerate(clean_bodies):
            if i >= j or 'wire' in c2['name']: continue
            if calculate_iou(c1['box'], c2['box']) > SHORT_CIRCUIT_IOU:
                c1['is_short'] = True
                c2['is_short'] = True

    # 5. 결과 집계 및 그리기
    summary = {'total': 0, 'on': 0, 'off': 0, 'short': 0, 'details': {}}
    
    for comp in clean_bodies:
        # [Count Logic] 수량은 오직 Body(Res, Cap)만 셉니다. Wire 제외.
        raw_name = comp['name']
        norm_name = raw_name
        label = raw_name[:3].upper()
        
        if 'res' in raw_name: norm_name = 'resistor'; label="RES"
        elif 'cap' in raw_name: norm_name = 'capacitor'; label="CAP"
        elif 'wire' in raw_name: label="WIRE"
        
        # 카운팅 (와이어 제외)
        if 'wire' not in raw_name:
            if norm_name not in summary['details']: summary['details'][norm_name] = {'count': 0}
            summary['details'][norm_name]['count'] += 1

        # 시각화 상태 결정
        if comp['is_short']:
            color = (0, 0, 255) # Red
            text = "SHORT!"
            summary['short'] += 1
            summary['off'] += 1
        elif comp['is_on']:
            color = (0, 255, 0) # Green (무조건 초록)
            text = "ON"
            summary['on'] += 1
        else:
            color = (0, 0, 255) # Red
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
st.title("🧠 BrainBoard V58: Body-Count & Pin-Connect")
st.markdown("""
### 🎯 핵심 로직
1.  **수량 확인 (Quantity)**: 오직 부품의 **몸체(Body)** 개수만 카운트합니다.
2.  **연결 확인 (Connection)**: 부품의 몸체가 아닌, 주변의 **핀(Pin)**이 전원에 연결되었는지를 우선 확인합니다.
3.  **인식 개선**: 저항 인식 민감도를 대폭 완화(20%)하여 작은 저항도 놓치지 않습니다.
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
            st.warning(f"⚠️ **연결 끊김**: {t_dat['off']}개의 부품이 전원과 연결되지 않았습니다. (핀 연결 확인 필요)")
        elif all_match:
            st.balloons()
            st.success("🎉 수량과 연결 상태가 모두 완벽합니다!")
            
        col1, col2 = st.columns(2)
        col1.image(cv2.cvtColor(r_img, cv2.COLOR_BGR2RGB), caption="회로도", use_column_width=True)
        col2.image(cv2.cvtColor(t_img, cv2.COLOR_BGR2RGB), caption="실물 검증 (Pin-Logic)", use_column_width=True)
