import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import math
from PIL import Image

# ==========================================
# [1. 설정 및 라이브러리]
# ==========================================
st.set_page_config(page_title="BrainBoard V35 (Topology Check)", layout="wide")

MODEL_REAL_PATH = 'best.pt'
MODEL_SYM_PATH = 'symbol.pt'
LEG_EXTENSION_RANGE = 180         

# ==========================================
# [2. 유틸리티 함수]
# ==========================================
def calculate_iou(box1, box2):
    x1, y1, x2, y2 = max(box1[0], box2[0]), max(box1[1], box2[1]), min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

def solve_overlap(parts, dist_thresh=0, iou_thresh=0.4, is_schematic=False):
    if not parts: return []
    if is_schematic:
        parts.sort(key=lambda x: (x['box'][2]-x['box'][0]) * (x['box'][3]-x['box'][1]))
    else:
        parts.sort(key=lambda x: x.get('conf', 0), reverse=True)
    
    final = []
    for curr in parts:
        is_dup = False
        for k in final:
            x1 = max(curr['box'][0], k['box'][0])
            y1 = max(curr['box'][1], k['box'][1])
            x2 = min(curr['box'][2], k['box'][2])
            y2 = min(curr['box'][3], k['box'][3])
            inter_area = max(0, x2-x1) * max(0, y2-y1)
            
            if is_schematic:
                if inter_area > 0: is_dup = True; break
                dist = math.sqrt((curr['center'][0]-k['center'][0])**2 + (curr['center'][1]-k['center'][1])**2)
                if dist < 80: is_dup = True; break
            else:
                area_curr = (curr['box'][2]-curr['box'][0]) * (curr['box'][3]-curr['box'][1])
                area_k = (k['box'][2]-k['box'][0]) * (k['box'][3]-k['box'][1])
                min_area = min(area_curr, area_k)
                ratio = inter_area / min_area if min_area > 0 else 0
                iou = calculate_iou(curr['box'], k['box'])
                if ratio > 0.8: is_dup = True; break
                if iou > iou_thresh: is_dup = True; break
                if dist_thresh > 0:
                    dist = math.sqrt((curr['center'][0]-k['center'][0])**2 + (curr['center'][1]-k['center'][1])**2)
                    if dist < dist_thresh: is_dup = True; break
        if not is_dup: final.append(curr)
    return final

def get_center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

# ==========================================
# [3. 회로도 분석 (토폴로지 로직 추가)]
# ==========================================
def analyze_schematic(img, model):
    res = model.predict(source=img, conf=0.15, verbose=False)
    raw = []
    for b in res[0].boxes:
        cls_id = int(b.cls[0])
        raw_name = model.names[cls_id].lower()
        conf = float(b.conf[0])
        
        name = raw_name
        if raw_name == 'v': name = 'source'
        elif any(x in raw_name for x in ['volt', 'batt', 'source']): name = 'source'
        elif 'cap' in raw_name: name = 'capacitor'
        elif 'res' in raw_name: name = 'resistor'
        elif 'ind' in raw_name: name = 'inductor'
        elif 'dio' in raw_name: name = 'diode'
        
        raw.append({'name': name, 'box': b.xyxy[0].tolist(), 'center': get_center(b.xyxy[0].tolist()), 'conf': conf})
    
    clean = solve_overlap(raw, dist_thresh=0, iou_thresh=0.1, is_schematic=True)
    
    # 전원 보정
    has_source = any(p['name'] == 'source' for p in clean)
    if not has_source and clean:
        min(clean, key=lambda p: p['center'][0])['name'] = 'source'

    # -----------------------------------------------------------
    # [NEW] 시작 부품 찾기 (Source와 가장 가까운 부품)
    # -----------------------------------------------------------
    source_part = next((p for p in clean if p['name'] == 'source'), None)
    first_part_name = None
    
    if source_part:
        min_dist = float('inf')
        for p in clean:
            if p['name'] == 'source': continue
            # 소스 중심과 부품 중심 거리 계산
            d = math.sqrt((source_part['center'][0]-p['center'][0])**2 + (source_part['center'][1]-p['center'][1])**2)
            if d < min_dist:
                min_dist = d
                first_part_name = p['name']
    
    summary_details = {}
    for p in clean:
        name = p['name']
        x1, y1, x2, y2 = map(int, p['box'])
        if name == 'source':
            box_color = (255, 0, 0)
            disp_name = "V"
        else:
            box_color = (0, 0, 255)
            disp_name = name
            
        cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(img, disp_name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
        summary_details[name] = summary_details.get(name, 0) + 1
        
    return img, {'total': len(clean), 'details': summary_details, 'first_conn': first_part_name}

# ==========================================
# [4. 실물 분석 (토폴로지 로직 추가)]
# ==========================================
def analyze_real(img, model):
    h, w, _ = img.shape
    res = model.predict(source=img, conf=0.10, verbose=False)
    
    bodies = []
    pins = [] 
    
    for b in res[0].boxes:
        name = model.names[int(b.cls[0])].lower()
        coords = b.xyxy[0].tolist()
        center = get_center(coords)
        conf = float(b.conf[0])
        
        if 'cap' in name: min_conf = 0.15
        elif 'res' in name: min_conf = 0.60
        elif 'wire' in name: min_conf = 0.15
        else: min_conf = 0.25
            
        if conf < min_conf: continue

        if any(x in name for x in ['pin', 'leg', 'lead']) and 'wire' not in name:
            pins.append({'center': center, 'box': coords})
        elif 'breadboard' in name:
            continue
        else:
            bodies.append({'name': name, 'box': coords, 'center': center, 'conf': conf, 'is_on': False})

    clean_bodies = solve_overlap(bodies, dist_thresh=60, iou_thresh=0.3, is_schematic=False)
    
    # -----------------------------------------------------------
    # [NEW] 실물 시작 부품 찾기 (전원선과 직접 연결된 부품)
    # -----------------------------------------------------------
    power_rail_top = h * 0.2
    power_rail_bot = h * 0.8
    direct_power_components = set()

    # 1. 전원 와이어 찾기
    power_wires = []
    for b in clean_bodies:
        if 'wire' in b['name']:
            if b['center'][1] < power_rail_top or b['center'][1] > power_rail_bot:
                power_wires.append(b)
    
    # 2. 전원 와이어와 가까운 부품 찾기
    for comp in clean_bodies:
        if 'wire' in comp['name']: continue
        
        # 전원 와이어와 가까우면 전원 직결 부품으로 간주
        for wire in power_wires:
            dist = math.sqrt((comp['center'][0]-wire['center'][0])**2 + (comp['center'][1]-wire['center'][1])**2)
            if dist < 180: # 와이어 길이 고려한 거리
                # 정규화된 이름으로 저장
                n_name = comp['name']
                if 'res' in n_name: n_name = 'resistor'
                elif 'cap' in n_name: n_name = 'capacitor'
                direct_power_components.add(n_name)

    # [기존 연결 로직]
    power_active = False
    for b in clean_bodies:
        if 'wire' in b['name'] and b['center'][1] < h * 0.45:
            power_active = True; break
    if not power_active:
        for p in pins:
            if p['center'][1] < h * 0.45:
                power_active = True; break

    if power_active:
        for comp in clean_bodies:
            cy = comp['center'][1]
            if cy < h*0.48 or cy > h*0.52: comp['is_on'] = True

        for _ in range(3): 
            for comp in clean_bodies:
                if comp['is_on']: continue 
                cx, cy = comp['center']
                for p in pins:
                    px, py = p['center']
                    if py < h*0.48 or py > h*0.52:
                         dist = math.sqrt((cx - px)**2 + (cy - py)**2)
                         if dist < LEG_EXTENSION_RANGE: comp['is_on'] = True; break
                if comp['is_on']: continue
                for other in clean_bodies:
                    if not other['is_on']: continue
                    ocx, ocy = other['center']
                    dist = math.sqrt((cx - ocx)**2 + (cy - ocy)**2)
                    if dist < LEG_EXTENSION_RANGE * 1.5: comp['is_on'] = True; break

    off_count = 0
    real_details = {} 
    
    for comp in clean_bodies:
        is_on = comp['is_on']
        raw_name = comp['name']
        norm_name = raw_name
        label_name = "" 
        
        if 'res' in raw_name: norm_name = 'resistor'; label_name = "RES"
        elif 'cap' in raw_name: norm_name = 'capacitor'; label_name = "CAP"
        elif 'wire' in raw_name: label_name = "WIRE"
        else: label_name = raw_name[:3].upper()
        
        if 'wire' not in raw_name:
            real_details[norm_name] = real_details.get(norm_name, 0) + 1

        if is_on:
            color = (0, 255, 0)
            status = "ON"
        else:
            color = (0, 0, 255)
            status = "OFF"
            off_count += 1
        
        x1, y1, x2, y2 = map(int, comp['box'])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.putText(img, f"{label_name}: {status}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
    return img, {'off': off_count, 'total': len(clean_bodies), 'details': real_details, 'direct_conns': list(direct_power_components)}

# ==========================================
# [5. 메인 UI]
# ==========================================
st.title("🧠 BrainBoard V35: Topology Check")
st.markdown("### 회로 구성(순서) 불일치 감지 기능 추가")

@st.cache_resource
def load_models():
    return YOLO(MODEL_REAL_PATH), YOLO(MODEL_SYM_PATH)

try:
    model_real, model_sym = load_models()
except Exception as e:
    st.error(f"모델 로드 실패: {e}")
    st.stop()

col1, col2 = st.columns(2)
ref_file = col1.file_uploader("1. 회로도", type=['jpg', 'png', 'jpeg'])
tgt_file = col2.file_uploader("2. 실물 사진", type=['jpg', 'png', 'jpeg'])

if ref_file and tgt_file:
    ref_image = Image.open(ref_file)
    tgt_image = Image.open(tgt_file)
    ref_cv = cv2.cvtColor(np.array(ref_image), cv2.COLOR_RGB2BGR)
    tgt_cv = cv2.cvtColor(np.array(tgt_image), cv2.COLOR_RGB2BGR)

    if st.button("🚀 정밀 분석 실행"):
        with st.spinner("AI 분석 중..."):
            res_ref_img, ref_data = analyze_schematic(ref_cv.copy(), model_sym)
            res_tgt_img, tgt_data = analyze_real(tgt_cv.copy(), model_real)

            st.divider()
            
            # 1. 개수 비교
            st.info("📊 **부품 인식 현황**")
            r_ref = ref_data['details'].get('resistor', 0)
            r_tgt = tgt_data['details'].get('resistor', 0)
            c_ref = ref_data['details'].get('capacitor', 0)
            c_tgt = tgt_data['details'].get('capacitor', 0)
            
            st.write(f"- 저항: 회로도 {r_ref} vs 실물 {r_tgt}")
            st.write(f"- 커패시터: 회로도 {c_ref} vs 실물 {c_tgt}")

            # 2. [NEW] 토폴로지(연결 구조) 비교
            st.subheader("🔗 연결 구조 진단")
            
            expected_start = ref_data.get('first_conn') # 회로도에서 전원 바로 옆 부품
            actual_starts = tgt_data.get('direct_conns', []) # 실물에서 전원선에 연결된 부품들

            topology_error = False
            if expected_start:
                # 회로도 시작 부품이 실물 전원 연결 목록에 없는 경우
                # (예: 회로도는 저항 시작인데, 실물은 커패시터가 전원에 붙어있음)
                if expected_start not in actual_starts and actual_starts:
                    # 엄격한 검사: 실물 전원 연결 부품 중 예상된 부품이 하나도 없을 때
                    if expected_start == 'resistor' and 'capacitor' in actual_starts:
                        st.error(f"🚨 **치명적 오류**: 회로 순서가 다릅니다!")
                        st.write(f"👉 회로도: 전원이 **저항(Resistor)**으로 먼저 들어갑니다.")
                        st.write(f"👉 실물: 전원이 **커패시터(Capacitor)**에 직접 연결되었습니다.")
                        topology_error = True
                    elif expected_start == 'capacitor' and 'resistor' in actual_starts:
                        st.error(f"🚨 **치명적 오류**: 회로 순서가 다릅니다! (예상: CAP, 실물: RES)")
                        topology_error = True

            if not topology_error:
                st.success("✅ 회로 연결 순서가 논리적으로 일치합니다.")

            st.divider()
            
            mismatch = []
            if r_ref != r_tgt: mismatch.append("저항 개수 불일치")
            if c_ref != c_tgt: mismatch.append("커패시터 개수 불일치")
            
            col_res1, col_res2 = st.columns(2)
            col_res1.image(cv2.cvtColor(res_ref_img, cv2.COLOR_BGR2RGB), caption=f"회로도 (시작: {expected_start})", use_column_width=True)
            col_res2.image(cv2.cvtColor(res_tgt_img, cv2.COLOR_BGR2RGB), caption=f"실물 (전원직결: {actual_starts})", use_column_width=True)

            if mismatch:
                st.error(f"❌ 개수 불일치: {', '.join(mismatch)}")
            elif not topology_error and tgt_data['off'] == 0:
                st.balloons()
                st.success("🎉 완벽합니다! (개수 & 구조 일치)")
