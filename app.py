import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import math
from PIL import Image

# ==========================================
# [1. 설정 및 라이브러리]
# ==========================================
st.set_page_config(page_title="BrainBoard V17 (Universal Tuner)", layout="wide")

MODEL_REAL_PATH = 'best.pt'
MODEL_SYM_PATH = 'symbol.pt'
LEG_EXTENSION_RANGE = 180        

# ==========================================
# [2. 유틸리티 함수: 이원화된 중복 제거]
# ==========================================
def calculate_iou(box1, box2):
    x1, y1, x2, y2 = max(box1[0], box2[0]), max(box1[1], box2[1]), min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

def solve_overlap(parts, dist_thresh=0, iou_thresh=0.4, is_schematic=False):
    """
    is_schematic=True: 회로도용 (공격적 통합)
    is_schematic=False: 실물용 (정교한 분리)
    """
    if not parts: return []
    parts.sort(key=lambda x: x.get('conf', 0), reverse=True)
    
    final = []
    for curr in parts:
        is_dup = False
        for k in final:
            # 1. IoU 계산
            iou = calculate_iou(curr['box'], k['box'])
            
            # 2. 포함 관계 계산 (작은 박스가 큰 박스에 먹혔나?)
            x1 = max(curr['box'][0], k['box'][0])
            y1 = max(curr['box'][1], k['box'][1])
            x2 = min(curr['box'][2], k['box'][2])
            y2 = min(curr['box'][3], k['box'][3])
            
            inter_area = max(0, x2-x1) * max(0, y2-y1)
            area_curr = (curr['box'][2]-curr['box'][0]) * (curr['box'][3]-curr['box'][1])
            area_k = (k['box'][2]-k['box'][0]) * (k['box'][3]-k['box'][1])
            min_area = min(area_curr, area_k)

            # [모드별 분기]
            if is_schematic:
                # 회로도: 같은 부품끼리는 조금만 겹쳐도(1%) 합체 (끊긴 선 방지)
                if curr['name'] == k['name']:
                    if iou > 0.01: is_dup = True; break
                    # 거리 가까우면 합체
                    dist = math.sqrt((curr['center'][0]-k['center'][0])**2 + (curr['center'][1]-k['center'][1])**2)
                    if dist < 50: is_dup = True; break
                else:
                    # 다른 부품은 80% 이상 먹혔을 때만 제거
                    if min_area > 0 and (inter_area / min_area) > 0.8:
                        is_dup = True; break
            else:
                # 실물: 일반적인 IoU 기준 적용
                if iou > iou_thresh: is_dup = True; break
                if min_area > 0 and (inter_area / min_area) > 0.8: is_dup = True; break
                if dist_thresh > 0:
                    dist = math.sqrt((curr['center'][0]-k['center'][0])**2 + (curr['center'][1]-k['center'][1])**2)
                    if dist < dist_thresh: is_dup = True; break

        if not is_dup:
            final.append(curr)
    return final

def get_center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

# ==========================================
# [3. 회로도 분석 (범용성 강화)]
# ==========================================
def analyze_schematic(img, model, conf_thresh):
    # 사용자가 설정한 슬라이더 값(conf_thresh)을 적용
    res = model.predict(source=img, conf=conf_thresh, verbose=False)
    
    raw = []
    for b in res[0].boxes:
        cls_id = int(b.cls[0])
        raw_name = model.names[cls_id].lower()
        conf = float(b.conf[0])
        
        raw.append({
            'name': raw_name, 
            'box': b.xyxy[0].tolist(), 
            'center': get_center(b.xyxy[0].tolist()),
            'conf': conf
        })
    
    # 회로도 전용 중복 제거
    clean = solve_overlap(raw, dist_thresh=0, iou_thresh=0.1, is_schematic=True)
    
    # [수정] "무조건 왼쪽이 전원" 로직 삭제 -> AI 인식명 우선 (오류 감소)
    # 대신 전원이 하나도 없으면 가장 왼쪽 것을 전원으로 추측
    has_source = any(p['name'] in ['source', 'volt', 'batt'] for p in clean)
    leftmost_idx = -1
    
    if not has_source and clean:
        min_x = float('inf')
        for i, p in enumerate(clean):
            if p['center'][0] < min_x:
                min_x = p['center'][0]
                leftmost_idx = i

    summary_details = {}
    
    for i, p in enumerate(clean):
        raw_name = p['name']
        name = raw_name 
        
        # 이름 정규화
        if 'cap' in raw_name: name = 'capacitor'
        elif 'res' in raw_name: name = 'resistor'
        elif 'ind' in raw_name: name = 'inductor'
        elif 'dio' in raw_name: name = 'diode'
        elif any(x in raw_name for x in ['volt', 'batt', 'source']): name = 'source'

        # 전원이 없을 때만 위치 기반 추측 사용
        if i == leftmost_idx:
            name = 'source'
        
        x1, y1, x2, y2 = map(int, p['box'])
        box_color = (255, 0, 0) if name == 'source' else (0, 0, 255)
        
        cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(img, name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
        
        summary_details[name] = summary_details.get(name, 0) + 1
        
    return img, {'total': len(clean), 'details': summary_details}

# ==========================================
# [4. 실물 분석 (사용자 튜닝)]
# ==========================================
def analyze_real(img, model, conf_res, conf_cap, conf_wire):
    h, w, _ = img.shape
    
    # 기본 스캔은 낮게 시작 (내부에서 필터링)
    res = model.predict(source=img, conf=0.10, verbose=False)
    
    bodies = []
    pins = [] 
    
    for b in res[0].boxes:
        name = model.names[int(b.cls[0])].lower()
        coords = b.xyxy[0].tolist()
        center = get_center(coords)
        conf = float(b.conf[0])
        
        # [핵심] 사용자가 슬라이더로 조절한 값을 적용
        if 'cap' in name: min_conf = conf_cap
        elif 'res' in name: min_conf = conf_res
        elif 'wire' in name: min_conf = conf_wire
        else: min_conf = 0.25
            
        if conf < min_conf: continue

        if any(x in name for x in ['pin', 'leg', 'lead']) and 'wire' not in name:
            pins.append({'center': center, 'box': coords})
        elif 'breadboard' in name:
            continue
        else:
            bodies.append({'name': name, 'box': coords, 'center': center, 'conf': conf, 'is_on': False})

    # 실물 전용 중복 제거
    clean_bodies = solve_overlap(bodies, dist_thresh=60, iou_thresh=0.3, is_schematic=False)
    
    # [연결 로직]
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
            if cy < h*0.48 or cy > h*0.52: 
                comp['is_on'] = True

        for _ in range(3): 
            for comp in clean_bodies:
                if comp['is_on']: continue 
                cx, cy = comp['center']
                
                for p in pins:
                    px, py = p['center']
                    if py < h*0.48 or py > h*0.52:
                         dist = math.sqrt((cx - px)**2 + (cy - py)**2)
                         if dist < LEG_EXTENSION_RANGE:
                             comp['is_on'] = True; break

                if comp['is_on']: continue

                for other in clean_bodies:
                    if not other['is_on']: continue
                    ocx, ocy = other['center']
                    dist = math.sqrt((cx - ocx)**2 + (cy - ocy)**2)
                    if dist < LEG_EXTENSION_RANGE * 1.5:
                        comp['is_on'] = True; break

    off_count = 0
    real_details = {} 
    
    for comp in clean_bodies:
        is_on = comp['is_on']
        raw_name = comp['name']
        
        norm_name = raw_name
        label_name = "" 
        
        if 'res' in raw_name: 
            norm_name = 'resistor'; label_name = "RES"
        elif 'cap' in raw_name: 
            norm_name = 'capacitor'; label_name = "CAP"
        elif 'wire' in raw_name:
            label_name = "WIRE"
        else:
            label_name = raw_name[:3].upper()
        
        if 'wire' not in raw_name:
            real_details[norm_name] = real_details.get(norm_name, 0) + 1

        if is_on:
            color = (0, 255, 0)
            status = "ON"
        else:
            color = (0, 0, 255)
            status = "OFF"
            off_count += 1
        
        display_text = f"{label_name}: {status}"
        x1, y1, x2, y2 = map(int, comp['box'])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.putText(img, display_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
    return img, {'off': off_count, 'total': len(clean_bodies), 'details': real_details}

# ==========================================
# [5. 메인 UI (튜너 추가)]
# ==========================================
st.title("🧠 BrainBoard V17: Universal Tuner")
st.markdown("### 민감도를 직접 조절하여 모든 회로에 대응하세요.")

@st.cache_resource
def load_models():
    return YOLO(MODEL_REAL_PATH), YOLO(MODEL_SYM_PATH)

try:
    model_real, model_sym = load_models()
    st.sidebar.success("✅ 모델 로드 성공")
except Exception as e:
    st.error(f"모델 로드 실패: {e}")
    st.stop()

# ------------------------------------------------------------------
# [SIDEBAR] 민감도 조절 슬라이더 (사용자가 직접 튜닝!)
# ------------------------------------------------------------------
st.sidebar.header("🎛️ 분석 민감도 설정")
st.sidebar.info("부품이 안 잡히면 낮추고, 엉뚱한게 잡히면 높이세요.")

# 회로도 설정
st.sidebar.markdown("---")
st.sidebar.subheader("📄 회로도 설정")
conf_sym = st.sidebar.slider("회로도 인식 민감도", 0.0, 1.0, 0.20, 0.05)

# 실물 설정
st.sidebar.markdown("---")
st.sidebar.subheader("📸 실물 설정")
conf_res = st.sidebar.slider("저항(Resistor) 민감도", 0.0, 1.0, 0.40, 0.05)
conf_cap = st.sidebar.slider("커패시터(Capacitor) 민감도", 0.0, 1.0, 0.20, 0.05)
conf_wire = st.sidebar.slider("와이어(Wire) 민감도", 0.0, 1.0, 0.15, 0.05)

col1, col2 = st.columns(2)
ref_file = col1.file_uploader("1. 회로도 업로드", type=['jpg', 'png', 'jpeg'])
tgt_file = col2.file_uploader("2. 실물 사진 업로드", type=['jpg', 'png', 'jpeg'])

if ref_file and tgt_file:
    ref_image = Image.open(ref_file)
    tgt_image = Image.open(tgt_file)
    
    ref_cv = cv2.cvtColor(np.array(ref_image), cv2.COLOR_RGB2BGR)
    tgt_cv = cv2.cvtColor(np.array(tgt_image), cv2.COLOR_RGB2BGR)

    # 버튼 누를 때 슬라이더 값을 함수로 전달
    if st.button("🚀 정밀 분석 실행"):
        with st.spinner("사용자 설정 적용 중..."):
            res_ref_img, ref_data = analyze_schematic(ref_cv.copy(), model_sym, conf_sym)
            res_tgt_img, tgt_data = analyze_real(tgt_cv.copy(), model_real, conf_res, conf_cap, conf_wire)

            st.divider()
            
            st.info("📊 **부품 인식 현황**")
            
            r_ref = ref_data['details'].get('resistor', 0)
            r_tgt = tgt_data['details'].get('resistor', 0)
            st.write(f"- **저항 (Resistor):** 회로도 {r_ref}개 vs 실물 {r_tgt}개")
            
            c_ref = ref_data['details'].get('capacitor', 0)
            c_tgt = tgt_data['details'].get('capacitor', 0)
            st.write(f"- **커패시터 (Capacitor):** 회로도 {c_ref}개 vs 실물 {c_tgt}개")

            st.divider()

            mismatch_errors = []
            if r_ref != r_tgt:
                mismatch_errors.append(f"⚠️ RESISTOR 불일치: 회로도 {r_ref}개 vs 실물 {r_tgt}개")
            if c_ref != c_tgt:
                mismatch_errors.append(f"⚠️ CAPACITOR 불일치: 회로도 {c_ref}개 vs 실물 {c_tgt}개")
            
            st.image(cv2.cvtColor(res_ref_img, cv2.COLOR_BGR2RGB), caption="회로도 분석 결과", use_column_width=True)
            st.image(cv2.cvtColor(res_tgt_img, cv2.COLOR_BGR2RGB), caption=f"실물 분석 (OFF: {tgt_data['off']})", use_column_width=True)
            
            if mismatch_errors:
                st.error("❌ 회로 구성이 다릅니다 (부품 개수 불일치)")
                for err in mismatch_errors:
                    st.write(err)
            elif tgt_data['off'] > 0:
                st.error(f"❌ 부품 연결이 끊어졌습니다 ({tgt_data['off']}개 OFF)")
            else:
                st.success("✅ 완벽합니다! (부품 일치 & 전원 연결 성공)")
