import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import math
from PIL import Image

# ==========================================
# [1. 설정 및 라이브러리]
# ==========================================
st.set_page_config(page_title="BrainBoard V34 (User-Trained Trust)", layout="wide")

MODEL_REAL_PATH = 'best.pt'
MODEL_SYM_PATH = 'symbol.pt'
LEG_EXTENSION_RANGE = 180        

# ==========================================
# [2. 유틸리티 함수: 순수 중복 제거]
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
    
    # -----------------------------------------------------------
    # [정렬 전략]
    # -----------------------------------------------------------
    if is_schematic:
        # 회로도: 면적이 작은 순서대로 정렬 (작은 알맹이를 우선시)
        parts.sort(key=lambda x: (x['box'][2]-x['box'][0]) * (x['box'][3]-x['box'][1]))
    else:
        # 실물: 신뢰도 순
        parts.sort(key=lambda x: x.get('conf', 0), reverse=True)
    
    final = []
    for curr in parts:
        # [수정] 사용자가 학습시킨 모델을 믿고, 기하학적 필터(모양 보고 삭제)를 모두 제거했습니다.
        
        is_dup = False
        for k in final:
            # 좌표 및 면적 계산
            x1 = max(curr['box'][0], k['box'][0])
            y1 = max(curr['box'][1], k['box'][1])
            x2 = min(curr['box'][2], k['box'][2])
            y2 = min(curr['box'][3], k['box'][3])
            
            inter_area = max(0, x2-x1) * max(0, y2-y1)
            
            # -----------------------------------------------------------
            # [MODE A] 회로도 전용 (순수 겹침 제거)
            # -----------------------------------------------------------
            if is_schematic:
                # k: 이미 등록된 작은 박스
                # curr: 지금 들어온 큰 박스
                
                # 1. 겹침 발생 시 (완전히 겹치거나 포함될 때만)
                # 모델이 직접 찾은거니까, '살짝 닿는 것' 정도는 허용할 수도 있음.
                # 하지만 '확실히 겹친다'면 큰 박스(curr)를 껍데기로 보고 제외.
                if inter_area > 0:
                    area_k = (k['box'][2]-k['box'][0]) * (k['box'][3]-k['box'][1])
                    overlap_ratio = inter_area / area_k # 작은 박스 기준 겹침 비율
                    
                    # 작은 박스가 50% 이상 덮이면 중복으로 간주
                    if overlap_ratio > 0.5:
                        is_dup = True; break
                
                # 2. 거리 기반 (완전히 똑같은 위치에 여러 개 잡히는 것 방지)
                dist = math.sqrt((curr['center'][0]-k['center'][0])**2 + (curr['center'][1]-k['center'][1])**2)
                if dist < 50: # 아주 가까우면 중복
                    is_dup = True; break

            # -----------------------------------------------------------
            # [MODE B] 실물 전용 (V15 로직 유지)
            # -----------------------------------------------------------
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

        if not is_dup:
            final.append(curr)
            
    return final

def get_center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

# ==========================================
# [3. 회로도 분석]
# ==========================================
def analyze_schematic(img, model, conf_thresh):
    # 슬라이더 값 적용
    res = model.predict(source=img, conf=conf_thresh, verbose=False)
    
    raw = []
    for b in res[0].boxes:
        cls_id = int(b.cls[0])
        raw_name = model.names[cls_id].lower()
        conf = float(b.conf[0])
        
        # [이름 매핑] 'v' -> 'source' 변환은 유지 (이건 필요하다고 하셨으니까요)
        name = raw_name
        if raw_name == 'v': 
            name = 'source'
        elif any(x in raw_name for x in ['volt', 'batt', 'source']):
            name = 'source'
        elif 'cap' in raw_name: name = 'capacitor'
        elif 'res' in raw_name: name = 'resistor'
        elif 'ind' in raw_name: name = 'inductor'
        elif 'dio' in raw_name: name = 'diode'
        
        raw.append({
            'name': name,
            'box': b.xyxy[0].tolist(), 
            'center': get_center(b.xyxy[0].tolist()),
            'conf': conf
        })
    
    # 중복 제거 (필터 없이 순수 중복만 체크)
    clean = solve_overlap(raw, dist_thresh=0, iou_thresh=0.1, is_schematic=True)
    
    # 전원 위치 보정
    leftmost_idx = -1
    min_x = float('inf')
    has_source = any(p['name'] == 'source' for p in clean)
    if not has_source and clean:
        for i, p in enumerate(clean):
            if p['center'][0] < min_x:
                min_x = p['center'][0]
                leftmost_idx = i

    summary_details = {}
    
    for i, p in enumerate(clean):
        name = p['name']
        
        if i == leftmost_idx:
            name = 'source'
        
        x1, y1, x2, y2 = map(int, p['box'])
        
        if name == 'source':
            box_color = (255, 0, 0) # Blue
            disp_name = "V"
        else:
            box_color = (0, 0, 255) # Red
            disp_name = name
            
        cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(img, disp_name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
        
        summary_details[name] = summary_details.get(name, 0) + 1
        
    return img, {'total': len(clean), 'details': summary_details}

# ==========================================
# [4. 실물 분석 (사용자 튜닝)]
# ==========================================
def analyze_real(img, model, conf_res, conf_cap, conf_wire):
    h, w, _ = img.shape
    
    res = model.predict(source=img, conf=0.10, verbose=False)
    
    bodies = []
    pins = [] 
    
    for b in res[0].boxes:
        name = model.names[int(b.cls[0])].lower()
        coords = b.xyxy[0].tolist()
        center = get_center(coords)
        conf = float(b.conf[0])
        
        # 슬라이더 값 적용
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

    # 실물 중복 제거
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
st.title("🧠 BrainBoard V34: User-Trained Trust")
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
# [SIDEBAR] 민감도 조절 슬라이더
# ------------------------------------------------------------------
st.sidebar.header("🎛️ 분석 민감도 설정")

# 회로도 설정
st.sidebar.markdown("---")
st.sidebar.subheader("📄 회로도 설정")
# 사용자가 직접 학습시켰으므로, 기본값을 적당히(0.15) 두어 노이즈만 거릅니다.
conf_sym = st.sidebar.slider("회로도 인식 민감도", 0.0, 1.0, 0.15, 0.05) 

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
