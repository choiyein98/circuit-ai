import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import math
from PIL import Image

# ==========================================
# [1. 설정 및 라이브러리]
# ==========================================
st.set_page_config(page_title="BrainBoard V31 (Shell & Noise Fix)", layout="wide")

MODEL_REAL_PATH = 'best.pt'
MODEL_SYM_PATH = 'symbol.pt'

# 연결 감지 범위
LEG_EXTENSION_RANGE = 180        

# ==========================================
# [2. 유틸리티 함수: 껍데기 박멸 & 중복 제거]
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
    # [정렬 전략] 회로도는 무조건 '작은 것' 우선
    # -----------------------------------------------------------
    if is_schematic:
        # 면적이 작은 순서대로 정렬 (진짜 심볼은 작고, 텍스트 박스는 크기 때문)
        parts.sort(key=lambda x: (x['box'][2]-x['box'][0]) * (x['box'][3]-x['box'][1]))
    else:
        # 실물은 신뢰도 높은 순
        parts.sort(key=lambda x: x.get('conf', 0), reverse=True)
    
    final = []
    for curr in parts:
        # [회로도 전용 필터] 가로로 너무 긴 박스(글자) 제거
        if is_schematic:
            w = curr['box'][2] - curr['box'][0]
            h = curr['box'][3] - curr['box'][1]
            if h > 0 and (w / h) > 3.0: # 가로가 세로보다 3배 이상 길면 글자로 간주
                continue 

        is_dup = False
        for k in final:
            # 좌표 및 면적 계산
            x1 = max(curr['box'][0], k['box'][0])
            y1 = max(curr['box'][1], k['box'][1])
            x2 = min(curr['box'][2], k['box'][2])
            y2 = min(curr['box'][3], k['box'][3])
            
            inter_area = max(0, x2-x1) * max(0, y2-y1)
            
            # -----------------------------------------------------------
            # [MODE A] 회로도 전용 (작은 놈이 짱이다)
            # -----------------------------------------------------------
            if is_schematic:
                # k: 이미 살아남은 '작은 박스' (진짜)
                # curr: 지금 검사하는 '큰 박스' (껍데기 후보)
                
                # [조건 1] 겹침 발생 시 (조금이라도 닿으면 삭제)
                if inter_area > 0:
                    # 작은 박스(k)가 이미 있는데, 큰 박스(curr)가 그 위를 덮거나 닿았다?
                    # -> curr는 껍데기입니다. 삭제.
                    is_dup = True; break
                
                # [조건 2] 거리 기반 삭제 (텍스트 박스 제거)
                # 겹치지 않아도 중심점이 100px 이내면 중복(설명 텍스트)으로 간주
                dist = math.sqrt((curr['center'][0]-k['center'][0])**2 + (curr['center'][1]-k['center'][1])**2)
                if dist < 100:
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
def analyze_schematic(img, model):
    # [핵심] 0.01로 아주 낮게 설정해서 일단 희미한 커패시터도 다 잡습니다.
    # 그 후 solve_overlap에서 가짜(큰 박스)를 걸러냅니다.
    res = model.predict(source=img, conf=0.01, verbose=False)
    
    raw = []
    for b in res[0].boxes:
        cls_id = int(b.cls[0])
        raw_name = model.names[cls_id].lower()
        conf = float(b.conf[0])
        
        # 이름 매핑 ('v' -> 'source')
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
    
    # [핵심] 껍데기 박멸 로직 실행
    clean = solve_overlap(raw, dist_thresh=0, iou_thresh=0.1, is_schematic=True)
    
    # 전원 위치 보정 (전원이 없을 때만)
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
        
        # 색상 설정 (V=파랑, 나머지=빨강)
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
# [4. 실물 분석 (V15 설정 유지)]
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
        
        # [V15 민감도]
        if 'cap' in name: min_conf = 0.15      # 커패시터: 15% 
        elif 'res' in name: min_conf = 0.60    # 저항: 60%
        elif 'wire' in name: min_conf = 0.15   # 와이어: 15%
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
# [5. 메인 UI]
# ==========================================
st.title("🧠 BrainBoard V31 (Shell & Noise Fix)")
st.markdown("### 1. 부품 일치 여부")
st.markdown("### 2. 연결 상태")

@st.cache_resource
def load_models():
    return YOLO(MODEL_REAL_PATH), YOLO(MODEL_SYM_PATH)

try:
    model_real, model_sym = load_models()
    st.sidebar.success("✅ 모델 로드 성공")
except Exception as e:
    st.error(f"모델 로드 실패: {e}")
    st.stop()

col1, col2 = st.columns(2)
ref_file = col1.file_uploader("1. 회로도 업로드", type=['jpg', 'png', 'jpeg'])
tgt_file = col2.file_uploader("2. 실물 사진 업로드", type=['jpg', 'png', 'jpeg'])

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
            
            st.image(cv2.cvtColor(res_ref_img, cv2.COLOR_BGR2RGB), caption="회로도 분석 (껍데기 제거 + V 인식)", use_column_width=True)
            st.image(cv2.cvtColor(res_tgt_img, cv2.COLOR_BGR2RGB), caption=f"실물 분석 (OFF: {tgt_data['off']})", use_column_width=True)
            
            if mismatch_errors:
                st.error("❌ 회로 구성이 다릅니다 (부품 개수 불일치)")
                for err in mismatch_errors:
                    st.write(err)
            elif tgt_data['off'] > 0:
                st.error(f"❌ 부품 연결이 끊어졌습니다 ({tgt_data['off']}개 OFF)")
            else:
                st.success("✅ 완벽합니다! (부품 일치 & 전원 연결 성공)")
