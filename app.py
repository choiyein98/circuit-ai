import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import math
from PIL import Image

# ==========================================
# [설정] BrainBoard V50: Hybrid (V48 Schem + V35 Real)
# ==========================================
st.set_page_config(page_title="BrainBoard V50", layout="wide")

MODEL_REAL_PATH = 'best(3).pt'  # 실물 모델
MODEL_SYM_PATH = 'symbol.pt'    # 회로도 모델
LEG_EXTENSION_RANGE = 180

# ==========================================
# [Helper Functions] 공통 함수
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

# ==========================================
# [중복 제거 1] V48 스타일 (회로도용)
# ==========================================
def solve_overlap_v48(parts, distance_threshold=80):
    if not parts: return []
    # 신뢰도순 정렬
    parts.sort(key=lambda x: x['conf'], reverse=True)
    
    final_parts = []
    for current in parts:
        is_duplicate = False
        for kept in final_parts:
            iou = calculate_iou(current['box'], kept['box'])
            cx1, cy1 = current['center']
            cx2, cy2 = kept['center']
            dist = math.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)
            
            # [V48 로직] IoU 0.1 이상이거나 거리 80 미만이면 중복 처리
            if iou > 0.1 or dist < distance_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            final_parts.append(current)
    return final_parts

# ==========================================
# [중복 제거 2] V35/V15 스타일 (실물용)
# ==========================================
def solve_overlap_v35(parts, dist_thresh=60, iou_thresh=0.4):
    if not parts: return []
    # 신뢰도순 정렬
    parts.sort(key=lambda x: x.get('conf', 0), reverse=True)
    
    final = []
    for curr in parts:
        is_dup = False
        for k in final:
            # 좌표 및 면적 계산
            x1 = max(curr['box'][0], k['box'][0])
            y1 = max(curr['box'][1], k['box'][1])
            x2 = min(curr['box'][2], k['box'][2])
            y2 = min(curr['box'][3], k['box'][3])
            
            inter_area = max(0, x2-x1) * max(0, y2-y1)
            area_curr = (curr['box'][2]-curr['box'][0]) * (curr['box'][3]-curr['box'][1])
            area_k = (k['box'][2]-k['box'][0]) * (k['box'][3]-k['box'][1])
            min_area = min(area_curr, area_k)
            
            ratio = inter_area / min_area if min_area > 0 else 0
            iou = calculate_iou(curr['box'], k['box'])
            
            # [V35 로직] 포함관계 0.8 or IoU 0.4 or 거리 60
            if ratio > 0.8: is_dup = True; break
            if iou > iou_thresh: is_dup = True; break
            
            dist = math.sqrt((curr['center'][0]-k['center'][0])**2 + (curr['center'][1]-k['center'][1])**2)
            if dist < dist_thresh: is_dup = True; break

        if not is_dup:
            final.append(curr)
            
    return final

# ==========================================
# [분석 1] 회로도 (V48 로직 적용)
# ==========================================
def analyze_schematic(img, model):
    # [V48 설정] conf=0.05 (5%) -> 아주 관대하게 다 잡음
    results = model.predict(source=img, save=False, conf=0.05, verbose=False)
    boxes = results[0].boxes
    raw_parts = []
    
    for box in boxes:
        cls_id = int(box.cls[0])
        name = model.names[cls_id].lower()
        conf = float(box.conf[0])
        coords = box.xyxy[0].tolist()
        center = get_center(coords)
        
        # [V48 이름 처리]
        base_name = name.split('_')[0].split(' ')[0]
        if base_name in ['vdc', 'vsource', 'battery', 'voltage', 'v']: base_name = 'source'
        if base_name in ['cap', 'c', 'capacitor']: base_name = 'capacitor'
        if base_name in ['res', 'r', 'resistor']: base_name = 'resistor'
        
        raw_parts.append({'name': base_name, 'box': coords, 'center': center, 'conf': conf})

    # [V48 중복 제거 사용]
    clean_parts = solve_overlap_v48(raw_parts)

    # 전원 보정
    if clean_parts:
        has_source = any(p['name'] == 'source' for p in clean_parts)
        if not has_source:
            leftmost_part = min(clean_parts, key=lambda p: p['center'][0])
            leftmost_part['name'] = 'source'

    summary = {'total': 0, 'details': {}}
    for part in clean_parts:
        name = part['name']
        x1, y1, x2, y2 = map(int, part['box'])
        
        # 시각화 (V48 스타일: 파란색 단색)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img, f"{name}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        summary['total'] += 1
        if name not in summary['details']: summary['details'][name] = 0
        summary['details'][name] += 1
        
    return img, summary

# ==========================================
# [분석 2] 실물 보드 (V35 로직 적용 - 사용자 선호)
# ==========================================
def analyze_real(img, model):
    h, w, _ = img.shape
    
    # [V35 기본 설정] conf=0.10
    res = model.predict(source=img, conf=0.10, verbose=False)
    
    bodies = []
    pins = [] 
    
    for b in res[0].boxes:
        name = model.names[int(b.cls[0])].lower()
        coords = b.xyxy[0].tolist()
        center = get_center(coords)
        conf = float(b.conf[0])
        
        # [V35/V15 민감도 설정] 사용자님이 "완벽하다"고 한 값
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

    # [V35 중복 제거 사용]
    clean_bodies = solve_overlap_v35(bodies, dist_thresh=60, iou_thresh=0.4)
    
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

    # 결과 집계 및 시각화
    summary = {'total': 0, 'on': 0, 'off': 0, 'details': {}}
    
    for comp in clean_bodies:
        is_on = comp['is_on']
        raw_name = comp['name']
        
        # 이름 정규화
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
        
        # 와이어는 개수 비교 제외
        if 'wire' not in raw_name:
            if norm_name not in summary['details']: summary['details'][norm_name] = {'count': 0}
            summary['details'][norm_name]['count'] += 1

        if is_on:
            color = (0, 255, 0)
            status = "ON"
            summary['on'] += 1
        else:
            color = (0, 0, 255)
            status = "OFF"
            summary['off'] += 1
        
        summary['total'] += 1
        
        display_text = f"{label_name}: {status}"
        x1, y1, x2, y2 = map(int, comp['box'])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.putText(img, display_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
    return img, summary

# ==========================================
# [Main UI]
# ==========================================
st.title("🧠 BrainBoard V50: Hybrid Perfect")
st.markdown("### 회로도(V48 로직) + 실물(V35 로직)")

@st.cache_resource
def load_models():
    return YOLO(MODEL_REAL_PATH), YOLO(MODEL_SYM_PATH)

try:
    model_real, model_sym = load_models()
    st.sidebar.success("✅ 모델 로드 성공")
except Exception as e:
    st.error(f"모델 파일을 찾을 수 없습니다: {e}")
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
            
            # V48 로직으로 회로도 분석
            res_ref_img, ref_data = analyze_schematic(ref_cv.copy(), model_sym)
            # V35 로직으로 실물 분석
            res_tgt_img, tgt_data = analyze_real(tgt_cv.copy(), model_real)

            issues = []
            # 비교 로직
            all_parts = set(ref_data['details'].keys()) | set(tgt_data['details'].keys())
            counts_match = True
            
            for part in all_parts:
                if part in ['wire', 'breadboard', 'text', 'hole']: continue
                
                ref_c = ref_data['details'].get(part, 0)
                tgt_c = tgt_data['details'].get(part, {}).get('count', 0)
                
                if ref_c != tgt_c:
                    issues.append(f"⚠️ {part.capitalize()} 개수 불일치 (회로도:{ref_c}개 vs 실물:{tgt_c}개)")
                    counts_match = False
                else:
                    issues.append(f"✅ {part.capitalize()} 개수 일치 ({ref_c}개)")

            st.divider()
            
            if counts_match:
                st.success("🎉 회로 구성이 완벽합니다!")
            else:
                st.warning("⚠️ 회로 구성에 차이가 있습니다.")
            
            for i in issues:
                if "✅" in i: st.caption(i)
                else: st.error(i)

            st.image(cv2.cvtColor(res_ref_img, cv2.COLOR_BGR2RGB), caption="회로도 분석 (V48 Logic)", use_column_width=True)
            st.image(cv2.cvtColor(res_tgt_img, cv2.COLOR_BGR2RGB), caption="실물 보드 분석 (V35 Logic)", use_column_width=True)
