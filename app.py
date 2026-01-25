import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import math
from PIL import Image

# ==========================================
# [설정] BrainBoard V49: Hybrid Perfect
# ==========================================
st.set_page_config(page_title="BrainBoard V49: Hybrid Perfect", layout="wide")

MODEL_REAL_PATH = 'best(3).pt'  # V48 기준 (실물)
MODEL_SYM_PATH = 'symbol.pt'    # V35 기준 (회로도)

# [V48 설정] 부품별 신뢰도 맵 (실물용)
CONFIDENCE_MAP = {
    'led': 0.50,
    'capacitor': 0.40,
    'voltage': 0.25,
    'source': 0.25,
    'resistor': 0.65, # 65% 이상만 인정
    'wire': 0.25,
    'default': 0.30
}

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

# [V48 Helper] 실물 필터링용
def is_valid_size(box, img_w, img_h):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    area = w * h
    img_area = img_w * img_h
    if area < img_area * 0.001: return False 
    return True

# [V48 Helper] 와이어 오인식 방지
def is_wire_misclassified_as_resistor(box):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    if w == 0 or h == 0: return False
    ratio = max(w, h) / min(w, h)
    if ratio > 6.0: return True 
    return False

# [통합 중복 제거 함수]
# V35(회로도)와 V48(실물)의 로직을 모두 수용
def solve_overlap(parts, dist_thresh=80, iou_thresh=0.4, is_schematic=False):
    if not parts: return []
    
    # -----------------------------------------------------------
    # 정렬 전략
    # -----------------------------------------------------------
    if is_schematic:
        # [회로도] V35: 면적이 '작은' 순서대로 정렬 (껍데기 제거용)
        # 딕셔너리 구조가 V35와 V48이 약간 다를 수 있으므로 처리
        parts.sort(key=lambda x: (x['box'][2]-x['box'][0]) * (x['box'][3]-x['box'][1]))
    else:
        # [실물] V48: 신뢰도(conf) 높은 순서대로 정렬
        parts.sort(key=lambda x: x.get('conf', 0), reverse=True)
    
    final = []
    for curr in parts:
        is_dup = False
        for k in final:
            # 좌표 및 면적 계산
            inter_area = 0
            iou = calculate_iou(curr['box'], k['box'])
            
            # 중심 거리 계산
            dist = math.sqrt((curr['center'][0]-k['center'][0])**2 + (curr['center'][1]-k['center'][1])**2)

            # -----------------------------------------------------------
            # [MODE A] 회로도 (V35 로직: 껍데기 박멸)
            # -----------------------------------------------------------
            if is_schematic:
                # 면적 계산 필요
                x1 = max(curr['box'][0], k['box'][0])
                y1 = max(curr['box'][1], k['box'][1])
                x2 = min(curr['box'][2], k['box'][2])
                y2 = min(curr['box'][3], k['box'][3])
                inter_area = max(0, x2-x1) * max(0, y2-y1)

                # k: 이미 등록된 '작은 진짜 박스'
                # curr: 지금 검사하는 '큰 박스' (나중에 들어옴)
                
                # [조건 1] 겹침 발생 시 삭제
                if inter_area > 0:
                    is_dup = True; break
                
                # [조건 2] 거리 기반 삭제 (텍스트 박스)
                if dist < 80:
                    is_dup = True; break

            # -----------------------------------------------------------
            # [MODE B] 실물 (V48 로직: 엄격한 필터링)
            # -----------------------------------------------------------
            else:
                # V48의 로직: IoU > 0.1 이거나 거리가 80 미만이면 중복
                if iou > 0.1 or dist < dist_thresh:
                    is_dup = True; break

        if not is_dup:
            final.append(curr)
            
    return final

# ==========================================
# [분석 1] 회로도 (V35 로직 적용)
# ==========================================
def analyze_schematic(img, model):
    # [V35 설정] 0.15 Conf
    res = model.predict(source=img, conf=0.15, verbose=False)
    
    raw = []
    for b in res[0].boxes:
        cls_id = int(b.cls[0])
        raw_name = model.names[cls_id].lower()
        conf = float(b.conf[0])
        
        # [V35 이름 매핑] 'V' 인식 및 이름 통일
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
    
    # [V35 중복 제거] is_schematic=True (작은 것 우선)
    clean_parts = solve_overlap(raw, dist_thresh=80, iou_thresh=0.1, is_schematic=True)
    
    # 전원 위치 보정
    leftmost_idx = -1
    min_x = float('inf')
    has_source = any(p['name'] == 'source' for p in clean_parts)
    if not has_source and clean_parts:
        for i, p in enumerate(clean_parts):
            if p['center'][0] < min_x:
                min_x = p['center'][0]
                leftmost_idx = i

    summary = {'total': 0, 'details': {}}
    
    for i, p in enumerate(clean_parts):
        name = p['name']
        if i == leftmost_idx: name = 'source'
        
        x1, y1, x2, y2 = map(int, p['box'])
        
        # [V35 시각화] V는 파랑, 나머지는 빨강
        if name == 'source':
            box_color = (255, 0, 0) # Blue
            disp_name = "V"
        else:
            box_color = (0, 0, 255) # Red
            disp_name = name
            
        cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(img, disp_name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
        
        if name not in summary['details']: summary['details'][name] = 0
        summary['details'][name] += 1
        summary['total'] += 1
        
    return img, summary

# ==========================================
# [분석 2] 실물 보드 (V48 로직 유지)
# ==========================================
def analyze_real(img, model):
    height, width, _ = img.shape
    # [V48 설정] 기본 10%
    results = model.predict(source=img, save=False, conf=0.1, verbose=False)
    boxes = results[0].boxes

    raw_objects = {'body': [], 'leg': [], 'plus': [], 'minus': []}
    
    for box in boxes:
        cls_id = int(box.cls[0])
        name = model.names[cls_id].lower()
        conf = float(box.conf[0])
        
        # [V48 필터링] 신뢰도 맵 적용
        threshold = CONFIDENCE_MAP.get('default')
        for key in CONFIDENCE_MAP:
            if key in name: threshold = CONFIDENCE_MAP[key]; break
        
        if conf < threshold: continue
        if name in ['breadboard', 'text', 'hole']: continue
        
        coords = box.xyxy[0].tolist()
        if not is_valid_size(coords, width, height): continue
        
        # [V48 필터링] 저항/와이어 비율 체크
        if 'resistor' in name and is_wire_misclassified_as_resistor(coords):
            continue

        item = {'name': name, 'box': coords, 'center': get_center(coords), 'conf': conf}
        
        if any(x in name for x in ['pin', 'leg', 'lead']): raw_objects['leg'].append(item)
        elif any(x in name for x in ['plus', 'positive', 'vcc', '5v']): raw_objects['plus'].append(item)
        elif any(x in name for x in ['minus', 'negative', 'gnd']): raw_objects['minus'].append(item)
        else: raw_objects['body'].append(item)

    # [V48 중복 제거] is_schematic=False (점수순, V48 로직)
    clean_bodies = solve_overlap(raw_objects['body'], dist_thresh=80, is_schematic=False)

    # [V48 시각화] 가상 레일
    virtual_rails = {'plus': [], 'minus': []}
    virtual_rails['plus'].append({'box': [0, 0, width, height*0.20], 'type': 'VCC'})
    virtual_rails['minus'].append({'box': [0, height*0.80, width, height], 'type': 'GND'})
    
    for r in virtual_rails['plus']:
        cv2.rectangle(img, (0, 0), (width, int(height*0.20)), (0, 255, 255), 1)
    for r in virtual_rails['minus']:
        cv2.rectangle(img, (0, int(height*0.80)), (width, height), (255, 200, 0), 1)

    components = []
    for body in clean_bodies:
        components.append({'body': body, 'is_active': True})

    summary = {'total': 0, 'on': 0, 'off': 0, 'details': {}}
    
    # 가상 Source 로직
    wire_count = sum(1 for c in components if 'wire' in c['body']['name'])
    if wire_count >= 2:
        summary['details']['source'] = {'count': 1}

    for comp in components:
        name = comp['body']['name']
        color = (0, 255, 0)
        status = f"{name} {comp['body']['conf']:.2f}"
        
        x1, y1, x2, y2 = map(int, comp['body']['box'])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        
        label_size, baseline = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        y_label = max(y1, label_size[1] + 10)
        cv2.rectangle(img, (x1, y_label - label_size[1] - 10), (x1 + label_size[0], y_label + baseline - 10), color, -1)
        cv2.putText(img, status, (x1, y_label - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        summary['total'] += 1
        summary['on'] += 1

        base_name = name.split('_')[0].split(' ')[0]
        if base_name in ['voltage', 'source', 'battery']: base_name = 'source'
        if base_name in ['cap', 'c', 'capacitor']: base_name = 'capacitor'
        if base_name in ['res', 'r', 'resistor']: base_name = 'resistor'
        
        if base_name not in summary['details']: summary['details'][base_name] = {'count': 0}
        summary['details'][base_name]['count'] += 1
        
    return img, summary

# ==========================================
# [Main UI] (V48 스타일 유지)
# ==========================================
st.title("🧠 BrainBoard V49: Hybrid Perfect")
st.markdown("### 회로도(V35 로직) + 실물(V48 로직) 통합 버전")

@st.cache_resource
def load_models():
    return YOLO(MODEL_REAL_PATH), YOLO(MODEL_SYM_PATH)

try:
    model_real, model_sym = load_models()
    st.sidebar.success("✅ AI 모델 로드 완료!")
except Exception as e:
    st.error(f"모델 파일을 찾을 수 없습니다: {e}")
    st.stop()

col1, col2 = st.columns(2)
ref_file = col1.file_uploader("1. 회로도(Schematic) 업로드", type=['jpg', 'png', 'jpeg'])
tgt_file = col2.file_uploader("2. 실물(Real Board) 업로드", type=['jpg', 'png', 'jpeg'])

if ref_file and tgt_file:
    ref_image = Image.open(ref_file)
    tgt_image = Image.open(tgt_file)
    ref_cv = cv2.cvtColor(np.array(ref_image), cv2.COLOR_RGB2BGR)
    tgt_cv = cv2.cvtColor(np.array(tgt_image), cv2.COLOR_RGB2BGR)

    if st.button("🚀 분석 시작 (Analyze)"):
        with st.spinner("AI가 정밀 분석 중입니다..."):
            
            # 회로도 분석 (V35 로직)
            res_ref_img, ref_data = analyze_schematic(ref_cv.copy(), model_sym)
            # 실물 분석 (V48 로직)
            res_tgt_img, tgt_data = analyze_real(tgt_cv.copy(), model_real)

            issues = []
            # 비교 로직 (V48 스타일)
            # 회로도 데이터 구조: ref_data['details'] = {'resistor': 3, ...}
            # 실물 데이터 구조: tgt_data['details'] = {'resistor': {'count': 3}, ...}
            
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

            st.image(cv2.cvtColor(res_ref_img, cv2.COLOR_BGR2RGB), caption="회로도 분석 (V35 Logic)", use_column_width=True)
            st.image(cv2.cvtColor(res_tgt_img, cv2.COLOR_BGR2RGB), caption="실물 보드 분석 (V48 Logic)", use_column_width=True)
