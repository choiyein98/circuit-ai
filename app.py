import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import math
from PIL import Image
from collections import defaultdict
import gc
from datetime import datetime

# ==========================================
# [설정] CircuitMate AI V71: History Recall
# ==========================================
st.set_page_config(page_title="CircuitMate AI", layout="wide", page_icon="⚡")

# [세션 상태 초기화]
if 'history' not in st.session_state:
    st.session_state['history'] = []
if 'active_result' not in st.session_state:
    st.session_state['active_result'] = None  # 현재 화면에 보여줄 데이터

REAL_MODEL_PATH = 'best(3).pt' 
MODEL_SYM_PATH = 'symbol.pt'

# ==========================================
# [Core Logic] 기존 V69/V70 로직 (변경 없음)
# ==========================================
def resize_image_smart(image, max_size=1024):
    h, w = image.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image

def get_center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = ((box1[2]-box1[0])*(box1[3]-box1[1])) + ((box2[2]-box2[0])*(box2[3]-box2[1])) - inter
    return inter / union if union > 0 else 0

def normalize_name(name):
    name = name.lower()
    if 'res' in name: return 'resistor'
    if 'cap' in name: return 'capacitor'
    if 'wire' in name: return 'wire'
    if any(x in name for x in ['source', 'batt', 'volt', 'vdc']): return 'source'
    if any(x in name for x in ['leg', 'pin', 'lead']): return 'leg'
    return name

def solve_overlap_real(parts):
    if not parts: return []
    parts.sort(key=lambda x: x.get('conf', 0), reverse=True)
    final = []
    for curr in parts:
        is_dup = False
        for k in final:
            iou = calculate_iou(curr['box'], k['box'])
            dist = math.sqrt((curr['center'][0]-k['center'][0])**2 + (curr['center'][1]-k['center'][1])**2)
            if curr['name'] != 'leg' and (iou > 0.4 or dist < 60): 
                is_dup = True; break
        if not is_dup: final.append(curr)
    return final

def sort_parts_LRTB(parts, image_width):
    if not parts: return []
    parts.sort(key=lambda x: x['center'][0])
    sorted_sequence = []
    current_column = []
    X_THRESHOLD = image_width * 0.10
    
    current_column.append(parts[0])
    ref_x = parts[0]['center'][0]
    
    for i in range(1, len(parts)):
        curr = parts[i]
        curr_x = curr['center'][0]
        if abs(curr_x - ref_x) < X_THRESHOLD:
            current_column.append(curr)
        else:
            current_column.sort(key=lambda x: x['center'][1])
            sorted_sequence.extend(current_column)
            current_column = [curr]
            ref_x = curr_x
            
    if current_column:
        current_column.sort(key=lambda x: x['center'][1])
        sorted_sequence.extend(current_column)
    return sorted_sequence

def analyze_schematic(img, model):
    img = resize_image_smart(img)
    w = img.shape[1]
    results = model.predict(source=img, save=False, conf=0.05, verbose=False)
    raw_parts = []
    for box in results[0].boxes:
        raw_name = model.names[int(box.cls[0])]
        norm_name = normalize_name(raw_name)
        if norm_name == 'wire' or norm_name == 'leg': continue
        coords = box.xyxy[0].tolist()
        raw_parts.append({'name': norm_name, 'box': coords, 'center': get_center(coords), 'conf': float(box.conf[0])})

    parts = []
    raw_parts.sort(key=lambda x: x['conf'], reverse=True)
    for p in raw_parts:
        if not any(calculate_iou(p['box'], k['box']) > 0.1 for k in parts): parts.append(p)

    if parts and not any(p['name'] == 'source' for p in parts):
         leftmost = min(parts, key=lambda p: p['center'][0])
         leftmost['name'] = 'source'

    for p in parts:
        x1, y1, x2, y2 = map(int, p['box'])
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img, p['name'], (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    sorted_parts = sort_parts_LRTB(parts, w)
    for i, p in enumerate(sorted_parts):
        cx, cy = map(int, p['center'])
        cv2.circle(img, (cx, cy), 15, (0, 0, 255), -1)
        cv2.putText(img, str(i+1), (cx-5, cy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return img, {'parts': sorted_parts}

def analyze_real(img, model):
    img = resize_image_smart(img)
    h, w, _ = img.shape
    res = model.predict(source=img, conf=0.10, verbose=False)
    raw_objects = []
    for b in res[0].boxes:
        raw_name = model.names[int(b.cls[0])]
        norm_name = normalize_name(raw_name)
        conf = float(b.conf[0])
        if norm_name == 'capacitor' and conf < 0.20: continue
        if norm_name == 'resistor' and conf < 0.25: continue
        if 'breadboard' in raw_name: continue
        coords = b.xyxy[0].tolist()
        raw_objects.append({'name': norm_name, 'box': coords, 'center': get_center(coords), 'conf': conf})

    parts_candidates = [p for p in raw_objects if p['name'] != 'leg']
    legs = [p for p in raw_objects if p['name'] == 'leg']
    parts = solve_overlap_real(parts_candidates)

    TOP_RAIL = h * 0.20; BOTTOM_RAIL = h * 0.80
    has_source = False
    if any(p['name'] == 'source' for p in parts): has_source = True
    if not has_source:
        for p in raw_objects:
            if p['center'][1] < TOP_RAIL or p['center'][1] > BOTTOM_RAIL:
                if p['name'] == 'wire' or p['name'] == 'leg':
                    has_source = True; break
    if has_source and not any(p['name'] == 'source' for p in parts):
        parts.append({'name': 'source', 'box': [0,0,0,0], 'center': (0,0), 'conf': 1.0})

    for p in parts:
        if p['name'] == 'wire': continue
        color = (0, 255, 0)
        if p['name'] == 'source': color = (0, 255, 255)
        if p['box'][2] > 0:
            x1, y1, x2, y2 = map(int, p['box'])
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            cv2.putText(img, p['name'].upper(), (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        elif p['name'] == 'source':
            cv2.putText(img, "SOURCE DETECTED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    main_parts = [p for p in parts if p['name'] != 'wire']
    sorted_parts = sort_parts_LRTB(main_parts, w)

    for i, p in enumerate(sorted_parts):
        if p['box'][2] > 0:
            cx, cy = map(int, p['center'])
            cv2.circle(img, (cx, cy), 15, (0, 0, 255), -1)
            cv2.putText(img, str(i+1), (cx-5, cy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return img, {'parts': sorted_parts}

# ==========================================
# [Renderer] 분석 결과를 화면에 그려주는 함수
# ==========================================
def render_result(result_data):
    """저장된 분석 데이터를 화면에 출력하는 함수"""
    
    st.divider()
    st.markdown("## 📊 분석 결과 리포트")

    bom_match = result_data['bom_match']
    is_seq_match = result_data['is_seq_match']
    bom_data = result_data['bom_data']
    ref_list = result_data['ref_list']
    tgt_list = result_data['tgt_list']
    
    col_res1, col_res2 = st.columns([1, 1])
    
    with col_res1:
        st.markdown("### 📋 부품 목록 확인")
        st.dataframe(bom_data, hide_index=True)

    with col_res2:
        st.markdown("### 🔗 연결 순서 검증")
        if not bom_match:
            st.warning("⚠️ 부품 개수가 달라서 정확한 순서 비교가 어렵습니다.")
            st.caption(f"회로도: {' → '.join(ref_list)}")
            st.caption(f"실물: {' → '.join(tgt_list)}")
        else:
            for i in range(len(ref_list)):
                r_item = ref_list[i]
                t_item = tgt_list[i]
                if r_item == t_item:
                    st.info(f"**Step {i+1}:** {r_item.upper()} ✅ 정상 연결됨")
                else:
                    st.error(f"**Step {i+1}:** 불일치 감지! (회로도: {r_item} vs 실물: {t_item})")
            
            if is_seq_match:
                st.success("완벽합니다! 회로 연결 순서가 정확해요. 🎉")
                st.balloons()

    st.markdown("### 📷 AI 인식 화면")
    img_col1, img_col2 = st.columns(2)
    with img_col1:
        st.image(result_data['res_ref_img'], caption="회로도 분석 (번호는 전류 흐름 순서)", use_column_width=True)
    with img_col2:
        st.image(result_data['res_tgt_img'], caption="실물 분석 (번호는 배치 순서)", use_column_width=True)


# ==========================================
# [UI/UX] Sidebar & Main Layout
# ==========================================

# [사이드바]
with st.sidebar:
    st.title("⚡ CircuitMate AI")
    st.caption("Your Personal Circuit Assistant")
    st.divider()
    
    try:
        if 'models_loaded' not in st.session_state:
            gc.collect()
            st.session_state['model_real'] = YOLO(REAL_MODEL_PATH)
            st.session_state['model_sym'] = YOLO(MODEL_SYM_PATH)
            st.session_state['models_loaded'] = True
        st.success("✅ 시스템 준비 완료")
    except Exception as e:
        st.error(f"시스템 오류: {e}")
        st.stop()

    st.divider()
    st.markdown("### 🕒 최근 검증 기록")
    
    # 히스토리 버튼 생성 (ChatGPT 스타일)
    if not st.session_state['history']:
        st.caption("아직 기록이 없습니다.")
    else:
        # 최신순으로 정렬
        for idx, item in enumerate(reversed(st.session_state['history'])):
            # 고유한 key를 위해 idx 사용
            btn_label = f"{item['time']} - {item['status']}"
            if st.button(btn_label, key=f"hist_{idx}", use_container_width=True):
                st.session_state['active_result'] = item # 클릭 시 해당 결과 활성화

# [메인 화면]
st.markdown("""
# 👋 안녕하세요! 회로 검증을 도와드릴게요.
회로도와 실물 브레드보드 사진을 업로드해주시면, **부품의 종류와 연결 순서**를 꼼꼼하게 비교해드립니다.
""")

col1, col2 = st.columns(2)
with col1:
    st.markdown("### 1️⃣ 회로도 (Schematic)")
    ref_file = st.file_uploader("회로도 이미지를 넣어주세요", type=['jpg', 'png', 'jpeg'])

with col2:
    st.markdown("### 2️⃣ 실물 사진 (Real Board)")
    tgt_file = st.file_uploader("브레드보드 사진을 넣어주세요", type=['jpg', 'png', 'jpeg'])

# 분석 로직
if ref_file and tgt_file:
    # 파일이 새로 올라오면 이미지 로드
    ref_image = Image.open(ref_file)
    tgt_image = Image.open(tgt_file)
    ref_cv = cv2.cvtColor(np.array(ref_image), cv2.COLOR_RGB2BGR)
    tgt_cv = cv2.cvtColor(np.array(tgt_image), cv2.COLOR_RGB2BGR)

    if st.button("✨ 분석 시작하기 (Analyze)", type="primary"):
        gc.collect()
        progress_text = "AI가 회로를 분석하고 있습니다... 잠시만 기다려주세요!"
        my_bar = st.progress(0, text=progress_text)

        # 1. 분석 실행
        res_ref_img, ref_data = analyze_schematic(ref_cv.copy(), st.session_state['model_sym'])
        my_bar.progress(50, text="실물 보드의 부품을 인식하고 있습니다...")
        res_tgt_img, tgt_data = analyze_real(tgt_cv.copy(), st.session_state['model_real'])
        my_bar.progress(90, text="검증 데이터를 정리 중입니다...")

        # 2. 데이터 가공
        ref_counts = defaultdict(int)
        tgt_counts = defaultdict(int)
        for p in ref_data['parts']: ref_counts[p['name']] += 1
        for p in tgt_data['parts']: tgt_counts[p['name']] += 1
        
        all_keys = set(ref_counts.keys()) | set(tgt_counts.keys())
        bom_match = True
        bom_data = []
        
        for k in all_keys:
            if k == 'wire': continue
            r = ref_counts[k]; t = tgt_counts[k]
            status = "✅ 일치" if r == t else "⚠️ 확인 필요"
            bom_data.append({"부품명": k.upper(), "회로도 개수": r, "실물 개수": t, "상태": status})
            if r != t: bom_match = False
            
        ref_list = [p['name'] for p in ref_data['parts']]
        tgt_list = [p['name'] for p in tgt_data['parts']]
        
        is_seq_match = True
        if not bom_match:
            is_seq_match = False
        else:
            for i in range(len(ref_list)):
                if ref_list[i] != tgt_list[i]: is_seq_match = False

        # 3. 결과 딕셔너리 생성 (저장용)
        timestamp = datetime.now().strftime("%H:%M:%S")
        status_msg = "성공 ✅" if (bom_match and is_seq_match) else "실패 ❌"
        
        result_packet = {
            "time": timestamp,
            "status": status_msg,
            "bom_match": bom_match,
            "is_seq_match": is_seq_match,
            "bom_data": bom_data,
            "ref_list": ref_list,
            "tgt_list": tgt_list,
            # 이미지를 RGB로 변환하여 저장 (출력용)
            "res_ref_img": cv2.cvtColor(res_ref_img, cv2.COLOR_BGR2RGB),
            "res_tgt_img": cv2.cvtColor(res_tgt_img, cv2.COLOR_BGR2RGB)
        }

        # 4. 세션에 저장 및 활성화
        st.session_state['history'].append(result_packet)
        st.session_state['active_result'] = result_packet
        
        my_bar.empty()
        gc.collect()

# [결과 화면 렌더링]
# active_result에 데이터가 있으면 화면에 표시 (방금 분석했거나, 히스토리에서 클릭했거나)
if st.session_state['active_result']:
    render_result(st.session_state['active_result'])
