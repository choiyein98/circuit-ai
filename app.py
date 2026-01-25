import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import math
from PIL import Image

# ==========================================
# [1. 설정: 모든 모델 경로 로드]
# ==========================================
st.set_page_config(page_title="BrainBoard V56: Triple Ensemble", layout="wide")

# 모든 실물 모델 리스트
REAL_MODEL_PATHS = ['best.pt', 'best(2).pt', 'best(3).pt']
MODEL_SYM_PATH = 'symbol.pt'

# ==========================================
# [2. 앙상블 분석 엔진]
# ==========================================
def analyze_ensemble(img, real_models, sym_model):
    h, w, _ = img.shape
    all_real_results = []
    
    # [핵심] 3개의 실물 모델을 차례로 돌려 결과 수집
    for model in real_models:
        res = model.predict(source=img, conf=0.50, imgsz=640, verbose=False) # 신뢰도 높여서 노이즈 차단
        for b in res[0].boxes:
            name = model.names[int(b.cls[0])].lower()
            if any(x in name for x in ['res', 'r']): norm_name = 'RESISTOR'
            elif any(x in name for x in ['cap', 'c']): norm_name = 'CAPACITOR'
            elif any(x in name for x in ['v', 'volt', 'batt', 'source']): norm_name = 'SOURCE'
            else: continue
            
            all_real_results.append({
                'name': norm_name,
                'box': b.xyxy[0].tolist(),
                'conf': float(b.conf[0]),
                'center': ((b.xyxy[0][0]+b.xyxy[0][2])/2, (b.xyxy[0][1]+b.xyxy[0][3])/2)
            })

    # [중복 제거] 3개 모델의 결과 중 겹치는 건 하나로 합침 (NMS)
    all_real_results.sort(key=lambda x: x['conf'], reverse=True)
    clean_real = []
    for curr in all_real_results:
        is_dup = False
        for k in clean_real:
            # IoU 계산
            x1, y1, x2, y2 = max(curr['box'][0], k['box'][0]), max(curr['box'][1], k['box'][1]), \
                             min(curr['box'][2], k['box'][2]), min(curr['box'][3], k['box'][3])
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            area1 = (curr['box'][2]-curr['box'][0]) * (curr['box'][3]-curr['box'][1])
            area2 = (k['box'][2]-k['box'][0]) * (k['box'][3]-k['box'][1])
            iou = inter / (area1 + area2 - inter) if (area1 + area2 - inter) > 0 else 0
            
            if iou > 0.2: # 20%만 겹쳐도 동일 부품으로 간주하여 중복 제거
                is_dup = True; break
        if not is_dup: clean_real.append(curr)

    # [회로도 분석] (기존 성능 유지)
    res_sym = sym_model.predict(source=img, conf=0.25, imgsz=640, verbose=False)
    clean_sym = []
    # (회로도 중복 제거 로직 생략 - 기존과 동일)
    
    # ... (생략된 시각화 및 카운트 로직)
    return clean_real # 최종 합쳐진 실물 결과 반환
