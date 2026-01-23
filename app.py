import cv2
import numpy as np
from ultralytics import YOLO
import sys
import math
import tkinter as tk
from tkinter import filedialog

# ==========================================
# [설정 및 상수]
# ==========================================
MODEL_REAL_PATH = 'best.pt'    # 실물 보드용 모델
MODEL_SYM_PATH = 'symbol.pt'   # 회로도용 모델
PIN_SENSITIVITY = 140          # 핀과 부품 간 연결 감지 범위 (픽셀 단위)

# ==========================================
# [Helper Functions]
# ==========================================
def solve_overlap(parts, dist_thresh=60):
    """
    중복 감지된 객체들을 거리 기준으로 필터링 (Conf 높은 것 우선)
    """
    if not parts: return []
    # conf(신뢰도)가 높은 순서대로 정렬
    if 'conf' in parts[0]:
        parts.sort(key=lambda x: x.get('conf', 0), reverse=True)
    
    final = []
    for curr in parts:
        # 이미 등록된 부품들과 너무 가까우면(중복이면) 건너뜀
        if not any(math.sqrt((curr['center'][0]-k['center'][0])**2 + (curr['center'][1]-k['center'][1])**2) < dist_thresh for k in final):
            final.append(curr)
    return final

# ==========================================
# [분석 함수 1: 회로도 (Schematic)]
# ==========================================
def analyze_schematic(img_path, model):
    img = cv2.imread(img_path)
    if img is None: return None
    
    # 모델 추론
    res = model.predict(source=img, conf=0.15, verbose=False)
    
    raw = []
    for b in res[0].boxes:
        raw.append({
            'name': model.names[int(b.cls[0])].lower(), 
            'box': b.xyxy[0].tolist(), 
            'center': ((b.xyxy[0][0]+b.xyxy[0][2])/2, (b.xyxy[0][1]+b.xyxy[0][3])/2),
            'conf': float(b.conf[0])
        })
    
    clean = solve_overlap(raw)
    
    for p in clean:
        name = p['name']
        # 가장 왼쪽 부품을 전원(Source)으로 강제 지정 (회로도 특성상)
        if p['center'][0] < img.shape[1] * 0.25: name = 'source'
        elif 'cap' in name: name = 'capacitor'
        elif 'res' in name: name = 'resistor'
        
        x1, y1, x2, y2 = map(int, p['box'])
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img, name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    return img

# ==========================================
# [분석 함수 2: 실물 (Real Board)]
# ==========================================
def analyze_real(img_path, model):
    img = cv2.imread(img_path)
    if img is None: return None, 0
    h, w, _ = img.shape
    
    # 모델 추론
    res = model.predict(source=img, conf=0.1, verbose=False)
    
    bodies = [] # 시각화할 부품 (몸체 + 와이어)
    pins = []   # 연결 확인용 핀 (다리) - 화면엔 안 그림
    
    for b in res[0].boxes:
        name = model.names[int(b.cls[0])].lower()
        coords = b.xyxy[0].tolist()
        center = ((coords[0]+coords[2])/2, (coords[1]+coords[3])/2)
        conf = float(b.conf[0])
        
        # [수정된 핵심 로직] 
        # 'wire'는 핀 리스트에서 제외하고 bodies(시각화 대상)로 분류
        
        # 1. 핀/다리(Leg) 처리 -> 화면에 안 그리고 좌표 계산용으로만 사용
        if any(x in name for x in ['pin', 'leg', 'lead']) and 'wire' not in name:
            pins.append(center)
        
        # 2. 브레드보드 배경 제외
        elif 'breadboard' in name:
            continue
            
        # 3. 그 외 부품 (저항, 커패시터, 그리고 WIRE 포함) -> 화면에 그림
        else:
            bodies.append({'name': name, 'box': coords, 'center': center, 'conf': conf})

    clean_bodies = solve_overlap(bodies, 60)
    
    # [전원 활성화 로직]
    # 1. 핀이 상단 전원 레일(높이의 45% 지점 위쪽)에 있는지 확인
    power_active = any(p[1] < h * 0.45 for p in pins)
    
    # 2. 핀이 감지되지 않았더라도, 'wire'가 상단 전원부에 있다면 전원 ON으로 간주
    if not power_active:
        for b in clean_bodies:
            if 'wire' in b['name'] and b['center'][1] < h * 0.45:
                power_active = True
                break
    
    off_count = 0
    
    for comp in clean_bodies:
        cx, cy = comp['center']
        name = comp['name']
        is_on = False
        
        # 와이어는 연결선이므로 주황색으로 표시하고 항상 활성 상태로 간주
        if 'wire' in name:
            color = (0, 165, 255) # 주황색 (BGR 순서: Blue=0, Green=165, Red=255)
            status = "WIRE"
            is_on = True # 와이어는 OFF 카운트에서 제외
        else:
            # 일반 부품 로직
            if power_active:
                # A. 부품 자체가 전원 레일 근처(중앙 분리대 위/아래)에 위치
                if cy < h*0.48 or cy > h*0.52: 
                    is_on = True
                else:
                    # B. 부품 근처에 핀이 있고, 그 핀이 전원 쪽에 연결되어 있는지 확인
                    for px, py in pins:
                        if math.sqrt((cx-px)**2 + (cy-py)**2) < PIN_SENSITIVITY:
                            # 핀의 y좌표가 중앙 영역을 벗어나 있으면(전원 레일 쪽) ON
                            if py < h*0.48 or py > h*0.52:
                                is_on = True; break
            
            if is_on:
                color = (0, 255, 0) # 초록 (ON)
                status = "ON"
            else:
                color = (0, 0, 255) # 빨강 (OFF)
                status = "OFF"
                off_count += 1
        
        # 결과 그리기
        x1, y1, x2, y2 = map(int, comp['box'])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.putText(img, status, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
    return img, off_count

# ==========================================
# [Main Execution]
# ==========================================
if __name__ == "__main__":
    try:
        # 윈도우 파일 탐색기 초기화
        root = tk.Tk()
        root.withdraw()
        
        print("--- BrainBoard V44 실행 ---")
        
        print("1. PSpice 회로도 이미지를 선택하세요...")
        p1 = filedialog.askopenfilename(title="1. PSpice 회로도 선택", filetypes=[("Images", "*.jpg;*.png;*.jpeg")])
        if not p1: 
            print("회로도 선택이 취소되었습니다.")
            sys.exit()
        
        print("2. 실물 회로(브레드보드) 이미지를 선택하세요...")
        p2 = filedialog.askopenfilename(title="2. 실물 사진 선택", filetypes=[("Images", "*.jpg;*.png;*.jpeg")])
        if not p2: 
            print("실물 사진 선택이 취소되었습니다.")
            sys.exit()

        print("분석 모델을 로드하고 있습니다...")
        m_real = YOLO(MODEL_REAL_PATH)
        m_sym = YOLO(MODEL_SYM_PATH)

        print("이미지 분석 중...")
        res1 = analyze_schematic(p1, m_sym)
        res2, off = analyze_real(p2, m_real)

        # 결과 이미지 병합 (가로로 이어붙이기)
        if res1 is not None and res2 is not None:
            h1, w1 = res1.shape[:2]
            h2, w2 = res2.shape[:2]
            max_h = max(h1, h2)
            
            canvas = np.zeros((max_h, w1 + w2, 3), dtype=np.uint8)
            canvas[:h1, :w1] = res1
            canvas[:h2, w1:w1+w2] = res2

            # 화면 크기에 맞춰 리사이징 (폭 1400px 넘으면 축소)
            if canvas.shape[1] > 1400:
                scale = 1400 / canvas.shape[1]
                canvas = cv2.resize(canvas, None, fx=scale, fy=scale)

            print(f"분석 완료! 발견된 비정상(OFF) 부품 수: {off}")
            cv2.imshow("BrainBoard Verification", canvas)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")
        import traceback
        traceback.print_exc()
        input("엔터를 누르면 종료합니다...")
