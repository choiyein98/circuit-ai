# ==========================================
# [분석 함수 2: 실물 (Real Board) - 다리 중심 시각화]
# ==========================================
def analyze_real(img, model):
    height, width, _ = img.shape
    
    # 1. 모델 예측
    results = model.predict(source=img, save=False, conf=0.15, verbose=False)
    boxes = results[0].boxes

    # 2. 객체 수집
    components = [] 
    legs = []       
    
    for box in boxes:
        cls_id = int(box.cls[0])
        name = model.names[cls_id].lower()
        coords = box.xyxy[0].tolist()
        center = get_center(coords)
        
        # 노이즈 제거
        if (coords[2]-coords[0]) * (coords[3]-coords[1]) < (width * height * 0.001): continue

        if any(x in name for x in ['pin', 'leg', 'lead', 'wire']):
            # 다리(Pin)는 별도 리스트로 저장
            legs.append({'center': center, 'box': coords, 'type': 'pin'})
        elif 'breadboard' in name:
            continue
        else:
            # 몸통(Body)
            components.append({
                'name': name, 'box': coords, 'center': center, 
                'my_legs': [], 'is_active': False
            })

    components = solve_overlap(components, distance_threshold=50)

    # 3. [가상 전원 레일] (인식 범위)
    top_rail_y = height * 0.20
    bottom_rail_y = height * 0.80
    
    # 전원 영역 표시 (노란 점선 느낌)
    cv2.rectangle(img, (0, 0), (width, int(top_rail_y)), (0, 255, 255), 1) 
    cv2.rectangle(img, (0, int(bottom_rail_y)), (width, height), (0, 255, 255), 1) 
    
    # 4. [다리 할당] 몸통과 가장 가까운 다리들을 찾아서 연결
    for comp in components:
        bw = comp['box'][2] - comp['box'][0]
        bh = comp['box'][3] - comp['box'][1]
        diag = math.sqrt(bw**2 + bh**2)
        search_radius = diag * 0.8  # 부품 크기 반경 내 검색

        for leg in legs:
            dist = math.sqrt((leg['center'][0]-comp['center'][0])**2 + (leg['center'][1]-comp['center'][1])**2)
            if dist < search_radius:
                comp['my_legs'].append(leg)
        
        # 다리가 인식 안 됐을 경우, 몸통 양 끝을 가상의 다리로 설정
        if len(comp['my_legs']) < 2:
            x1, y1, x2, y2 = comp['box']
            if bw > bh: # 가로형
                comp['my_legs'] = [{'center':(x1, (y1+y2)/2)}, {'center':(x2, (y1+y2)/2)}]
            else: # 세로형
                comp['my_legs'] = [{'center':((x1+x2)/2, y1)}, {'center':((x1+x2)/2, y2)}]

    # 5. [전류 흐름 시뮬레이션]
    # (A) 전원 소스 찾기 (레일에 닿은 다리)
    active_legs = [] 
    
    for comp in components:
        for leg in comp['my_legs']:
            ly = leg['center'][1]
            if ly < top_rail_y or ly > bottom_rail_y:
                comp['is_active'] = True
                active_legs.append(leg['center'])
    
    # (B) 전류 전파 (거리 기반)
    CONNECTION_THRESHOLD = 90 # 연결 허용 거리 (픽셀)
    
    changed = True
    while changed:
        changed = False
        for comp in components:
            if comp['is_active']: 
                # 내가 켜졌으면 내 다리들도 전원 소스가 됨
                for leg in comp['my_legs']:
                    if leg['center'] not in active_legs:
                        active_legs.append(leg['center'])
                        changed = True
                continue
            
            # 내가 꺼져 있으면 주변에 활성 다리가 있는지 확인
            for my_leg in comp['my_legs']:
                for active_pt in active_legs:
                    dist = math.sqrt((my_leg['center'][0]-active_pt[0])**2 + (my_leg['center'][1]-active_pt[1])**2)
                    if dist < CONNECTION_THRESHOLD:
                        comp['is_active'] = True
                        changed = True
                        break 
                if comp['is_active']: break

    # 6. [시각화 수정] 몸통이 아닌 "다리"에 집중해서 표시
    summary = {'total': 0, 'on': 0, 'off': 0, 'details': {}}
    
    for comp in components:
        name = comp['name']
        x1, y1, x2, y2 = map(int, comp['box'])
        center = comp['center']

        if comp['is_active']:
            color = (0, 255, 0) # 초록 (ON)
            status = "ON"
            summary['on'] += 1
        else:
            color = (0, 0, 255) # 빨강 (OFF)
            status = "OFF"
            summary['off'] += 1
            
        summary['total'] += 1
        
        # 1) 몸통 박스는 얇게 표시 (부품 식별용)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
        
        # 2) [핵심] 다리 위치에 '점' 찍고 연결선 그리기
        for leg in comp['my_legs']:
            lx, ly = map(int, leg['center'])
            
            # 몸통 중심에서 다리까지 선 그리기 (소속 표시)
            cv2.line(img, (int(center[0]), int(center[1])), (lx, ly), color, 2)
            
            # 다리 끝부분에 원 그리기 (여기가 연결 포인트!)
            cv2.circle(img, (lx, ly), 8, color, -1) 
            # 원 테두리 (잘 보이게)
            cv2.circle(img, (lx, ly), 8, (255, 255, 255), 2)

        # 3) 상태 텍스트 표시
        cv2.putText(img, status, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        base_name = name.split('_')[0]
        summary['details'][base_name] = summary['details'].get(base_name, 0) + 1

    return img, summary
