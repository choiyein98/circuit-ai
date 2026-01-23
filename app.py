# ==========================================
# [수정된 함수] 시각화 시 박스 크기 보정 기능 추가
# ==========================================
def analyze_real(img, model):
    height, width, _ = img.shape
    
    # 1. 모델 예측
    results = model.predict(source=img, save=False, conf=0.15, verbose=False)
    boxes = results[0].boxes

    # 2. 부품 분류
    components = [] 
    legs = []       
    
    for box in boxes:
        cls_id = int(box.cls[0])
        name = model.names[cls_id].lower()
        conf = float(box.conf[0])
        coords = box.xyxy[0].tolist()
        center = get_center(coords)
        
        # 노이즈 제거
        w_box, h_box = coords[2]-coords[0], coords[3]-coords[1]
        if w_box * h_box < (width * height * 0.001): continue

        if any(x in name for x in ['pin', 'leg', 'lead', 'wire']):
            legs.append({'center': center, 'box': coords})
        elif 'breadboard' in name:
            continue
        else:
            components.append({
                'name': name, 'box': coords, 'center': center, 
                'conf': conf, 'connected_nodes': set(), 'is_active': False
            })

    components = solve_overlap(components, distance_threshold=50)

    # 3. [가상 전원 레일]
    top_rail_y = height * 0.15
    bottom_rail_y = height * 0.85
    
    cv2.rectangle(img, (0, 0), (width, int(top_rail_y)), (0, 255, 255), 2) 
    cv2.rectangle(img, (0, int(bottom_rail_y)), (width, height), (0, 255, 255), 2) 
    cv2.putText(img, "Virtual Power Rail (VCC)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # 4. 연결 로직 (기존과 동일 - 넓은 박스 사용)
    active_nodes = set(["VCC_RAIL"]) 
    
    for comp in components:
        comp_legs = []
        for leg in legs:
            lx, ly = leg['center']
            cx, cy = comp['center']
            dist = math.sqrt((lx-cx)**2 + (ly-cy)**2)
            box_diag = math.sqrt((comp['box'][2]-comp['box'][0])**2 + (comp['box'][3]-comp['box'][1])**2)
            if dist < box_diag * 0.8:
                comp_legs.append(leg['center'])
        
        if len(comp_legs) < 2:
            x1, y1, x2, y2 = comp['box']
            if (x2-x1) > (y2-y1): 
                comp_legs = [(x1, (y1+y2)/2), (x2, (y1+y2)/2)]
            else: 
                comp_legs = [((x1+x2)/2, y1), ((x1+x2)/2, y2)]
        
        for lx, ly in comp_legs:
            node_id = None
            if ly < top_rail_y or ly > bottom_rail_y:
                node_id = "VCC_RAIL"
            else:
                col_index = int(lx / PROXIMITY_THRESHOLD) 
                node_id = f"Col_{col_index}"
            comp['connected_nodes'].add(node_id)

    # 5. 전류 흐름 시뮬레이션
    changed = True
    while changed:
        changed = False
        for comp in components:
            if comp['is_active']: continue
            if not comp['connected_nodes'].isdisjoint(active_nodes):
                comp['is_active'] = True
                new_nodes = comp['connected_nodes'] - active_nodes
                if new_nodes:
                    active_nodes.update(new_nodes)
                    changed = True

    # 6. 결과 시각화 (여기서 박스를 줄입니다!)
    summary = {'total': 0, 'on': 0, 'off': 0, 'details': {}}
    
    for comp in components:
        name = comp['name']
        x1, y1, x2, y2 = map(int, comp['box'])
        
        # [NEW] 시각화용 박스 축소 (저항/커패시터 몸통 위주로 보이게)
        # 가로가 길면 좌우를 쳐내고, 세로가 길면 위아래를 쳐냄
        w = x2 - x1
        h = y2 - y1
        shrink_factor = 0.4  # 박스 크기를 40%로 줄임 (중심 기준)

        if 'resistor' in name or 'cap' in name:
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            if w > h: # 가로로 긴 부품
                new_w = w * shrink_factor
                x1 = int(cx - new_w / 2)
                x2 = int(cx + new_w / 2)
            else: # 세로로 긴 부품
                new_h = h * shrink_factor
                y1 = int(cy - new_h / 2)
                y2 = int(cy + new_h / 2)

        if comp['is_active']:
            color = (0, 255, 0)
            status = "ON"
            summary['on'] += 1
        else:
            color = (0, 0, 255)
            status = "OFF"
            summary['off'] += 1
            
        summary['total'] += 1
        
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.putText(img, status, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        base_name = name.split('_')[0]
        summary['details'][base_name] = summary['details'].get(base_name, 0) + 1

    return img, summary
