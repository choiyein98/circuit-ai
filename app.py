# ... (상단 Helper 함수들은 동일) ...

def analyze_real_netlist(img, model_list):
    h, w, _ = img.shape
    raw_bodies = []
    raw_legs = [] 

    for model in model_list:
        res = model.predict(source=img, conf=0.05, verbose=False) # 임계값을 0.05로 확 낮춤
        for b in res[0].boxes:
            name = model.names[int(b.cls[0])].lower()
            coords = b.xyxy[0].tolist()
            conf = float(b.conf[0])
            
            # [수정 1] 인식을 위해 최소 conf 기준을 대폭 완화
            if 'cap' in name and conf < 0.05: continue 
            if 'res' in name and conf < 0.10: continue # 0.30 -> 0.10으로 하향
            if 'breadboard' in name or 'hole' in name: continue
            
            center = get_center(coords)
            if any(x in name for x in ['pin', 'leg', 'lead']):
                raw_legs.append({'box': coords, 'center': center})
            elif 'wire' not in name: 
                raw_bodies.append({'name': name, 'box': coords, 'center': center, 'conf': conf})

    parts = solve_overlap(raw_bodies, is_real=True)

    # 2. 노드 클러스터링 (동일)
    grouped_legs = []
    for leg in raw_legs:
        assigned = False
        for group in grouped_legs:
            ref = group[0] 
            if abs(leg['center'][0] - ref['center'][0]) < 30: # 범위를 25 -> 30으로 약간 확장
                group.append(leg); assigned = True; break
        if not assigned: grouped_legs.append([leg])

    # 3. 부품-노드 연결 매핑
    part_connections = defaultdict(set)
    for i, part in enumerate(parts):
        # 이름 정규화
        p_name = part['name'].lower()
        if 'res' in p_name: part['role'] = 'resistor'
        elif 'cap' in p_name: part['role'] = 'capacitor'
        elif any(x in p_name for x in ['source', 'volt', 'batt']): part['role'] = 'source'
        else: part['role'] = p_name

        for nid, group in enumerate(grouped_legs):
            for leg in group:
                dist = math.sqrt((part['center'][0]-leg['center'][0])**2 + (part['center'][1]-leg['center'][1])**2)
                # [수정 2] 핀 탐색 범위를 diag * 1.5로 대폭 확장하여 멀리 있는 핀도 잡음
                if dist < 200: 
                    part_connections[i].add(nid)

    # 4. 흐름 및 관계 도출 (잘못된 연결 체크 로직 포함)
    connections = []
    flow_errors = []
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            p1, p2 = parts[i], parts[j]
            shared_nodes = part_connections[i].intersection(part_connections[j])
            if shared_nodes:
                rel_type = 'Parallel' if len(shared_nodes) >= 2 else 'Series'
                connections.append({'p1': p1['role'], 'p2': p2['role'], 'type': rel_type})

                # [수정 3] 사용자 지적: Capacitor가 Source와 같은 노드면 에러
                if (p1['role'] == 'source' and p2['role'] == 'capacitor') or \
                   (p1['role'] == 'capacitor' and p2['role'] == 'source'):
                    flow_errors.append(f"❌ 흐름 오류: Capacitor가 저항을 거치지 않고 Source(마디 {list(shared_nodes)[0]})에 직접 연결됨")

    return img, {'parts': parts, 'connections': connections, 'errors': flow_errors}
