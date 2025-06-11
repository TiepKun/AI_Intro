from collections import deque

def bfs(graph, start, goal):
   
    if start not in graph or goal not in graph:
        return [], float('inf')
    
    if start == goal:
        return [start], 0
    
    # Queue cho BFS - lưu tuple (node, path_to_node, distance_to_node)
    queue = deque([(start, [start], 0)])
    visited = set([start])
    
    while queue:
        current, path, distance = queue.popleft()
        
        # Duyệt các node kề
        for neighbor in graph.get(current, {}):
            if neighbor not in visited:
                new_path = path + [neighbor]
                new_distance = distance + graph[current][neighbor]
                
                # Nếu tìm thấy đích
                if neighbor == goal:
                    return new_path, new_distance
                
                # Thêm vào queue để tiếp tục tìm kiếm
                queue.append((neighbor, new_path, new_distance))
                visited.add(neighbor)
    
    # Không tìm thấy đường đi
    return [], float('inf')

def bfs_all_paths(graph, start, goal, max_depth=None):
    
    if start not in graph or goal not in graph:
        return []
    
    if start == goal:
        return [([start], 0)]
    
    all_paths = []
    queue = deque([(start, [start], 0, 0)])  # (node, path, distance, depth)
    
    while queue:
        current, path, distance, depth = queue.popleft()
        
        # Kiểm tra giới hạn độ sâu
        if max_depth and depth >= max_depth:
            continue
            
        for neighbor in graph.get(current, {}):
            # Tránh cycle (không quay lại node đã đi qua)
            if neighbor not in path:
                new_path = path + [neighbor]
                new_distance = distance + graph[current][neighbor]
                
                if neighbor == goal:
                    all_paths.append((new_path, new_distance))
                else:
                    queue.append((neighbor, new_path, new_distance, depth + 1))
    
    return all_paths

def bfs_shortest_distance(graph, start):
    
    if start not in graph:
        return {}
    
    distances = {start: 0}
    queue = deque([start])
    
    while queue:
        current = queue.popleft()
        current_distance = distances[current]
        
        for neighbor in graph.get(current, {}):
            edge_weight = graph[current][neighbor]
            new_distance = current_distance + edge_weight
            
            # Nếu chưa thăm hoặc tìm thấy đường ngắn hơn
            if neighbor not in distances or new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                queue.append(neighbor)
    
    return distances