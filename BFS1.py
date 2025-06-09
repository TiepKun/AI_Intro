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

# Test function
if __name__ == "__main__":
    # Test graph đơn giản
    test_graph = {
        'A': {'B': 1, 'C': 4},
        'B': {'A': 1, 'C': 2, 'D': 5},
        'C': {'A': 4, 'B': 2, 'D': 1},
        'D': {'B': 5, 'C': 1}
    }
    
    print("Test BFS:")
    path, distance = bfs(test_graph, 'A', 'D')
    print(f"Đường đi từ A đến D: {path}")
    print(f"Tổng khoảng cách: {distance}")
    
    print("\nTất cả đường đi từ A đến D:")
    all_paths = bfs_all_paths(test_graph, 'A', 'D', max_depth=4)
    for i, (path, dist) in enumerate(all_paths, 1):
        print(f"Đường {i}: {' -> '.join(path)} (Khoảng cách: {dist})")