def dfs(graph, start, goal):
    
    if start not in graph or goal not in graph:
        return [], float('inf')
    
    if start == goal:
        return [start], 0
    
    visited = set()
    
    def dfs_recursive(current, path, distance):
        visited.add(current)
        
        # Nếu tìm thấy đích
        if current == goal:
            return path, distance
        
        # Duyệt các node kề
        for neighbor in graph.get(current, {}):
            if neighbor not in visited:
                new_path = path + [neighbor]
                new_distance = distance + graph[current][neighbor]
                
                result_path, result_distance = dfs_recursive(neighbor, new_path, new_distance)
                if result_path:  # Nếu tìm thấy đường đi
                    return result_path, result_distance
        
        visited.remove(current)  # Backtrack
        return [], float('inf')
    
    return dfs_recursive(start, [start], 0)

def dfs_iterative(graph, start, goal):
  
    if start not in graph or goal not in graph:
        return [], float('inf')
    
    if start == goal:
        return [start], 0
    
    # Stack lưu tuple (node, path_to_node, distance_to_node, visited_set)
    stack = [(start, [start], 0, {start})]
    
    while stack:
        current, path, distance, visited = stack.pop()
        
        if current == goal:
            return path, distance
        
        # Duyệt các node kề (ngược lại để giữ thứ tự)
        neighbors = list(graph.get(current, {}).keys())
        for neighbor in reversed(neighbors):
            if neighbor not in visited:
                new_path = path + [neighbor]
                new_distance = distance + graph[current][neighbor]
                new_visited = visited.copy()
                new_visited.add(neighbor)
                
                stack.append((neighbor, new_path, new_distance, new_visited))
    
    return [], float('inf')

def dfs_all_paths(graph, start, goal, max_depth=None):
   
    if start not in graph or goal not in graph:
        return []
    
    if start == goal:
        return [([start], 0)]
    
    all_paths = []
    
    def dfs_recursive(current, path, distance, visited, depth):
        if max_depth and depth > max_depth:
            return
        
        if current == goal:
            all_paths.append((path.copy(), distance))
            return
        
        for neighbor in graph.get(current, {}):
            if neighbor not in visited:
                path.append(neighbor)
                visited.add(neighbor)
                new_distance = distance + graph[current][neighbor]
                
                dfs_recursive(neighbor, path, new_distance, visited, depth + 1)
                
                # Backtrack
                path.pop()
                visited.remove(neighbor)
    
    dfs_recursive(start, [start], 0, {start}, 0)
    return all_paths

def dfs_longest_path(graph, start, goal):
    
    if start not in graph or goal not in graph:
        return [], 0
    
    if start == goal:
        return [start], 0
    
    longest_path = []
    max_distance = 0
    
    def dfs_recursive(current, path, distance, visited):
        nonlocal longest_path, max_distance
        
        if current == goal:
            if distance > max_distance:
                max_distance = distance
                longest_path = path.copy()
            return
        
        for neighbor in graph.get(current, {}):
            if neighbor not in visited:
                path.append(neighbor)
                visited.add(neighbor)
                new_distance = distance + graph[current][neighbor]
                
                dfs_recursive(neighbor, path, new_distance, visited)
                
                # Backtrack
                path.pop()
                visited.remove(neighbor)
    
    dfs_recursive(start, [start], 0, {start})
    return longest_path, max_distance

def dfs_cycle_detection(graph, start):
    
    visited = set()
    rec_stack = set()  # Recursion stack để phát hiện back edge
    
    def dfs_recursive(current, path):
        visited.add(current)
        rec_stack.add(current)
        
        for neighbor in graph.get(current, {}):
            if neighbor not in visited:
                cycle_found, cycle_path = dfs_recursive(neighbor, path + [neighbor])
                if cycle_found:
                    return True, cycle_path
            elif neighbor in rec_stack:
                # Tìm thấy back edge -> có cycle
                cycle_start_idx = path.index(neighbor)
                return True, path[cycle_start_idx:] + [neighbor]
        
        rec_stack.remove(current)
        return False, []
    
    return dfs_recursive(start, [start])