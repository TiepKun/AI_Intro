def dfs_iterative(graph, start, goal):
    stack = [(start, [start])]
    visited = set()

    while stack:
        current_node, path = stack.pop()
        
        if current_node == goal:
            length = sum(graph[path[i]][path[i+1]] for i in range(len(path)-1))
            return path, length
        
        if current_node not in visited:
            visited.add(current_node)
            for neighbor in graph.get(current_node, {}):
                if neighbor not in visited:
                    stack.append((neighbor, path + [neighbor]))
    
    return [], float('inf')  # Không tìm thấy đường đi
