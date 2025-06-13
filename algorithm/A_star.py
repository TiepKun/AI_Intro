import heapq
import math
# A* heuristic
def heuristic(a, b):
    #return ((a[0]-b[0])**2 + (a[1]-b[1])**2) ** 0.5
    return 1.5 * math.hypot(a[0] - b[0], a[1] - b[1])

def astar(graph, start, goal, positions):
    queue = []
    heapq.heappush(queue, (0, start))
    came_from = {}
    g_score = {node: float('inf') for node in graph}
    g_score[start] = 0
    f_score = {node: float('inf') for node in graph}
    f_score[start] = heuristic(positions[start], positions[goal])
    #Truy ngược đường đi
    while queue:
        _, current = heapq.heappop(queue)
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1] ,g_score[goal]
        #xét hàng xóm
        for neighbor in graph.get(current, {}):
            tentative = g_score[current] + graph[current][neighbor]
            if tentative < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative
                f_score[neighbor] = tentative + heuristic(positions[neighbor], positions[goal])
                heapq.heappush(queue, (f_score[neighbor], neighbor))
    return [],float('inf')
