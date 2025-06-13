import heapq

def dijkstra(graph, start, goal):
        queue = [(0, start)]
        distances = {node: float('inf') for node in graph}
        distances[start] = 0
        came_from = {}
       #Truy ngược đường đi
        while queue:
              cost, current = heapq.heappop(queue)
              if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1], distances[goal]  
              #xét hàng xóm
              for neighbor in graph.get(current, {}):
                new_cost = cost + graph[current][neighbor]
                if new_cost < distances[neighbor]:
                   distances[neighbor] = new_cost
                   came_from[neighbor] = current
                   heapq.heappush(queue, (new_cost, neighbor))

        return [], 0
