from osmnx.distance import nearest_nodes
from Dijkstra import dijkstra
from A_star import astar
import matplotlib.pyplot as plt
import osmnx as ox
import networkx as nx
center_point = (21.00538744581636, 105.834666695635)
G = ox.graph_from_point(center_point, dist=3000, network_type='drive')
# Chuyển G sang dạng dict trọng số theo 'length'
graph = {node: {} for node in G.nodes()}
for u, v, data in G.edges(data=True):
        length = data.get('length', 1)
        graph[u][v] = min(length, graph[u].get(v, float('inf')))

# Lấy tọa độ node để tính heuristic
positions = {node: (data['x'], data['y']) for node, data in G.nodes(data=True)}

# Tìm node gần nhất với 2 điểm
orig_point = (21.00538744581636, 105.84546706679828)
dest_point = (21.001962, 105.840989)

orig_node = nearest_nodes(G, X=orig_point[1], Y=orig_point[0])
dest_node = nearest_nodes(G, X=dest_point[1], Y=dest_point[0])
#orig_node = ox.distance.nearest_nodes(G, X=orig_point[1], Y=orig_point[0])
#dest_node = ox.distance.nearest_nodes(G, X=dest_point[1], Y=dest_point[0])
# Gọi thuật toán 
#dijkstra
#path = dijkstra(graph, orig_node, dest_node)
# Chạy A*
path,length = astar(graph, orig_node, dest_node, positions)

#path = nx.dijkstra_path(G, orig_node, dest_node, weight='length')
#length, path = nx.single_source_dijkstra(G, orig_node, dest_node, weight='length')
print("Đường đi ngắn nhất (node):", path)
print(f"Tổng chiều dài đường đi: {length:.2f} mét")
fig, ax = ox.plot_graph_route(G, path, route_linewidth=4, node_size=0, bgcolor='w', show=False, close=False)
# Ghi nhãn
x_start, y_start = G.nodes[orig_node]['x'], G.nodes[orig_node]['y']
x_end, y_end = G.nodes[dest_node]['x'], G.nodes[dest_node]['y']
ax.text(x_start, y_start, 'Start', fontsize=20, color='green', weight='bold')
ax.text(x_end, y_end, 'Goal', fontsize=20, color='red', weight='bold')
plt.show()
