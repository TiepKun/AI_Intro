import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt

# Lấy bản đồ đường ở Hà Nội
center_point = (21.00538744581636, 105.834666695635)
print("Đang tải bản đồ...")
G = ox.graph_from_point(center_point, dist=6000, network_type="drive")
print("Tải xong.")

# Tọa độ 2 điểm (lat, lon)
orig_point = (21.00538744581636, 105.84546706679828)   # BKHN
dest_point = (21.03714762423055, 105.834666695635)   # Lăng Bác

# Tìm node gần nhất
orig_node = ox.distance.nearest_nodes(G, X=orig_point[1], Y=orig_point[0])
dest_node = ox.distance.nearest_nodes(G, X=dest_point[1], Y=dest_point[0])
print(f"Bắt đầu từ node {orig_node}, đến node {dest_node}")
# Tìm đường đi
shortest_path = nx.dijkstra_path(G, orig_node, dest_node, weight='length')
print(f"Đường đi có {len(shortest_path)} điểm.")

# Vẽ bản đồ
fig, ax = ox.plot_graph_route(G, shortest_path, route_linewidth=4, node_size=0, bgcolor='w')
plt.show()

input("Nhấn Enter để thoát...")
