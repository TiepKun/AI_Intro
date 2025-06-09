import osmnx as ox
center_point = (21.00538744581636, 105.834666695635)
G = ox.graph_from_point(center_point, dist=3000, network_type='drive')
# Chuyển G sang dạng dict trọng số theo 'length'
graph = {}
for node in G.nodes():
    graph[node] = {}
for u, v, data in G.edges(data=True):
    length = data.get('length', 1)  # lấy trọng số độ dài, mặc định 1 nếu không có
    # Nếu nhiều cạnh giữa u->v thì ghi cạnh có độ dài nhỏ nhất
    if v not in graph[u] or length < graph[u][v]:
        graph[u][v] = length