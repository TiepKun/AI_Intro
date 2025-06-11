import streamlit as st
import folium
from streamlit_folium import st_folium
import osmnx as ox
import networkx as nx
import heapq
from Dijkstra import dijkstra
from A_star import astar
from BFS1 import bfs
from DFS import dfs_iterative
import time
st.set_page_config(layout="centered")
st.title("🚗 Tìm đường ngắn nhất")

# Input tọa độ
#col1, col2 = st.columns(2)
#with col1:
#    start_lat = st.number_input("Start Latitude", value=21.005387)
#    start_lon = st.number_input("Start Longitude", value=105.845467)
#with col2:
#    end_lat = st.number_input("End Latitude", value=21.037147)
#    end_lon = st.number_input("End Longitude", value=105.834666)
# Tọa độ các địa điểm
# ==========================
# 1. CHỌN CHẾ ĐỘ NHẬP
# ==========================
mode = st.radio("Chọn cách nhập điểm bắt đầu và kết thúc", ["Chọn từ danh sách", "Nhập tọa độ thủ công"])

# Tọa độ địa danh có sẵn
places = {
    "Đại học Bách Khoa Hà Nội": (21.005382677407056, 105.84541903995671),
    "Bệnh viện Bạch Mai": (21.001962, 105.840989),
    "Lăng Chủ tịch Hồ Chí Minh": (21.035480, 105.834167),
    "Hồ Gươm": (21.028708, 105.851313),
    "NEU": (21.000394667084407, 105.84252065330566),
    "Ng. Tự Do": (20.9987525070749, 105.84727359563394),
    "Chợ Đồng Xuân": (21.034682, 105.847419),
    "Văn Miếu": (21.027680, 105.835800),
}
# ==========================
# 2. NHẬP ĐIỂM BẮT ĐẦU VÀ KẾT THÚC
# ==========================
if mode == "Chọn từ danh sách":
    start_place = st.selectbox("Chọn điểm xuất phát", list(places.keys()), index=0)
    end_place = st.selectbox("Chọn điểm đến", list(places.keys()), index=3)
    start_lat, start_lon = places[start_place]
    end_lat, end_lon = places[end_place]
    st.write(f"📍 **Start:** {start_place} ({start_lat}, {start_lon})")
    st.write(f"🏁 **End:** {end_place} ({end_lat}, {end_lon})")
else:
    col1, col2 = st.columns(2)
    with col1:
        start_lat = st.number_input("Nhập Start Latitude", value=21.005382,format="%.6f")
        start_lon = st.number_input("Nhập Start Longitude", value=105.845419,format="%.6f")
    with col2:
        end_lat = st.number_input("Nhập End Latitude", value=21.037147,format="%.6f")
        end_lon = st.number_input("Nhập End Longitude", value=105.834666,format="%.6f")
# Sidebar để chọn thuật toán
st.sidebar.header("Chọn thuật toán")
algorithm = st.sidebar.selectbox(
    "Thuật toán tìm đường:",
    ["A* (A-Star)", "Dijkstra", "BFS (Breadth-First Search)", "DFS (Depth-First Search)"],
    index=0
)

# ==========================
# 3. TÌM ĐƯỜNG ĐI
# ==========================
# Khởi tạo session_state để lưu path và center_point
if "results" not in st.session_state:
    st.session_state.results = {}
if "center_point" not in st.session_state:
    st.session_state.center_point = None
def run_algorithm(graph, orig_node, dest_node, positions, algo_name):
    """Chạy thuật toán và đo thời gian"""
    start_time = time.time()
    
    if algo_name == "A* (A-Star)":
        path, length = astar(graph, orig_node, dest_node, positions)
    elif algo_name == "Dijkstra":
        path,length = dijkstra(graph, orig_node, dest_node)
    elif algo_name == "BFS (Breadth-First Search)":
        path, length = bfs(graph, orig_node, dest_node)
    elif algo_name == "DFS (Depth-First Search)":
        path, length = dfs_iterative(graph, orig_node, dest_node)
    
    end_time = time.time()
    execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
    
    return path, length, execution_time


if st.button("Tìm đường đi", type="primary"):
   with st.spinner("Đang tính toán..."):
    center_point = ((start_lat + end_lat) / 2, (start_lon + end_lon) / 2)
    G = ox.graph_from_point(center_point, dist=6000, network_type='drive')

    orig_node = ox.distance.nearest_nodes(G, start_lon, start_lat)
    dest_node = ox.distance.nearest_nodes(G, end_lon, end_lat)

    graph = {node: {} for node in G.nodes()}
    for u, v, data in G.edges(data=True):
        length = data.get('length', 1)
        graph[u][v] = min(length, graph[u].get(v, float('inf')))

    positions = {node: (data['x'], data['y']) for node, data in G.nodes(data=True)}

     # Chạy thuật toán được chọn
    path, length, exec_time = run_algorithm(graph, orig_node, dest_node, positions, algorithm)
    if not path:
        st.error("Không tìm thấy đường đi.")
        st.session_state.path = None
        st.session_state.center_point = None
    else:
        st.session_state.results = {
                'path': path,
                'length': length,
                'time': exec_time,
                'algorithm': algorithm,
                'G': G
            }
        st.session_state.center_point = center_point
        st.success(f" Tìm đường đi bằng {algorithm}!")
# ==========================
# 4. HIỂN THỊ BẢN ĐỒ
# ==========================
# Hiển thị kết quả
if st.session_state.results and st.session_state.center_point:
    results = st.session_state.results
    
    # Thông tin chi tiết
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Độ dài đường đi", f"{results['length']:.2f} m")
    with col2:
        st.metric("Thời gian tính toán", f"{results['time']:.2f} ms")
    with col3:
        st.metric("Số điểm trên đường", len(results['path']))
    
    # Hiển thị bản đồ
    st.subheader(f"🗺️ Đường đi tìm được bằng {results['algorithm']}")
    
    G = results['G']
    path = results['path']
    coords = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in path]
    
    # Tạo bản đồ với màu sắc khác nhau cho từng thuật toán
    color_map = {
        "A* (A-Star)": "red",
        "Dijkstra": "blue", 
        "BFS (Breadth-First Search)": "green",
        "DFS (Depth-First Search)": "purple"
    }
    
    m = folium.Map(location=st.session_state.center_point, zoom_start=13)
    
    # Marker điểm đầu và cuối
    folium.Marker(
        location=(start_lat, start_lon), 
        popup=f"Xuất phát: {start_place}", 
        icon=folium.Icon(color="green", icon="play")
    ).add_to(m)
    
    folium.Marker(
        location=(end_lat, end_lon), 
        popup=f"Đích: {end_place}", 
        icon=folium.Icon(color="red", icon="stop")
    ).add_to(m)
    
    # Đường đi
    route_color = color_map.get(results['algorithm'], "blue")
    folium.PolyLine(
        locations=coords, 
        color=route_color, 
        weight=4, 
        opacity=0.8,
        popup=f"{results['algorithm']}: {results['length']:.2f}m"
    ).add_to(m)
    
    st_folium(m, width=700, height=500)