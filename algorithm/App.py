import streamlit as st
import folium
from streamlit_folium import st_folium
import osmnx as ox
import heapq
from Dijkstra import dijkstra
from A_star import astar
st.set_page_config(layout="centered")
st.title("🚗 Tìm đường ngắn nhất bằng A*")

# Input tọa độ
#col1, col2 = st.columns(2)
#with col1:
#    start_lat = st.number_input("Start Latitude", value=21.005387)
#    start_lon = st.number_input("Start Longitude", value=105.845467)
#with col2:
#    end_lat = st.number_input("End Latitude", value=21.037147)
#    end_lon = st.number_input("End Longitude", value=105.834666)
# Tọa độ các địa điểm
places = {
    "Đại học Bách Khoa Hà Nội": (21.005382677407056, 105.84541903995671),
    "Bệnh viện Bạch Mai": (21.001962, 105.840989),
    "Lăng Chủ tịch Hồ Chí Minh": (21.035480, 105.834167),
    "Hồ Gươm": (21.028708, 105.851313),
}   
# Chọn điểm start và end
start_place = st.selectbox("Chọn điểm xuất phát", list(places.keys()), index=0)
end_place = st.selectbox("Chọn điểm đến", list(places.keys()), index=3)

start_lat, start_lon = places[start_place]
end_lat, end_lon = places[end_place]

st.write(f"Start: {start_place} tại ({start_lat}, {start_lon})")
st.write(f"End: {end_place} tại ({end_lat}, {end_lon})")

# Khởi tạo session_state để lưu path và center_point
if "path" not in st.session_state:
    st.session_state.path = None
if "center_point" not in st.session_state:
    st.session_state.center_point = None
if "length" not in st.session_state:
    st.session_state.length = 0


if st.button("Tìm đường đi"):
    center_point = ((start_lat + end_lat) / 2, (start_lon + end_lon) / 2)
    G = ox.graph_from_point(center_point, dist=3000, network_type='drive')

    orig_node = ox.distance.nearest_nodes(G, start_lon, start_lat)
    dest_node = ox.distance.nearest_nodes(G, end_lon, end_lat)

    graph = {node: {} for node in G.nodes()}
    for u, v, data in G.edges(data=True):
        length = data.get('length', 1)
        graph[u][v] = min(length, graph[u].get(v, float('inf')))

    positions = {node: (data['x'], data['y']) for node, data in G.nodes(data=True)}

    path,length = astar(graph, orig_node, dest_node, positions)
    #path = dijkstra(graph, orig_node, dest_node)
    if not path:
        st.error("Không tìm thấy đường đi.")
        st.session_state.path = None
        st.session_state.center_point = None
    else:
        st.session_state.path = path
        st.session_state.center_point = center_point
        st.session_state.length = length
     # Hiển thị tổng độ dài
    st.write(f"**Tổng độ dài đường đi:** {st.session_state.length:.2f} mét")
# Nếu có path lưu trong session_state, tạo map và hiển thị
if st.session_state.path and st.session_state.center_point:
    path = st.session_state.path
    center_point = st.session_state.center_point
    G = ox.graph_from_point(center_point, dist=3000, network_type='drive')

    coords = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in path]
    m = folium.Map(location=center_point, zoom_start=14)
    folium.Marker(location=(start_lat, start_lon), popup="Start", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker(location=(end_lat, end_lon), popup="Goal", icon=folium.Icon(color="red")).add_to(m)
    folium.PolyLine(locations=coords, color="blue", weight=5).add_to(m)
    st_folium(m, width=700, height=500)

