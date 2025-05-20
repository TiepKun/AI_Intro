# Tạo file run_simulation.py
import os
import cv2
import time
from pathlib import Path
from traffic_light_detector import TrafficLightDetector
from self_driving_taxi import SelfDrivingTaxi

# Đảm bảo thư mục đầu ra tồn tại
OUTPUT_PATH = Path("./output")
os.makedirs(OUTPUT_PATH, exist_ok=True)

def run_simulation(video_path=None):
    """
    Chạy mô phỏng hệ thống xe taxi tự lái.
    
    Args:
        video_path: Đường dẫn đến video đầu vào (hoặc None để sử dụng webcam)
    """
    # Khởi tạo detector và hệ thống xe taxi
    detector = TrafficLightDetector()
    taxi = SelfDrivingTaxi(detector)
    
    # Mở video hoặc webcam
    if video_path:
        cap = cv2.VideoCapture(video_path)
    else:
        cap = cv2.VideoCapture(0)  # Sử dụng webcam mặc định
    
    # Kiểm tra xem video có mở thành công không
    if not cap.isOpened():
        print("Không thể mở video/webcam!")
        return
    
    # Lấy thông tin về frame rate và kích thước của video
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:  # Nếu là webcam, fps có thể bằng 0
        fps = 30
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Khởi tạo VideoWriter để lưu video đầu ra
    output_path = OUTPUT_PATH / "taxi_simulation.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec cho video
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    print(f"Bắt đầu mô phỏng. Nhấn 'q' để thoát.")
    
    frame_count = 0
    
    while True:
        # Đọc frame từ video
        ret, frame = cap.read()
        
        # Nếu video kết thúc, thoát vòng lặp
        if not ret:
            break
        
        # Xử lý frame
        if frame_count % 5 == 0:  # Xử lý mỗi 5 frame để tăng hiệu suất
            # Xử lý frame và đưa ra quyết định
            decision, annotated_frame = taxi.process_frame(frame)
            
            # Hiển thị quyết định lái xe
            decision_text = f"Decision: {decision}"
            cv2.putText(annotated_frame, decision_text, (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(annotated_frame, decision_text, (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            
            # Lưu frame vào video đầu ra
            out.write(annotated_frame)
            
            # Hiển thị frame
            cv2.imshow('Self-driving Taxi Simulation', annotated_frame)
        else:
            # Hiển thị frame gốc nếu không xử lý
            cv2.imshow('Self-driving Taxi Simulation', frame)
            out.write(frame)
        
        # Tăng bộ đếm frame
        frame_count += 1
        
        # Kiểm tra phím 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Giải phóng tài nguyên
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"Mô phỏng đã kết thúc. Video đầu ra được lưu tại: {output_path}")

if __name__ == "__main__":
    # Chạy mô phỏng với webcam
    run_simulation(None)
    
    # Hoặc chạy với video (uncomment dòng dưới)
    # video_path = "./data/traffic_video.mp4"
    # run_simulation(video_path)