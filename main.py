# Cập nhật file main.py với các detector cải tiến
import os
import argparse
import cv2
import time
from pathlib import Path
from traffic_light_detector import TrafficLightDetector
from road_sign_detector import RoadSignDetector
from self_driving_taxi import SelfDrivingTaxi

def run_simulation(video_path=None, output_path="./output/taxi_simulation.mp4", traffic_only=False, debug_mode=False):
    """
    Chạy mô phỏng hệ thống xe taxi tự lái với các detector đã được cải tiến.
    
    Args:
        video_path: Đường dẫn đến video đầu vào (hoặc None để sử dụng webcam)
        output_path: Đường dẫn lưu video đầu ra
        traffic_only: Chỉ sử dụng detector đèn giao thông (không dùng detector biển báo)
        debug_mode: Bật/tắt chế độ debug
    """
    # Đảm bảo thư mục đầu ra tồn tại
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Khởi tạo các detector
    traffic_detector = TrafficLightDetector(conf_threshold=0.3)
    
    if traffic_only:
        taxi = SelfDrivingTaxi(traffic_detector)
        print("Taxi tự lái đã được khởi tạo với hệ thống phát hiện ĐÈN GIAO THÔNG đã cải tiến!")
    else:
        sign_detector = RoadSignDetector(conf_threshold=0.7)  # Tăng ngưỡng confidence cho biển báo
        taxi = SelfDrivingTaxi(traffic_detector, sign_detector)
        print("Taxi tự lái đã được khởi tạo với hệ thống phát hiện ĐÈN GIAO THÔNG và BIỂN BÁO đã cải tiến!")
    
    # Bật chế độ debug nếu được yêu cầu
    taxi.debug_mode = debug_mode
    
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
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec cho video
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Bắt đầu mô phỏng. Phím tắt:")
    print("- 'q': Thoát chương trình")
    print("- 's': Lưu ảnh hiện tại")
    print("- 'd': Bật/tắt chế độ debug")
    print("- '1': Bật/tắt detector đèn giao thông")
    print("- '2': Bật/tắt detector biển báo")
    
    # Biến đếm FPS
    frame_count = 0
    fps_start_time = time.time()
    fps_counter = 0
    fps = 0
    
    # Chế độ skip_frame để tăng tốc xử lý
    skip_frames = 1  # Xử lý mỗi 2 frame (bỏ qua 1)
    
    while True:
        # Đọc frame từ video
        ret, frame = cap.read()
        
        # Nếu video kết thúc, thoát vòng lặp
        if not ret:
            break
        
        # Đếm FPS
        fps_counter += 1
        if (time.time() - fps_start_time) > 1:
            fps = fps_counter
            fps_counter = 0
            fps_start_time = time.time()
        
        # Xử lý frame (chỉ xử lý mỗi skip_frames frame để tăng hiệu suất)
        if frame_count % (skip_frames + 1) == 0:
            # Xử lý frame và đưa ra quyết định
            decision, annotated_frame = taxi.process_frame(frame)
            
            # Thông tin đơn giản về thresholds 
            cv2.putText(annotated_frame, "Traffic=0.30, Sign=0.70", (10, annotated_frame.shape[0] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(annotated_frame, "Traffic=0.30, Sign=0.70", (10, annotated_frame.shape[0] - 10), 
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
        
        # Kiểm tra phím nhấn
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Lưu ảnh hiện tại
            img_path = f"./output/capture_{frame_count}.jpg"
            cv2.imwrite(img_path, annotated_frame if frame_count % (skip_frames + 1) == 0 else frame)
            print(f"Đã lưu ảnh tại: {img_path}")
        elif key == ord('d'):
            # Bật/tắt chế độ debug
            taxi.debug_mode = not taxi.debug_mode
            print(f"Debug mode: {'ON' if taxi.debug_mode else 'OFF'}")
        elif key == ord('1'):
            # Bật/tắt phát hiện đèn giao thông
            taxi.enable_traffic_lights = not taxi.enable_traffic_lights
            print(f"Traffic light detection: {'ON' if taxi.enable_traffic_lights else 'OFF'}")
        elif key == ord('2'):
            # Bật/tắt phát hiện biển báo
            taxi.enable_road_signs = not taxi.enable_road_signs
            print(f"Road sign detection: {'ON' if taxi.enable_road_signs else 'OFF'}")
    
    # Giải phóng tài nguyên
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"Mô phỏng đã kết thúc. Video đầu ra được lưu tại: {output_path}")

def main():
    """Hàm chính để chạy dự án."""
    parser = argparse.ArgumentParser(description='Hệ thống xe taxi tự lái với detector cải tiến')
    parser.add_argument('--video', type=str, default=None,
                       help='Đường dẫn đến video đầu vào (mặc định: sử dụng webcam)')
    parser.add_argument('--output', type=str, default='./output/taxi_simulation.mp4',
                       help='Đường dẫn lưu video đầu ra')
    parser.add_argument('--traffic-only', action='store_true',
                       help='Chỉ sử dụng detector đèn giao thông (không dùng detector biển báo)')
    parser.add_argument('--debug', action='store_true',
                       help='Bật chế độ debug để hiển thị thông tin chi tiết')
    parser.add_argument('--skip', type=int, default=1,
                       help='Số frame bỏ qua giữa mỗi lần xử lý (mặc định: 1, xử lý 1/2 frames)')
    
    args = parser.parse_args()
    
    print("===== Hệ thống xe taxi tự lái với detector cải tiến =====")
    print(f"Video: {'Webcam' if args.video is None else args.video}")
    print(f"Output: {args.output}")
    print(f"Mode: {'Traffic-only' if args.traffic_only else 'Traffic and Signs'}")
    print(f"Debug: {'ON' if args.debug else 'OFF'}")
    print(f"Processing: 1/{args.skip + 1} frames")
    print("============================================")
    
    # Cập nhật biến skip_frames
    global skip_frames
    skip_frames = args.skip
    
    # Chạy mô phỏng
    run_simulation(args.video, args.output, args.traffic_only, args.debug)
    
    print("Chương trình đã kết thúc.")

if __name__ == "__main__":
    main()