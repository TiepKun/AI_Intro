# Cập nhật file video_analysis.py với các detector cải tiến
import cv2
import time
import argparse
import os
from pathlib import Path
from traffic_light_detector import TrafficLightDetector
from road_sign_detector import RoadSignDetector
from self_driving_taxi import SelfDrivingTaxi

def analyze_video(input_video, output_video, use_signs=True, show_preview=True, process_ratio=1, debug_mode=False):
    """
    Phân tích video đường xá và phát hiện đèn giao thông và biển báo với các detector cải tiến.
    
    Args:
        input_video: Đường dẫn đến video đầu vào
        output_video: Đường dẫn lưu video đầu ra
        use_signs: Có sử dụng detector biển báo hay không
        show_preview: Có hiển thị preview khi xử lý hay không
        process_ratio: Tỷ lệ xử lý frames (1 = tất cả frames, 2 = 1/2 frames, 3 = 1/3 frames, v.v.)
        debug_mode: Bật/tắt chế độ debug để hiển thị thông tin chi tiết
    """
    # Kiểm tra file đầu vào tồn tại
    if not os.path.exists(input_video):
        print(f"Không tìm thấy file video đầu vào: {input_video}")
        return
    
    print(f"Đang mở file video: {input_video}")
    
    # Đảm bảo thư mục đầu ra tồn tại
    output_dir = os.path.dirname(output_video)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Khởi tạo các detector với ngưỡng thấp hơn
    print("Đang khởi tạo các detector cải tiến...")
    traffic_detector = TrafficLightDetector(conf_threshold=0.4)
    
    if use_signs:
        sign_detector = RoadSignDetector(conf_threshold=0.5)
        taxi = SelfDrivingTaxi(traffic_detector, sign_detector)
        print("Đã khởi tạo detector đèn giao thông và biển báo cải tiến!")
    else:
        taxi = SelfDrivingTaxi(traffic_detector)
        print("Đã khởi tạo detector đèn giao thông cải tiến!")
    
    # Bật chế độ debug nếu cần
    taxi.debug_mode = debug_mode
    
    # Mở video đầu vào
    print(f"Đang mở video đầu vào: {input_video}")
    cap = cv2.VideoCapture(input_video)
    
    if not cap.isOpened():
        print("Không thể mở video đầu vào!")
        return
    
    # Lấy thông tin về video
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Thông tin video: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Khởi tạo VideoWriter để lưu video đầu ra
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec cho video
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    # Biến đếm và thời gian
    frame_count = 0
    processed_count = 0
    start_time = time.time()
    last_fps_update = time.time()
    processing_fps = 0
    
    print(f"Bắt đầu xử lý video. Phím tắt:")
    print("- 'q': Dừng xử lý")
    print("- 's': Lưu ảnh hiện tại")
    print("- 'd': Bật/tắt chế độ debug")
    print("- '1': Bật/tắt detector đèn giao thông")
    print("- '2': Bật/tắt detector biển báo")
    
    while True:
        # Đọc frame từ video
        ret, frame = cap.read()
        
        # Nếu video kết thúc, thoát vòng lặp
        if not ret:
            break
        
        # Chỉ xử lý theo tỷ lệ process_ratio
        if frame_count % process_ratio == 0:
            # Xử lý frame
            decision, annotated_frame = taxi.process_frame(frame)
            
            # Tính FPS xử lý
            processed_count += 1
            if time.time() - last_fps_update > 1.0:
                processing_fps = processed_count / (time.time() - last_fps_update)
                processed_count = 0
                last_fps_update = time.time()
            
            # Chỉ hiển thị thông tin tiến trình đơn giản
            progress = frame_count / total_frames * 100 if total_frames > 0 else 0
            
            # Thanh tiến trình ở dưới cùng
            cv2.rectangle(annotated_frame, (0, height - 30), (int(progress * width / 100), height), (0, 255, 0), -1)
            cv2.rectangle(annotated_frame, (0, height - 30), (width, height), (0, 0, 0), 1)
            
            # Hiển thị tiến trình 
            cv2.rectangle(annotated_frame, (0, height - 30), (width, height), (0, 0, 0), -1)  # Nền đen
            progress_text = f"Progress: {progress:.1f}% | Frame: {frame_count}/{total_frames}"
            cv2.putText(annotated_frame, progress_text, (10, height - 10), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)
            
            # Lưu frame vào video đầu ra
            out.write(annotated_frame)
            
            # Hiển thị frame nếu cần
            if show_preview:
                cv2.imshow('Video Analysis', annotated_frame)
                
                # Kiểm tra phím nhấn
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Lưu ảnh hiện tại
                    img_path = f"./output/frame_{frame_count}.jpg"
                    cv2.imwrite(img_path, annotated_frame)
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
        else:
            # Với các frame không xử lý, chỉ ghi lại frame gốc
            out.write(frame)
        
        # Tăng bộ đếm frame
        frame_count += 1
        
        # In tiến trình mỗi 100 frames
        if frame_count % 100 == 0:
            progress = frame_count / total_frames * 100 if total_frames > 0 else 0
            elapsed = time.time() - start_time
            eta = (elapsed / frame_count) * (total_frames - frame_count) if frame_count > 0 else 0
            
            print(f"Đã xử lý: {frame_count}/{total_frames} frames ({progress:.1f}%) | "
                  f"Thời gian đã trôi qua: {elapsed/60:.1f} phút | "
                  f"ETA: {eta/60:.1f} phút")
    
    # Giải phóng tài nguyên
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    total_time = time.time() - start_time
    print(f"Xử lý video hoàn tất! Thời gian: {total_time/60:.1f} phút")
    print(f"Video đầu ra được lưu tại: {output_video}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Phân tích video đường xá với hệ thống phát hiện cải tiến')
    parser.add_argument('--input', type=str, required=True,
                       help='Đường dẫn đến video đầu vào')
    parser.add_argument('--output', type=str, default='./output/analyzed_video.mp4',
                       help='Đường dẫn lưu video đầu ra')
    parser.add_argument('--signs', action='store_true',
                       help='Sử dụng detector biển báo cùng với đèn giao thông')
    parser.add_argument('--no-preview', action='store_true',
                       help='Không hiển thị preview khi xử lý')
    parser.add_argument('--ratio', type=int, default=2,
                       help='Tỷ lệ xử lý frames (1=tất cả, 2=1/2 frames, 3=1/3 frames, v.v.)')
    parser.add_argument('--debug', action='store_true',
                       help='Bật chế độ debug để hiển thị thông tin chi tiết')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"ERROR: Input file {args.input} does not exist!")
        print("Please provide a valid video file path.")
        exit(1)
    
    print("===== Phân tích video đường xá với detector cải tiến =====")
    print(f"Video đầu vào: {args.input}")
    print(f"Video đầu ra: {args.output}")
    print(f"Detector: {'Đèn giao thông và Biển báo' if args.signs else 'Chỉ đèn giao thông'}")
    print(f"Preview: {'Tắt' if args.no_preview else 'Bật'}")
    print(f"Tỷ lệ xử lý: 1/{args.ratio} frames")
    print(f"Debug mode: {'ON' if args.debug else 'OFF'}")
    print("====================================")
    
    analyze_video(args.input, args.output, args.signs, not args.no_preview, args.ratio, args.debug)