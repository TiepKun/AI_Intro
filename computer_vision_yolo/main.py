# main.py - Enhanced Traffic Detection - Phiên bản đơn giản
import cv2
import os
import numpy as np
from ultralytics import YOLO
from pathlib import Path

# Import từ các file hiện tại
from config import *
from utils import draw_boxes, get_files_in_folder

class VNSignsDetector:
    """Class phát hiện biển báo VN"""
    
    def __init__(self, model_path=None):
        self.model = None
        self.is_loaded = False
        
        # Classes biển báo VN (52 classes) - KHÔNG DẤU
        self.vn_classes = {
            0: 'w224',
            1: 'w205c',
            2: 'p102',
            3: 'r302a',
            4: 'w205a',
            5: 'w207',
            6: 'w201a',
            7: 'p123a',
            8: 'i434a',
            9: 'r303',
            10: 'p130',
            11: 'i409',
            12: 'r415a',
            13: 'w245a',
            14: 'p106axe_tai',
            15: 'w203c',
            16: 'p117',
            17: 'p124a',
            18: 'p107',
            19: 'p124d',
            20: 'p103a',
            21: 'w203b',
            22: 'w221b',
            23: 'p111',
            24: 'p129',
            25: 's505axe_may',
            26: 'w246a',
            27: 'w225',
            28: 's505axe_tai_va_cong',
            29: 'p104',
            30: 's505axe_tai',
            31: 'camera',
            32: 'p123b',
            33: 'w202b',
            34: 'b8a',
            35: 'p137',
            36: 'p139',
            37: 'w205b',
            38: 'p12750',
            39: 'p12760',
            40: 'p12780',
            41: 'p12740',
            42: 'r301e',
            43: 'w239b',
            44: 'w233',
            45: 'i407a',
            46: 'p131a',
            47: 'p124b1',
            48: 'w210',
            49: 'p124c',
            50: 'w201b',
            51: 'w246c'
        }
        
        # Màu sắc theo loại biển báo
        self.sign_colors = {
            'prohibition': [2, 7, 10, 14, 17, 18, 19, 20, 23, 29, 32, 34, 35, 36, 46, 47, 49],
            'warning': [0, 1, 4, 5, 6, 15, 16, 21, 22, 26, 27, 33, 43, 44, 48, 50, 51],
            'mandatory': [3, 9, 11, 12, 25, 30, 42, 45],
            'information': [8, 13, 24, 28, 31, 37],
            'speed_limit': [38, 39, 40, 41]
        }
        
        self.color_map = {
            'prohibition': (0, 0, 255),     # Đỏ - Biển cấm
            'warning': (0, 165, 255),       # Cam - Biển cảnh báo
            'mandatory': (255, 0, 0),       # Xanh dương - Biển hiệu lệnh
            'information': (0, 255, 0),     # Xanh lá - Biển chỉ dẫn
            'speed_limit': (0, 255, 255),   # Vàng - Biển tốc độ
            'default': (128, 128, 128)      # Xám
        }
        
        if model_path and os.path.exists(model_path):
            try:
                self.model = YOLO(model_path)
                self.is_loaded = True
                print(f"✅ Đã load model biển báo VN")
            except Exception as e:
                print(f"❌ Lỗi load model VN: {e}")
        else:
            print("⚠️ Model biển báo VN chưa có (sẽ chỉ phát hiện phương tiện)")
    
    def detect(self, image, conf_threshold=0.25):
        """Phát hiện biển báo VN"""
        if not self.is_loaded:
            return []
        
        try:
            results = self.model(image, conf=conf_threshold, iou=0.45)
            return results
        except Exception as e:
            print(f"❌ Lỗi phát hiện biển báo VN: {e}")
            return []
    
    def get_sign_category(self, class_id):
        """Xác định loại biển báo"""
        for category, class_ids in self.sign_colors.items():
            if class_id in class_ids:
                return category
        return 'default'
    
    def get_sign_color(self, class_id):
        """Lấy màu cho biển báo"""
        category = self.get_sign_category(class_id)
        return self.color_map[category]
    
    def draw_vn_signs(self, image, vn_results):
        """Vẽ biển báo VN lên ảnh"""
        for result in vn_results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    
                    # Lấy thông tin biển báo
                    sign_name = self.vn_classes.get(class_id, f'VN_Sign_{class_id}')
                    box_color = self.get_sign_color(class_id)
                    category = self.get_sign_category(class_id)
                    
                    # Rút gọn tên nếu quá dài và làm sạch encoding
                    if len(sign_name) > 25:
                        sign_name = sign_name[:22] + "..."
                    
                    # Đảm bảo encoding an toàn
                    try:
                        sign_name = sign_name.encode('ascii', 'ignore').decode('ascii')
                    except:
                        sign_name = f'sign_class_{class_id}'
                    
                    # Vẽ bounding box dày hơn cho biển báo
                    cv2.rectangle(image, (x1, y1), (x2, y2), box_color, 3)
                    
                    # Tạo label ngắn gọn - KHÔNG DẤU
                    cat_short = {
                        'prohibition': 'CAM',
                        'warning': 'CANH_BAO', 
                        'mandatory': 'HIEU_LENH',
                        'information': 'CHI_DAN',
                        'speed_limit': 'TOC_DO'
                    }.get(category, category.upper())
                    
                    label = f'[{cat_short}] {sign_name} {confidence:.2f}'
                    
                    # Tính kích thước label
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    
                    # Vẽ nền label
                    cv2.rectangle(image, (x1, y1-label_size[1]-12), 
                                (x1+label_size[0]+8, y1), box_color, -1)
                    
                    # Chọn màu text phù hợp
                    text_color = (255, 255, 255) if sum(box_color) < 400 else (0, 0, 0)
                    
                    # Vẽ text
                    cv2.putText(image, label, (x1+4, y1-6), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
        
        return image

class TrafficDetector:
    """Class tích hợp phát hiện phương tiện + biển báo VN"""
    
    def __init__(self):
        print("🚀 Đang khởi tạo Traffic Detector...")
        
        # Load model YOLO gốc cho phương tiện
        if not os.path.exists(MODEL_PATH):
            print(f"📥 Đang tải model {MODEL_NAME}...")
        
        self.model = YOLO(MODEL_PATH)
        print("✅ Model phương tiện đã sẵn sàng!")
        
        # Load model biển báo VN
        vn_model_path = os.path.join(MODEL_DIR, 'vn_traffic_signs.pt')
        self.vn_detector = VNSignsDetector(vn_model_path)
    
    def detect_image(self, image_path, save_result=True):
        """Phát hiện ảnh tích hợp"""
        print(f"🔍 Đang xử lý ảnh: {os.path.basename(image_path)}")
        
        # Đọc ảnh
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Không thể đọc ảnh: {image_path}")
            return None
        
        print(f"📏 Kích thước ảnh: {image.shape[1]}x{image.shape[0]}")
        
        # 1. Phát hiện phương tiện và đèn giao thông
        print("🚗 Đang phát hiện phương tiện và đèn giao thông...")
        vehicle_results = self.model(image, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD)
        
        # 2. Phát hiện biển báo VN (nếu có model)
        vn_results = []
        if self.vn_detector.is_loaded:
            print("🚦 Đang phát hiện biển báo VN...")
            vn_results = self.vn_detector.detect(image)
        
        # 3. Vẽ kết quả tích hợp
        result_image = self.draw_all_results(image.copy(), vehicle_results, vn_results)
        
        # 4. Thống kê
        self.print_stats(vehicle_results, vn_results)
        
        # 5. Lưu kết quả
        if save_result:
            output_path = os.path.join(OUTPUT_DIR, 'images', f'detected_{os.path.basename(image_path)}')
            cv2.imwrite(output_path, result_image)
            print(f"💾 Đã lưu kết quả: {output_path}")
        
        return result_image
    
    def draw_all_results(self, image, vehicle_results, vn_results):
        """Vẽ tất cả kết quả lên ảnh"""
        
        # 1. Vẽ vehicles và traffic lights
        image = draw_boxes(image, vehicle_results)
        
        # 2. Vẽ biển báo VN
        if vn_results and len(vn_results) > 0:
            image = self.vn_detector.draw_vn_signs(image, vn_results)
        
        return image
    
    def print_stats(self, vehicle_results, vn_results):
        """In thống kê ngắn gọn"""
        
        # Đếm phương tiện
        vehicle_count = {}
        traffic_lights = 0
        
        for result in vehicle_results:
            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    if class_id == 9:  # traffic light
                        traffic_lights += 1
                    else:
                        class_name = TRAFFIC_CLASSES.get(class_id, f'object_{class_id}')
                        vehicle_count[class_name] = vehicle_count.get(class_name, 0) + 1
        
        # Đếm biển báo VN (chỉ khi có model)
        vn_sign_count = 0
        vn_categories = {}
        
        if vn_results and self.vn_detector.is_loaded:
            for result in vn_results:
                if result.boxes is not None:
                    for box in result.boxes:
                        class_id = int(box.cls[0])
                        category = self.vn_detector.get_sign_category(class_id)
                        vn_categories[category] = vn_categories.get(category, 0) + 1
                        vn_sign_count += 1
        
        # In kết quả ngắn gọn
        total_vehicles = sum(vehicle_count.values())
        
        if self.vn_detector.is_loaded:
            print(f"\n📊 KẾT QUẢ: {total_vehicles} phương tiện, {traffic_lights} đèn, {vn_sign_count} biển báo VN")
        else:
            print(f"\n📊 KẾT QUẢ: {total_vehicles} phương tiện, {traffic_lights} đèn giao thông")
        
        if vehicle_count:
            vehicles_str = ", ".join([f"{v}({c})" for v, c in vehicle_count.items()])
            print(f"🚗 Phương tiện: {vehicles_str}")
        
        if vn_categories:
            signs_str = ", ".join([f"{cat}({count})" for cat, count in vn_categories.items()])
            print(f"🚦 Biển báo: {signs_str}")
        elif not self.vn_detector.is_loaded:
            print("🚦 Biển báo VN: Model chưa có (chỉ phát hiện phương tiện)")
    
    def detect_video(self, video_path, save_result=True):
        """Phát hiện video tích hợp"""
        print(f"🎥 Đang xử lý video: {os.path.basename(video_path)}")
        
        # Mở video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Không thể mở video: {video_path}")
            return
        
        # Lấy thông tin video
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📹 Video: {width}x{height}, {fps}fps, {total_frames} frames")
        
        # Thiết lập video writer
        if save_result:
            output_path = os.path.join(OUTPUT_DIR, 'videos', f'detected_{os.path.basename(video_path)}')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if frame_count % 30 == 0:  # In progress mỗi 30 frames
                print(f"⏳ Xử lý: {frame_count}/{total_frames} frames ({frame_count/total_frames*100:.1f}%)")
            
            # Chạy detection (giảm frequency để tăng tốc)
            if frame_count % 2 == 0:  # Detect mỗi 2 frames
                vehicle_results = self.model(frame, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD)
                
                vn_results = []
                if self.vn_detector.is_loaded:
                    vn_results = self.vn_detector.detect(frame, 0.3)
                
                frame = self.draw_all_results(frame, vehicle_results, vn_results)
            
            # Lưu frame
            if save_result:
                out.write(frame)
            
            # Hiển thị (nhỏ để tăng tốc)
            display_frame = cv2.resize(frame, (640, 480))
            cv2.imshow('Video Processing', display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("⏹️ Dừng xử lý video (nhấn q)")
                break
        
        # Dọn dẹp
        cap.release()
        if save_result:
            out.release()
            print(f"💾 Đã lưu video kết quả: {output_path}")
        
        cv2.destroyAllWindows()
        print("✅ Xử lý video hoàn thành!")
    
    def detect_webcam(self):
        """Phát hiện webcam real-time"""
        print("🎥 Đang mở webcam...")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Không thể mở webcam!")
            return
        
        # Set resolution để tăng tốc
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("✅ Webcam sẵn sàng!")
        print("🎮 Điều khiển: 'q'=thoát, 's'=save ảnh, 'r'=reset")
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Detect mỗi 3 frames để tăng tốc
            if frame_count % 3 == 0:
                # Phát hiện phương tiện
                vehicle_results = self.model(frame, conf=0.4, iou=0.5)
                
                # Phát hiện biển báo VN (nếu có)
                vn_results = []
                if self.vn_detector.is_loaded:
                    vn_results = self.vn_detector.detect(frame, 0.4)
                
                frame = self.draw_all_results(frame, vehicle_results, vn_results)
            
            # Hiển thị FPS
            cv2.putText(frame, f'Frame: {frame_count}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Enhanced Webcam Detection', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                save_path = os.path.join(OUTPUT_DIR, 'images', f'webcam_{frame_count}.jpg')
                cv2.imwrite(save_path, frame)
                print(f"💾 Đã lưu: {save_path}")
            elif key == ord('r'):
                frame_count = 0
                print("🔄 Reset frame counter")
        
        cap.release()
        cv2.destroyAllWindows()
        print("👋 Đã thoát webcam")

def main():
    """Main function đơn giản"""
    detector = TrafficDetector()
    
    while True:
        print("\n" + "="*50)
        print("🚦 TRAFFIC DETECTION - VIETNAM ENHANCED")
        print("="*50)
        print("1. Phát hiện ảnh")
        print("2. Phát hiện video") 
        print("3. Webcam real-time")
        print("4. Xử lý batch ảnh")
        print("0. Thoát")
        print("-"*50)
        
        choice = input("👉 Chọn chức năng (0-4): ").strip()
        
        if choice == '1':
            # Phát hiện ảnh đơn
            image_path = input("📁 Đường dẫn ảnh: ").strip()
            if os.path.exists(image_path):
                result = detector.detect_image(image_path)
                if result is not None:
                    # Resize để hiển thị nếu quá lớn
                    h, w = result.shape[:2]
                    if max(h, w) > 1000:
                        scale = 1000 / max(h, w)
                        new_w, new_h = int(w * scale), int(h * scale)
                        result = cv2.resize(result, (new_w, new_h))
                    
                    cv2.imshow('Detection Result', result)
                    print("👁️ Nhấn phím bất kỳ để đóng...")
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
            else:
                print("❌ File không tồn tại!")
        
        elif choice == '2':
            # Phát hiện video
            video_path = input("📁 Đường dẫn video: ").strip()
            if os.path.exists(video_path):
                detector.detect_video(video_path)
            else:
                print("❌ File không tồn tại!")
        
        elif choice == '3':
            # Webcam real-time
            detector.detect_webcam()
        
        elif choice == '4':
            # Batch ảnh
            folder_path = input("📁 Đường dẫn thư mục ảnh: ").strip()
            if os.path.exists(folder_path):
                image_files = get_files_in_folder(folder_path, ['.jpg', '.jpeg', '.png', '.bmp'])
                if image_files:
                    print(f"🔍 Tìm thấy {len(image_files)} ảnh")
                    
                    for i, img_path in enumerate(image_files, 1):
                        print(f"\n[{i}/{len(image_files)}] {os.path.basename(img_path)}")
                        detector.detect_image(img_path)
                    
                    print("✅ Xử lý batch hoàn thành!")
                else:
                    print("❌ Không tìm thấy ảnh nào!")
            else:
                print("❌ Thư mục không tồn tại!")
        
        elif choice == '0':
            print("👋 Tạm biệt!")
            break
        
        else:
            print("❌ Lựa chọn không hợp lệ!")

if __name__ == "__main__":
    main()