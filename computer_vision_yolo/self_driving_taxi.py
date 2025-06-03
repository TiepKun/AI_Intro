# Cập nhật file self_driving_taxi.py để sử dụng các detector cải tiến
import cv2
import numpy as np
from traffic_light_detector import TrafficLightDetector
from road_sign_detector import RoadSignDetector
from road_object_detector import RoadObjectDetector

class SelfDrivingTaxi:
    """Mô phỏng hệ thống xe taxi tự lái với phát hiện đa đối tượng."""
    
    def __init__(self, traffic_light_detector=None, road_sign_detector=None, road_object_detector=None):
        """
        Khởi tạo xe taxi tự lái.
        
        Args:
            traffic_light_detector: Instance của TrafficLightDetector
            road_sign_detector: Instance của RoadSignDetector
            road_object_detector: Instance của RoadObjectDetector
        """
        # Khởi tạo các detector
        self.traffic_detector = traffic_light_detector if traffic_light_detector else TrafficLightDetector(conf_threshold=0.3)
        self.sign_detector = road_sign_detector if road_sign_detector else RoadSignDetector(conf_threshold=0.7)
        self.object_detector = road_object_detector if road_object_detector else RoadObjectDetector(conf_threshold=0.35)
        
        self.speed = 0  # Tốc độ hiện tại (0-100)
        self.status = "stopped"  # Trạng thái xe (stopped, moving, stopping)
        self.steering_angle = 0  # Góc lái (-45 đến 45 độ, 0 là đi thẳng)
        
        # Lưu trữ các phát hiện gần đây
        self.recent_traffic_lights = []
        self.recent_signs = []
        self.recent_objects = []
        self.collision_risks = []
        
        # Thêm cờ để bật/tắt nhận diện từng loại đối tượng
        self.enable_traffic_lights = True
        self.enable_road_signs = True
        self.enable_road_objects = True
        
        # Thêm biến để lưu trữ các đối tượng hiện tại trên màn hình
        self.current_traffic_lights = []
        self.current_road_signs = []
        self.current_road_objects = []
        
        # Thêm biến debug
        self.debug_mode = False
        
        print("Self-driving Taxi đã được khởi tạo với đầy đủ các hệ thống phát hiện cải tiến!")
    
    def process_frame(self, frame):
        """
        Xử lý frame từ camera và đưa ra quyết định lái xe.
        
        Args:
            frame: Frame hình ảnh từ camera
            
        Returns:
            decision: Quyết định lái xe
            annotated_frame: Frame với thông tin kết quả
        """
        # Tạo bản sao của frame để vẽ kết quả
        annotated_frame = frame.copy()
        
        # 1. Phát hiện đèn giao thông
        if self.enable_traffic_lights:
            traffic_detections, traffic_frame = self.traffic_detector.detect(frame)
            self.recent_traffic_lights = traffic_detections
            self.current_traffic_lights = traffic_detections
        else:
            traffic_detections = []
            traffic_frame = frame.copy()
        
        # 2. Phát hiện biển báo
        if self.enable_road_signs:
            sign_detections, sign_frame = self.sign_detector.detect(frame)
            self.recent_signs = sign_detections
            self.current_road_signs = sign_detections
        else:
            sign_detections = []
            sign_frame = frame.copy()
        
        # 3. Phát hiện đối tượng trên đường
        if self.enable_road_objects:
            object_detections, object_frame = self.object_detector.detect(frame)
            self.recent_objects = object_detections
            self.current_road_objects = object_detections
            
            # 4. Ước lượng rủi ro va chạm
            self.collision_risks = self.object_detector.estimate_collision_risk(
                object_detections, frame.shape[1], frame.shape[0])
        else:
            object_detections = []
            object_frame = frame.copy()
            self.collision_risks = []
        
        # 5. Đưa ra quyết định lái xe dựa trên tất cả các phát hiện
        decision, steering = self._make_driving_decision()
        
        # 6. Cập nhật trạng thái xe
        self._update_state(decision, steering)
        
        # 7. Kết hợp tất cả các kết quả phát hiện vào một frame
        combined_frame = self._combine_detection_results(
            frame, traffic_frame, sign_frame, object_frame)
        
        # 8. Thêm thông tin về trạng thái của xe
        self._add_status_to_frame(combined_frame)
        
        # 9. Thêm thông tin debug nếu cần
        if self.debug_mode:
            self._add_debug_info(combined_frame)
        
        return decision, combined_frame
    
    def _make_driving_decision(self):
        """
        Đưa ra quyết định lái xe dựa trên tất cả các phát hiện.
        
        Returns:
            decision: Quyết định lái xe ('accelerate', 'maintain', 'slow_down', 'stop')
            steering: Góc lái (-45 đến 45 độ)
        """
        # Khởi tạo với quyết định mặc định
        decision = 'maintain'
        steering = 0
        
        # 1. Ưu tiên rủi ro va chạm cao
        high_risks = [risk for risk in self.collision_risks if risk['risk_level'] == 'high']
        if high_risks:
            # Nếu có rủi ro cao, dừng lại hoặc giảm tốc
            decision = 'stop'
            
            # Tính toán góc lái để tránh va chạm
            # (Đơn giản hóa: nếu đối tượng ở bên trái, lái sang phải và ngược lại)
            for risk in high_risks:
                object_id = risk['object_id']
                for det in self.recent_objects:
                    if id(det) == object_id:
                        x1, y1, x2, y2 = det['bbox']
                        obj_center_x = (x1 + x2) // 2
                        frame_center_x = self.traffic_detector.model.stride  # Giả sử center là stride từ model
                        
                        if obj_center_x < frame_center_x:
                            steering = 25  # Tăng góc lái từ 20 lên 25
                        else:
                            steering = -25  # Tăng góc lái từ -20 lên -25
                        break
        
        # 2. Nếu không có rủi ro cao, kiểm tra đèn giao thông
        elif self.current_traffic_lights:
            # Xem xét đèn giao thông gần nhất/có kích thước lớn nhất
            closest_light = max(self.current_traffic_lights, 
                               key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]))
            
            # Quyết định dựa trên trạng thái đèn
            if closest_light['state'] == 'red':
                decision = 'stop'
            elif closest_light['state'] == 'yellow':
                decision = 'slow_down'
            elif closest_light['state'] == 'green':
                decision = 'accelerate'
            # Thêm xử lý cho đèn số
            elif 'type' in closest_light and closest_light['type'] == 'digital':
                # Với đèn số, thường chỉ hiển thị một màu, dựa vào giá trị state đã phát hiện
                if closest_light['state'] == 'red':
                    decision = 'stop'
                elif closest_light['state'] == 'green':
                    decision = 'accelerate'
                elif closest_light['state'] == 'yellow':
                    decision = 'slow_down'
                else:
                    # Nếu không xác định được rõ ràng, an toàn là giảm tốc
                    decision = 'slow_down'
        
        # 3. Nếu không có rủi ro cao và đèn giao thông, kiểm tra biển báo
        elif self.current_road_signs:
            for sign in self.current_road_signs:
                sign_type = sign['type']
                
                # Xử lý các loại biển báo khác nhau
                if 'stop' in sign_type or 'red_octagon' in sign_type:
                    decision = 'stop'
                    break
                elif 'speed' in sign_type or 'limit' in sign_type or 'red_circle' in sign_type:
                    # Giả sử biển báo tốc độ giới hạn hoặc biển cấm
                    decision = 'slow_down'
                    break
                elif 'yield' in sign_type or 'yellow_triangle' in sign_type:
                    decision = 'slow_down'
                    break
                elif 'blue_direction' in sign_type or 'blue_sign' in sign_type:
                    # Biển chỉ dẫn - có thể duy trì tốc độ hoặc tăng tốc nhẹ
                    decision = 'maintain'
                    break
        
        # 4. Nếu có vật thể trên đường nhưng không có rủi ro cao
        elif self.recent_objects:
            medium_risks = [risk for risk in self.collision_risks if risk['risk_level'] == 'medium']
            if medium_risks:
                # Nếu có rủi ro trung bình, giảm tốc
                decision = 'slow_down'
                
                # Tính toán góc lái để tránh va chạm nhẹ
                for risk in medium_risks:
                    object_id = risk['object_id'] 
                    for det in self.recent_objects:
                        if id(det) == object_id:
                            x1, y1, x2, y2 = det['bbox']
                            obj_center_x = (x1 + x2) // 2
                            frame_center_x = self.traffic_detector.model.stride
                            
                            if obj_center_x < frame_center_x:
                                steering = 15  # Tăng từ 10 lên 15
                            else:
                                steering = -15  # Tăng từ -10 lên -15
                            break
        
        # 5. Nếu không có gì, duy trì tốc độ và đi thẳng
        else:
            decision = 'maintain'
            steering = 0
        
        return decision, steering
    
    def _update_state(self, decision, steering):
        """
        Cập nhật trạng thái và tốc độ của xe dựa trên quyết định.
        
        Args:
            decision: Quyết định lái xe
            steering: Góc lái
        """
        # Cập nhật tốc độ dựa trên quyết định
        if decision == 'stop':
            self.speed = max(0, self.speed - 15)  # Tăng giảm tốc từ 10 lên 15
            self.status = "stopping" if self.speed > 0 else "stopped"
        elif decision == 'slow_down':
            self.speed = max(20, self.speed - 8)  # Tăng giảm tốc từ 5 lên 8
            self.status = "slowing down"
        elif decision == 'accelerate':
            self.speed = min(80, self.speed + 8)  # Tăng tốc tối đa từ 60 lên 80
            self.status = "moving"
        elif decision == 'maintain':
            # Tự duy trì tốc độ ở mức hợp lý nếu đang dừng
            if self.speed < 30:
                self.speed = min(40, self.speed + 5)
            self.status = "moving" if self.speed > 0 else "stopped"
        
        # Cập nhật góc lái
        self.steering_angle = steering
    
    def _combine_detection_results(self, original_frame, traffic_frame, sign_frame, object_frame):
        """
        Kết hợp kết quả phát hiện từ tất cả các detector vào một frame.
        
        Args:
            original_frame: Frame gốc
            traffic_frame: Frame với kết quả phát hiện đèn giao thông
            sign_frame: Frame với kết quả phát hiện biển báo
            object_frame: Frame với kết quả phát hiện đối tượng
            
        Returns:
            combined_frame: Frame kết hợp tất cả các kết quả
        """
        # Lấy các phần phát hiện từ các frame và đưa vào frame gốc
        # Điều này cho phép chúng ta thấy tất cả các phát hiện cùng một lúc
        
        # Cách đơn giản: Lấy tất cả bounding boxes và vẽ lại vào frame gốc
        combined_frame = original_frame.copy()
        
        # Vẽ đèn giao thông
        for det in self.current_traffic_lights:
            x1, y1, x2, y2 = det['bbox']
            state = det['state']
            light_type = det.get('type', 'traditional')
            
            # Chọn màu dựa trên trạng thái
            if state == 'red':
                color = (0, 0, 255)  # BGR: Red
            elif state == 'yellow':
                color = (0, 255, 255)  # BGR: Yellow
            elif state == 'green':
                color = (0, 255, 0)  # BGR: Green
            else:
                color = (255, 255, 255)  # BGR: White (unknown)
            
            # Vẽ bounding box với độ dày lớn hơn cho đèn số
            thickness = 3 if light_type == 'digital' else 2
            cv2.rectangle(combined_frame, (x1, y1), (x2, y2), color, thickness)
            
            # Hiển thị trạng thái
            if light_type == 'digital':
                label = f"ĐÈN SỐ: {state.upper()}"
                # Thêm giá trị đèn số nếu có
                if 'value' in det:
                    label += f" ({det['value']})"
            else:
                label = f"ĐÈN GT: {state.upper()}"
                
            # Thêm nền đen cho text để dễ đọc
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(combined_frame, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), (0, 0, 0), -1)
            
            cv2.putText(combined_frame, label, (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Vẽ biển báo
        for det in self.current_road_signs:
            x, y, w, h = det['bbox']
            sign_type = det['type']
            
            # Chọn màu cho biển báo
            if 'red' in sign_type:
                color = (0, 0, 255)  # Red
            elif 'yellow' in sign_type:
                color = (0, 165, 255)  # Orange
            elif 'blue' in sign_type:
                color = (255, 0, 0)  # Blue
            else:
                color = (255, 255, 255)  # White
            
            # Vẽ bounding box
            cv2.rectangle(combined_frame, (x, y), (x + w, y + h), color, 2)
            
            # Hiển thị loại biển báo với nền đen để dễ đọc
            if 'blue_direction' in sign_type or 'blue_rectangle' in sign_type:
                label = "BIỂN CHỈ DẪN"
            elif 'red_circle' in sign_type:
                label = "BIỂN CẤM"
            elif 'yellow_triangle' in sign_type:
                label = "BIỂN CẢNH BÁO"
            elif 'red_octagon' in sign_type:
                label = "BIỂN STOP"
            else:
                label = "BIỂN BÁO"
                
            # Thêm nền đen cho text
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(combined_frame, (x, y - text_size[1] - 5), (x + text_size[0], y), (0, 0, 0), -1)
            
            cv2.putText(combined_frame, label, (x, y - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Vẽ đối tượng đường và rủi ro va chạm
        for det in self.current_road_objects:
            x1, y1, x2, y2 = det['bbox']
            obj_type = det['type']
            
            # Tìm rủi ro va chạm tương ứng với đối tượng này
            risk = next((r for r in self.collision_risks if r['object_id'] == id(det)), None)
            
            # Chọn màu dựa trên mức độ rủi ro
            if risk and risk['risk_level'] == 'high':
                color = (0, 0, 255)  # Red for high risk
                label = obj_type  # Chỉ hiển thị loại đối tượng khi có rủi ro cao
                
                # Vẽ bounding box
                cv2.rectangle(combined_frame, (x1, y1), (x2, y2), color, 2)
                
                # Thêm nền đen cho text
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(combined_frame, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), (0, 0, 0), -1)
                
                # Vẽ text màu trắng đơn giản
                cv2.putText(combined_frame, label, (x1, y1 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return combined_frame
    
    def _add_status_to_frame(self, frame):
        """
        Thêm thông tin trạng thái của xe vào frame.
        
        Args:
            frame: Frame hình ảnh
        """
        # Thông tin cơ bản: số lượng đối tượng phát hiện
        detection_text = f"Detected: {len(self.current_traffic_lights)} lights, {len(self.current_road_signs)} signs, {len(self.current_road_objects)} objects"
        
        # Thêm bảng thông tin ở góc trên bên trái với màu nền đen
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)  # Nền đen full width
        
        # Vẽ text trắng
        cv2.putText(frame, detection_text, (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
    
    def _add_debug_info(self, frame):
        """
        Thêm thông tin debug vào frame.
        
        Args:
            frame: Frame hình ảnh
        """
        # Thêm thông tin về ngưỡng confidence
        debug_text1 = f"Thresholds: Traffic={self.traffic_detector.conf_threshold:.2f}, Sign={self.sign_detector.conf_threshold:.2f}, Object={self.object_detector.conf_threshold:.2f}"
        cv2.putText(frame, debug_text1, (10, frame.shape[0] - 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(frame, debug_text1, (10, frame.shape[0] - 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Thông tin về phương pháp phát hiện
        debug_text2 = f"Digital Light Detection: {'ON' if hasattr(self.traffic_detector, 'detect_digital_lights') and self.traffic_detector.detect_digital_lights else 'OFF'}"
        cv2.putText(frame, debug_text2, (10, frame.shape[0] - 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(frame, debug_text2, (10, frame.shape[0] - 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Thêm thông tin về điều kiện phát hiện
        debug_text3 = f"Detection: Traffic={self.enable_traffic_lights}, Signs={self.enable_road_signs}, Objects={self.enable_road_objects}"
        cv2.putText(frame, debug_text3, (10, frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(frame, debug_text3, (10, frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

# Hàm test đơn giản để kiểm tra taxi với tất cả các hệ thống phát hiện
def test_advanced_taxi():
    import cv2
    import time
    
    # Khởi tạo các detector với ngưỡng thấp hơn
    traffic_detector = TrafficLightDetector(conf_threshold=0.4)
    sign_detector = RoadSignDetector(conf_threshold=0.5)
    object_detector = RoadObjectDetector(conf_threshold=0.4)
    
    # Khởi tạo taxi với tất cả các detector
    taxi = SelfDrivingTaxi(traffic_detector, sign_detector, object_detector)
    
    # Bật chế độ debug
    taxi.debug_mode = True
    
    # Mở webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Không thể mở webcam!")
        return
    
    print("Đang test taxi tự lái nâng cao với webcam. Nhấn 'q' để thoát, 'd' để bật/tắt debug, số 1-3 để bật/tắt detector.")
    
    # Khởi tạo biến đếm FPS
    fps_start_time = time.time()
    fps_counter = 0
    fps = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Đếm FPS
        fps_counter += 1
        if (time.time() - fps_start_time) > 1:
            fps = fps_counter
            fps_counter = 0
            fps_start_time = time.time()
        
        # Xử lý frame và lấy quyết định
        decision, annotated_frame = taxi.process_frame(frame)
        
        # Hiển thị quyết định lái xe
        decision_text = f"Decision: {decision} | FPS: {fps}"
        cv2.putText(annotated_frame, decision_text, (10, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(annotated_frame, decision_text, (10, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        # Hiển thị kết quả
        cv2.imshow('Advanced Self-driving Taxi', annotated_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
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
        elif key == ord('3'):
            # Bật/tắt phát hiện đối tượng
            taxi.enable_road_objects = not taxi.enable_road_objects
            print(f"Road object detection: {'ON' if taxi.enable_road_objects else 'OFF'}")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_advanced_taxi()