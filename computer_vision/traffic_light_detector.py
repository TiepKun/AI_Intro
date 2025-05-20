# Cải tiến file traffic_light_detector.py - Nhận diện thêm đèn số
import cv2
import numpy as np
import torch
import time

class TrafficLightDetector:
    """Phát hiện đèn giao thông đơn giản hóa nhưng hiệu quả, hỗ trợ đèn số."""
    
    def __init__(self, model_path=None, conf_threshold=0.3):  # Giảm ngưỡng từ 0.4 xuống 0.3
        """Khởi tạo detector đèn giao thông."""
        # Điều chỉnh ngưỡng confidence để tăng khả năng phát hiện  
        self.conf_threshold = conf_threshold
        
        # Điều chỉnh các tham số để phát hiện đèn ngang và dọc
        self.min_ratio = 0.2  # Giảm mạnh để bắt được đèn ngang (tỷ lệ cao/rộng)
        self.max_ratio = 5.0  # Giữ nguyên cho đèn dọc
        self.min_size = 100   # Giảm kích thước tối thiểu để bắt được đèn xa
        
        # Tắt bộ phát hiện đèn số
        self.detect_digital_lights = False
        
        # Nạp model YOLO
        try:
            if model_path is None:
                print("Đang tải model YOLOv5...")
                self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            else:
                self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
            
            # Cài đặt tham số cho model
            self.model.conf = 0.25  # Giảm ngưỡng YOLO từ 0.4 xuống 0.25
            
            # Lưu trữ các phát hiện gần đây để làm mượt kết quả
            self.recent_detections = []
            self.max_history = 3  # Lưu 3 frame gần nhất
            
            print("Đã tải model thành công!")
        except Exception as e:
            print(f"Lỗi khi tải model: {e}")
            raise e
    
    def detect(self, image):
        """Phát hiện đèn giao thông trong ảnh."""
        try:
            # Tạo bản sao để vẽ kết quả
            annotated_img = image.copy()
            
            # Lấy kích thước ảnh
            height, width = image.shape[:2]
            
            # 1. Loại bỏ phần UI (thường ở 1/6 trên của màn hình) - giảm từ 1/5 xuống 1/6
            roi_height = int(height * 0.15)  # Giảm để có thể bắt được đèn cao hơn
            working_image = image[roi_height:, :]
            
            # Tăng cường chất lượng ảnh
            enhanced_image = self._enhance_image(working_image)
            
            # Bước mới: Phát hiện đèn giao thông số
            digital_detections = []
            if self.detect_digital_lights:
                digital_detections = self._detect_digital_traffic_lights(enhanced_image, roi_height)
            
            # Thực hiện dự đoán YOLO cho đèn truyền thống
            results = self.model(enhanced_image)
            
            # Lọc kết quả YOLO để lấy đèn giao thông
            yolo_detections = []
            
            all_yolo_detections = results.xyxy[0].cpu().numpy()
            
            for i, det in enumerate(all_yolo_detections):
                x1, y1, x2, y2, conf, cls = det
                cls_id = int(cls)
                
                # Phải điều chỉnh lại tọa độ y do cắt ảnh
                y1 = y1 + roi_height
                y2 = y2 + roi_height
                
                # Lọc nhiều loại đối tượng hơn cho đèn giao thông (class 9 và 10)
                # Class 9: Traffic Light, Class 10: Fire Hydrant (đôi khi nhầm với đèn giao thông)
                # Loại bỏ class 0 (Person) vì người không phải đèn giao thông
                if (cls_id == 9) and conf >= self.conf_threshold:
                    w = x2 - x1
                    h = y2 - y1
                    
                    # Nới lỏng điều kiện tỷ lệ cạnh cho đèn giao thông
                    aspect_ratio = h / w
                    if aspect_ratio < 0.1 or aspect_ratio > 8.0:  # Mở rộng từ 0.2-6.0 thành 0.1-8.0
                        continue
                    
                    # THÊM: Loại bỏ vùng trắng đơn thuần
                    gray_roi = cv2.cvtColor(image[int(y1):int(y2), int(x1):int(x2)], cv2.COLOR_BGR2GRAY)
                    if gray_roi.size > 0:
                        _, binary = cv2.threshold(gray_roi, 200, 255, cv2.THRESH_BINARY)
                        white_ratio = cv2.countNonZero(binary) / (w * h)
                        
                        # Nếu >70% vùng là trắng, bỏ qua
                        if white_ratio > 0.7:
                            continue
                    
                    # Nới lỏng điều kiện kích thước
                    area = w * h
                    if area < 100 or area > (width * height * 0.1):  # Tăng từ 8% lên 10%
                        continue
                    
                    # 4. Vùng đèn giao thông
                    roi = image[int(y1):int(y2), int(x1):int(x2)]
                    if roi.size == 0:
                        continue
                    
                    # Nới lỏng điều kiện cấu trúc đèn
                    # Nếu confidence rất cao, bỏ qua kiểm tra cấu trúc
                    if conf > 0.8 or self._check_traffic_light_structure(roi):
                        # 6. Xác định trạng thái đèn
                        state, color_confidence = self._identify_light_state(roi)
                        
                        # 7. Giảm ngưỡng màu từ 0.4 xuống 0.3
                        if color_confidence < 0.3:
                            # Nếu không xác định được màu rõ ràng, đặt là unknown
                            state = "unknown"
                            color_confidence = 0.3  # Ngưỡng tối thiểu
                        
                        # Thêm vào danh sách kết quả
                        yolo_detections.append({
                            'bbox': (int(x1), int(y1), int(x2), int(y2)),
                            'confidence': float(conf),
                            'state': state,
                            'color_confidence': color_confidence,
                            'type': 'traditional'  # Đánh dấu đèn truyền thống
                        })
            
            # Kết hợp các phát hiện từ cả hai phương pháp
            detections = yolo_detections + digital_detections
            
            # Cập nhật lịch sử phát hiện để làm mượt kết quả
            self.recent_detections.append(detections)
            if len(self.recent_detections) > self.max_history:
                self.recent_detections.pop(0)
            
            # Làm mượt kết quả
            smoothed_detections = self._smooth_detections()
            
            # Vẽ kết quả
            for det in smoothed_detections:
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
                
                # Vẽ bounding box
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
                
                # Hiển thị trạng thái và loại đèn
                if light_type == 'digital':
                    label = f"Digital Light: {state.upper()}"
                    # Thêm giá trị đèn số nếu có
                    if 'value' in det:
                        label += f" ({det['value']})"
                else:
                    label = f"Traffic Light: {state.upper()}"
                
                cv2.putText(annotated_img, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            return smoothed_detections, annotated_img
        
        except Exception as e:
            print(f"Lỗi khi phát hiện: {e}")
            import traceback
            traceback.print_exc()
            return [], image.copy()
    
    def _detect_digital_traffic_lights(self, image, y_offset=0):
        """
        Phát hiện đèn giao thông số trong ảnh.
        
        Args:
            image: Ảnh đầu vào
            y_offset: Độ lệch y để điều chỉnh tọa độ
            
        Returns:
            digital_lights: Danh sách các đèn giao thông số đã phát hiện
        """
        digital_lights = []
        
        try:
            # Tăng cường ảnh để nhận diện đèn số tốt hơn
            # 1. Chuyển sang grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 2. Áp dụng ngưỡng thích ứng để tìm vùng có độ tương phản cao
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY_INV, 13, 5)
            
            # 3. Tìm kiếm các contour
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 4. Lọc các contour có hình dạng chữ nhật và kích thước phù hợp
            for contour in contours:
                # Tính diện tích
                area = cv2.contourArea(contour)
                
                # Bỏ qua contour quá nhỏ
                if area < 300:
                    continue
                
                # Lấy bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Tỷ lệ cạnh cho đèn số (thường là chữ nhật ngang)
                aspect_ratio = float(w) / h
                
                # Đèn số thường có tỷ lệ ngang nhiều hơn dọc
                if not (1.5 < aspect_ratio < 6.0):
                    continue
                
                # Kiểm tra xem đây có phải là đèn số không
                is_digital_light, state, value = self._check_digital_light(image[y:y+h, x:x+w])
                
                if is_digital_light:
                    # Điều chỉnh tọa độ y
                    y += y_offset
                    
                    digital_lights.append({
                        'bbox': (x, y, x+w, y+h),
                        'confidence': 0.7,  # Confidence mặc định
                        'state': state,
                        'color_confidence': 0.8,
                        'type': 'digital',
                        'value': value if value else "N/A"
                    })
            
            # Thêm 1 phương pháp khác: Tìm đèn số dựa trên màu sắc đặc trưng
            # Đèn số thường có màu đen nền và đỏ/xanh/vàng số
            
            # Tìm vùng có màu đỏ (đèn đỏ số)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Định nghĩa màu cho đèn số
            # Màu đỏ cho các đèn số
            lower_red1 = np.array([0, 150, 150])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 150, 150])
            upper_red2 = np.array([180, 255, 255])
            
            # Màu xanh lá cho các đèn số
            lower_green = np.array([40, 100, 100])
            upper_green = np.array([90, 255, 255])
            
            # Màu vàng cam cho đèn số
            lower_yellow = np.array([15, 150, 150])
            upper_yellow = np.array([35, 255, 255])
            
            # Tạo mask cho từng màu
            mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask_red = cv2.bitwise_or(mask_red1, mask_red2)
            
            mask_green = cv2.inRange(hsv, lower_green, upper_green)
            mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
            
            # Kết hợp các mask
            color_masks = [
                ("red", mask_red),
                ("green", mask_green),
                ("yellow", mask_yellow)
            ]
            
            for color_name, mask in color_masks:
                # Tìm contour từ mask
                color_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in color_contours:
                    area = cv2.contourArea(contour)
                    if area < 100:  # Bỏ qua vùng màu quá nhỏ
                        continue
                    
                    # Lấy bounding box
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Mở rộng bounding box để bắt được toàn bộ đèn số
                    x_expanded = max(0, x - int(w * 0.5))
                    y_expanded = max(0, y - int(h * 0.2))
                    w_expanded = min(image.shape[1] - x_expanded, int(w * 2.0))
                    h_expanded = min(image.shape[0] - y_expanded, int(h * 1.4))
                    
                    # Kiểm tra xem vùng mở rộng có phải đèn số không
                    roi = image[y_expanded:y_expanded+h_expanded, x_expanded:x_expanded+w_expanded]
                    if roi.size == 0:
                        continue
                    
                    # Kiểm tra tỷ lệ màu đen/nền tối trong ROI
                    is_digital_light, detected_state, value = self._check_digital_light(roi)
                    
                    if is_digital_light:
                        # Nếu màu phát hiện được khác với màu từ mask, ưu tiên màu từ mask
                        state = color_name if color_name in ["red", "green", "yellow"] else detected_state
                        
                        # Điều chỉnh tọa độ y
                        y_expanded += y_offset
                        
                        # Thêm vào danh sách phát hiện
                        digital_lights.append({
                            'bbox': (x_expanded, y_expanded, x_expanded+w_expanded, y_expanded+h_expanded),
                            'confidence': 0.75,
                            'state': state,
                            'color_confidence': 0.8,
                            'type': 'digital',
                            'value': value if value else "N/A"
                        })
            
            return digital_lights
            
        except Exception as e:
            print(f"Lỗi khi phát hiện đèn số: {e}")
            return []
    
    def _check_digital_light(self, roi):
        """
        Kiểm tra xem ROI có phải là đèn giao thông số không.
        
        Args:
            roi: Vùng ảnh cần kiểm tra
            
        Returns:
            is_digital: True nếu là đèn số
            state: Trạng thái đèn (red, green, yellow, unknown)
            value: Giá trị số (nếu đọc được)
        """
        if roi.size == 0:
            return False, "unknown", None
        
        try:
            # Đèn số thường có nền tối và chữ/số sáng có màu
            # 1. Kiểm tra tỷ lệ vùng tối
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
            
            # Đếm số pixel tối và sáng
            dark_pixels = np.sum(binary == 0)
            light_pixels = np.sum(binary == 255)
            total_pixels = roi.shape[0] * roi.shape[1]
            
            dark_ratio = dark_pixels / total_pixels
            
            # Đèn số thường có tỷ lệ tối > 30%
            if dark_ratio < 0.3:
                return False, "unknown", None
            
            # 2. Kiểm tra màu chủ đạo (của vùng sáng)
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # Tạo mask cho vùng sáng
            light_mask = binary.copy()
            
            # Áp dụng mask để chỉ lấy màu của vùng sáng
            masked_hsv = cv2.bitwise_and(hsv, hsv, mask=light_mask)
            
            # Đếm pixel cho từng khoảng màu
            red_mask1 = cv2.inRange(masked_hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
            red_mask2 = cv2.inRange(masked_hsv, np.array([170, 100, 100]), np.array([180, 255, 255]))
            red_count = cv2.countNonZero(cv2.bitwise_or(red_mask1, red_mask2))
            
            green_mask = cv2.inRange(masked_hsv, np.array([40, 100, 100]), np.array([90, 255, 255]))
            green_count = cv2.countNonZero(green_mask)
            
            yellow_mask = cv2.inRange(masked_hsv, np.array([15, 100, 100]), np.array([35, 255, 255]))
            yellow_count = cv2.countNonZero(yellow_mask)
            
            # Xác định màu chủ đạo
            color_counts = {
                "red": red_count,
                "green": green_count,
                "yellow": yellow_count
            }
            
            if max(color_counts.values()) > 0:
                dominant_color = max(color_counts, key=color_counts.get)
                color_ratio = color_counts[dominant_color] / light_pixels if light_pixels > 0 else 0
                
                # Nếu tỷ lệ màu đủ lớn, có thể là đèn số
                if color_ratio > 0.1:  # Hạ ngưỡng xuống 0.1
                    # Thử đọc số từ đèn (sử dụng phương pháp contour đơn giản)
                    # Lưu ý: Đây chỉ là phương pháp sơ bộ, cần OCR thực sự để chính xác
                    value = self._extract_number_from_light(binary)
                    return True, dominant_color, value
            
            # 3. Kiểm tra hình dạng chữ nhật với góc tròn
            # Tính chu vi và xấp xỉ hình dạng
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                peri = cv2.arcLength(largest_contour, True)
                approx = cv2.approxPolyDP(largest_contour, 0.04 * peri, True)
                
                # Đèn số thường có hình dạng chữ nhật với 4-8 điểm (góc tròn)
                if 4 <= len(approx) <= 8:
                    return True, "unknown", None
            
            return False, "unknown", None
            
        except Exception as e:
            print(f"Lỗi khi kiểm tra đèn số: {e}")
            return False, "unknown", None
    
    def _extract_number_from_light(self, binary_image):
        """
        Cố gắng trích xuất số từ đèn giao thông số.
        
        Args:
            binary_image: Ảnh nhị phân của đèn
            
        Returns:
            value: Giá trị số trích xuất được (hoặc None)
        """
        try:
            # Đảo ngược ảnh nhị phân (số sáng trên nền tối -> số tối trên nền sáng)
            inverted = cv2.bitwise_not(binary_image)
            
            # Tìm contour của các ký tự
            contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Nếu số lượng contour quá nhiều hoặc quá ít, có thể không phải số
            if not (1 <= len(contours) <= 3):  # Tối đa 3 chữ số
                return None
            
            # Sắp xếp contour từ trái sang phải
            contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
            
            # Tính tỷ lệ cạnh trung bình
            aspect_ratios = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratios.append(float(h) / w)
            
            avg_aspect_ratio = sum(aspect_ratios) / len(aspect_ratios) if aspect_ratios else 0
            
            # Số thường cao hơn rộng
            if avg_aspect_ratio < 1.0:
                return None
            
            # Nếu tất cả các điều kiện thỏa mãn, đây có thể là đèn số
            # Tuy nhiên, để đọc chính xác số cần OCR, nên ở đây chỉ đánh dấu là có số
            return "Có số"
            
        except Exception as e:
            print(f"Lỗi khi trích xuất số: {e}")
            return None
    
    def _check_traffic_light_structure(self, roi):
        """Kiểm tra xem ROI có cấu trúc đèn giao thông không."""
        try:
            if roi.size == 0:
                return False
                
            # Chuyển sang HSV
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            height, width = roi.shape[:2]
            
            # Chia thành 3 phần: trên, giữa, dưới
            top_region = hsv[0:height//3, :]
            middle_region = hsv[height//3:2*height//3, :]
            bottom_region = hsv[2*height//3:, :]
            
            # Kiểm tra độ tương phản giữa các vùng
            top_avg = np.mean(top_region[:,:,2])  # Độ sáng vùng trên
            middle_avg = np.mean(middle_region[:,:,2])  # Độ sáng vùng giữa
            bottom_avg = np.mean(bottom_region[:,:,2])  # Độ sáng vùng dưới
            
            # Kiểm tra sự khác biệt giữa các vùng
            diff1 = abs(top_avg - middle_avg)
            diff2 = abs(middle_avg - bottom_avg)
            diff3 = abs(top_avg - bottom_avg)
            
            # Giảm ngưỡng khác biệt tối thiểu từ 10 xuống 5
            if max(diff1, diff2, diff3) < 5:
                return False
            
            # Kiểm tra màu sắc đặc trưng của đèn giao thông
            # Đèn giao thông thường có ít nhất một trong 3 màu: đỏ, vàng, xanh
            # Tính số pixel của mỗi màu
            
            # Màu đỏ (2 khoảng trong HSV)
            lower_red1 = np.array([0, 70, 70])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 70, 70])
            upper_red2 = np.array([180, 255, 255])
            
            # Màu vàng
            lower_yellow = np.array([15, 70, 70])
            upper_yellow = np.array([35, 255, 255])
            
            # Màu xanh lá
            lower_green = np.array([35, 70, 70])
            upper_green = np.array([90, 255, 255])
            
            mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
            red_pixels = cv2.countNonZero(cv2.bitwise_or(mask_red1, mask_red2))
            
            mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
            yellow_pixels = cv2.countNonZero(mask_yellow)
            
            mask_green = cv2.inRange(hsv, lower_green, upper_green)
            green_pixels = cv2.countNonZero(mask_green)
            
            total_pixels = roi.shape[0] * roi.shape[1]
            color_ratio = (red_pixels + yellow_pixels + green_pixels) / total_pixels
            
            # Đèn giao thông thường có tỷ lệ màu đặc trưng cao
            if color_ratio > 0.1:  # Giảm ngưỡng từ mặc định xuống 0.1
                return True
            
            # Nếu không đạt yêu cầu màu sắc, kiểm tra hình dạng
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                                      param1=50, param2=30, minRadius=5, maxRadius=30)
                                      
            # Đèn giao thông thường có hình tròn bên trong
            if circles is not None and len(circles[0]) >= 1:
                return True
            
            # Mặc định, trả về False nếu không thỏa mãn các điều kiện
            return False
            
        except Exception as e:
            print(f"Lỗi khi kiểm tra cấu trúc đèn: {e}")
            return False
    
    def _enhance_image(self, image):
        """Tăng cường chất lượng ảnh cho điều kiện ánh sáng kém."""
        try:
            # Tăng độ sáng và tương phản
            alpha = 1.3  # Tăng tương phản từ 1.2 lên 1.3
            beta = 15    # Tăng độ sáng từ 10 lên 15
            enhanced = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
            
            # Cân bằng sáng cho ảnh
            lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            
            # Làm mịn ảnh nhẹ để giảm nhiễu
            enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
            
            return enhanced
        except Exception as e:
            print(f"Lỗi khi tăng cường ảnh: {e}")
            return image
    
    def _identify_light_state(self, light_img):
        """Xác định trạng thái đèn dựa trên màu sắc."""
        try:
            # Tăng cường ảnh đèn
            enhanced = cv2.convertScaleAbs(light_img, alpha=1.3, beta=15)
            
            # Chuyển sang không gian màu HSV
            hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
            
            # Định nghĩa ngưỡng màu rộng hơn cho các điều kiện ánh sáng khác nhau
            # Màu đỏ có 2 khoảng Hue trong HSV
            lower_red1 = np.array([0, 40, 40])       # Giảm ngưỡng saturation và value
            upper_red1 = np.array([15, 255, 255])    # Mở rộng dải màu đỏ
            lower_red2 = np.array([160, 40, 40])     # Giảm ngưỡng saturation và value
            upper_red2 = np.array([180, 255, 255])
            
            # Màu vàng
            lower_yellow = np.array([10, 40, 40])    # Mở rộng dải màu vàng
            upper_yellow = np.array([40, 255, 255])  # Mở rộng để bắt những árng vàng nhạt
            
            # Màu xanh lá
            lower_green = np.array([35, 30, 30])     # Giảm ngưỡng để bắt được xanh nhạt
            upper_green = np.array([95, 255, 255])   # Mở rộng dải màu xanh
            
            # Tạo mask cho từng màu
            mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask_red = cv2.bitwise_or(mask_red1, mask_red2)
            
            mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
            mask_green = cv2.inRange(hsv, lower_green, upper_green)
            
            # Cải thiện mask bằng morphology
            kernel = np.ones((3,3), np.uint8)
            mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
            mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel)
            mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)
            
            # Đếm số pixel của từng màu
            red_count = cv2.countNonZero(mask_red)
            yellow_count = cv2.countNonZero(mask_yellow) 
            green_count = cv2.countNonZero(mask_green)
            
            # Tổng số pixel
            total_pixels = light_img.shape[0] * light_img.shape[1]
            
            # Xác định màu với số pixel nhiều nhất
            color_counts = {
                'red': red_count,
                'yellow': yellow_count,
                'green': green_count
            }
            
            # Tìm màu có nhiều pixel nhất
            max_color = max(color_counts, key=color_counts.get)
            max_count = color_counts[max_color]
            
            # Tính tỉ lệ và độ tin cậy
            ratio = max_count / total_pixels if total_pixels > 0 else 0
            confidence = min(1.0, ratio * 3.5)  # Scale up (tăng độ tin cậy)
            
            # Giảm ngưỡng tối thiểu số pixel
            if max_count < 3 or ratio < 0.01:  # Giảm ngưỡng từ 0.02 xuống 0.01
                return 'unknown', 0.0
            
            return max_color, confidence
                
        except Exception as e:
            print(f"Lỗi khi xác định trạng thái: {e}")
            return 'unknown', 0.0
            
    def _smooth_detections(self):
        """Làm mượt kết quả phát hiện qua nhiều frame."""
        if not self.recent_detections:
            return []
            
        if len(self.recent_detections) == 1:
            return self.recent_detections[0]
            
        # Kết hợp tất cả các khung hình, ưu tiên khung hiện tại
        all_detections = []
        weights = []
        
        # Gán trọng số cho từng khung hình (mới hơn = quan trọng hơn)
        for i, dets in enumerate(self.recent_detections):
            weight = (i + 1) / len(self.recent_detections)
            all_detections.extend(dets)
            weights.extend([weight] * len(dets))
        
        if not all_detections:
            return []
        
        # Nhóm các phát hiện của cùng một đèn giao thông
        grouped_detections = {}
        
        for i, det in enumerate(all_detections):
            x1, y1, x2, y2 = det['bbox']
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            light_type = det.get('type', 'traditional')
            
            # Tạo khóa duy nhất cho mỗi vị trí
            key = f"{center_x // 30}_{center_y // 30}_{light_type}"  # Thêm loại đèn vào key
            
            if key not in grouped_detections:
                grouped_detections[key] = {
                    'dets': [det],
                    'weights': [weights[i]],
                    'states': [det['state']]
                }
            else:
                grouped_detections[key]['dets'].append(det)
                grouped_detections[key]['weights'].append(weights[i])
                grouped_detections[key]['states'].append(det['state'])
        
        # Tạo kết quả cuối cùng
        smoothed_detections = []
        
        for key, group in grouped_detections.items():
            dets = group['dets']
            w = group['weights']
            states = group['states']
            
            # Lấy detection có trọng số cao nhất
            best_idx = w.index(max(w))
            best_det = dets[best_idx].copy()  # Tạo bản sao để tránh thay đổi gốc
            
            # Đếm tần suất của các trạng thái
            state_count = {}
            for i, state in enumerate(states):
                if state not in state_count:
                    state_count[state] = w[i]
                else:
                    state_count[state] += w[i]
            
            # Chọn trạng thái phổ biến nhất
            best_state = max(state_count, key=state_count.get)
            
            # Cập nhật trạng thái
            best_det['state'] = best_state
            
            smoothed_detections.append(best_det)
        
        return smoothed_detections


def test_detector():
    """Test detector với webcam."""
    import cv2
    import time
    
    print("Khởi tạo detector...")
    detector = TrafficLightDetector(conf_threshold=0.4)  # Giảm ngưỡng
    
    print("Đang mở webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Không thể mở webcam!")
        return
    
    print("Đang test detector đèn giao thông. Nhấn 'q' để thoát.")
    
    # Biến tính FPS
    frame_count = 0
    start_time = time.time()
    fps = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Không đọc được frame từ webcam!")
            break
        
        # Tính FPS
        frame_count += 1
        elapsed = time.time() - start_time
        if elapsed >= 1:
            fps = frame_count / elapsed
            frame_count = 0
            start_time = time.time()
            
        # Phát hiện đèn
        detections, annotated_frame = detector.detect(frame)
        
        # Hiển thị thông tin
        text = f"Traffic Lights: {len(detections)} | FPS: {fps:.1f}"
        cv2.putText(annotated_frame, text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(annotated_frame, text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
        
        # Hiển thị kết quả
        cv2.imshow('Traffic Light Detection', annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_detector()