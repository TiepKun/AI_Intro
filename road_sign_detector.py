# Cải tiến file road_sign_detector.py - Nhận diện nhiều loại biển báo hơn
import cv2
import numpy as np
import math

class RoadSignDetector:
    """Phát hiện biển báo giao thông với thuật toán được cải tiến."""
    
    def __init__(self, conf_threshold=0.7):  # Tăng ngưỡng từ 0.6 lên 0.7
        """
        Khởi tạo detector biển báo giao thông.
        
        Args:
            conf_threshold: Ngưỡng confidence để phát hiện
        """
        self.conf_threshold = conf_threshold
        
        # Lưu trữ phát hiện qua nhiều frame để giảm false positives
        self.history = []
        self.max_history = 3
        
        # Lưu trữ kết quả cuối cùng
        self.last_detections = []
        
        print("Road Sign Detector đã được khởi tạo với thuật toán mới cải tiến!")
        
    def detect(self, image):
        """
        Phát hiện biển báo giao thông trong ảnh.
        
        Args:
            image: Ảnh đầu vào (định dạng OpenCV)
            
        Returns:
            detections: List các biển báo đã phát hiện
            annotated_img: Ảnh với kết quả phát hiện
        """
        # Tạo bản sao ảnh để xử lý và hiển thị
        frame = image.copy()
        annotated_img = image.copy()
        
        # Phát hiện các biển báo tiềm năng
        candidate_signs = self._detect_road_signs(frame)
        
        # Lọc và xác nhận các biển báo thực sự
        verified_signs = self._verify_road_signs(frame, candidate_signs)
        
        # Cập nhật lịch sử phát hiện
        self.history.append(verified_signs)
        if len(self.history) > self.max_history:
            self.history.pop(0)
        
        # Lọc phát hiện qua nhiều frame để giảm false positives
        self.last_detections = self._temporal_filtering()
        
        # Vẽ kết quả lên ảnh
        for sign in self.last_detections:
            x, y, w, h = sign['bbox']
            sign_type = sign['type']
            confidence = sign['confidence']
            
            # Màu dựa trên loại biển báo
            if 'red' in sign_type:
                color = (0, 0, 255)  # BGR: Red
            elif 'blue' in sign_type:
                color = (255, 0, 0)  # BGR: Blue
            elif 'yellow' in sign_type:
                color = (0, 255, 255)  # BGR: Yellow
            else:
                color = (255, 255, 255)  # BGR: White
            
            # Vẽ bounding box
            cv2.rectangle(annotated_img, (x, y), (x + w, y + h), color, 2)
            
            # Hiển thị loại biển báo và độ tin cậy
            if 'circle' in sign_type:
                label = "ROUND"
            elif 'triangle' in sign_type:
                label = "TRIANGLE"
            elif 'rectangle' in sign_type or 'square' in sign_type:
                label = "RECTANGLE"
            elif 'octagon' in sign_type:
                label = "OCTAGON"
            elif 'arrow' in sign_type:
                label = "ARROW"
            else:
                label = "SIGN"
            
            conf_text = f"{confidence:.2f}"
            cv2.putText(annotated_img, label, (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(annotated_img, conf_text, (x, y + h + 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return self.last_detections, annotated_img
    
    def _detect_road_signs(self, image):
        """
        Phát hiện các biển báo tiềm năng dựa trên màu sắc và hình dạng.
        
        Args:
            image: Ảnh đầu vào
            
        Returns:
            candidate_signs: Danh sách các biển báo tiềm năng
        """
        candidate_signs = []
        
        # Tiền xử lý ảnh để giảm nhiễu và tăng cường màu sắc
        # Giảm nhiễu bằng Gaussian blur nhẹ
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        
        # Tăng cường tương phản
        lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))  # Tăng clipLimit
        cl = clahe.apply(l)
        enhanced = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # Chuyển sang không gian màu HSV
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
        
        # Định nghĩa các ngưỡng màu cho biển báo
        # Màu đỏ (2 khoảng trong HSV) - Mở rộng dải màu
        lower_red1 = np.array([0, 70, 70])  # Giảm ngưỡng saturation và value
        upper_red1 = np.array([15, 255, 255])  # Mở rộng khoảng hue
        lower_red2 = np.array([160, 70, 70])  # Giảm ngưỡng saturation và value
        upper_red2 = np.array([180, 255, 255])
        
        # Màu xanh dương cho biển báo chỉ dẫn - Mở rộng dải màu
        lower_blue = np.array([90, 80, 80])  # Mở rộng dải màu xanh
        upper_blue = np.array([150, 255, 255])  # Bắt được nhiều sắc thái xanh hơn
        
        # Màu vàng cho biển báo cảnh báo - Mở rộng dải màu
        lower_yellow = np.array([15, 100, 100])  # Giảm ngưỡng
        upper_yellow = np.array([35, 255, 255])  # Mở rộng khoảng hue
        
        # Tạo mask cho từng màu
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Ứng dụng các phép toán hình thái học để làm sạch mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
        
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel)
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)
        
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, kernel)
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel)
        
        # Xử lý từng mask màu và tìm contours
        masks = [
            ("red", mask_red),
            ("blue", mask_blue),
            ("yellow", mask_yellow)
        ]
        
        for color_name, mask in masks:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Bỏ qua contour quá nhỏ - tránh nhiễu
                area = cv2.contourArea(contour)
                if area < 150:  # Giảm ngưỡng diện tích tối thiểu từ 200 xuống 150
                    continue
                
                # Tính chu vi và lấy hình dạng xấp xỉ
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
                
                # Lấy bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # THÊM: Loại bỏ vùng nghi ngờ là lá cây
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                
                # Màu lá cây (xanh lá cây)
                lower_green_leaf = np.array([35, 80, 40])  # Màu xanh lá cây tự nhiên
                upper_green_leaf = np.array([90, 255, 255])
                
                # Tạo mask cho lá cây
                mask_green_leaf = cv2.inRange(hsv, lower_green_leaf, upper_green_leaf)
                
                # Kiểm tra tỷ lệ màu xanh lá cây tự nhiên trong vùng này
                roi_green = mask_green_leaf[y:y+h, x:x+w]
                
                if roi_green.size > 0:
                    green_ratio = cv2.countNonZero(roi_green) / (w * h)
                    
                    # Nếu tỷ lệ xanh lá cây tự nhiên cao, có thể là lá cây, không phải biển báo
                    if green_ratio > 0.3:  # Nếu >30% vùng này là xanh lá cây tự nhiên
                        continue
                
                # THÊM: Loại bỏ vùng nghi ngờ là phần đường hoặc vạch kẻ đường
                # Vạch kẻ đường thường có màu trắng/xám trên nền tối
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                roi_gray = gray[y:y+h, x:x+w]
                
                if roi_gray.size > 0:
                    # Tính tỷ lệ điểm sáng (vạch kẻ đường)
                    _, thresh_gray = cv2.threshold(roi_gray, 180, 255, cv2.THRESH_BINARY)
                    white_ratio = cv2.countNonZero(thresh_gray) / (w * h)
                    
                    # Nếu tỷ lệ điểm trắng cao và hình dạng dài, có thể là vạch kẻ đường
                    if white_ratio > 0.4 and (w > 3*h or h > 3*w):  # Vạch kẻ đường thường dài và mỏng
                        continue
                
                # Tính các thông số hình học
                aspect_ratio = float(w) / h
                extent = float(area) / (w * h)  # Tỷ lệ diện tích contour so với bounding box
                
                # Tính circularity (độ tròn)
                circularity = 4 * np.pi * area / (peri * peri) if peri > 0 else 0
                
                # Tính convexity (độ lõm)
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                convexity = float(area) / hull_area if hull_area > 0 else 0
                
                # Ngưỡng size hợp lý (không quá lớn so với ảnh)
                max_dimension = max(image.shape[0], image.shape[1])
                size_ratio = max(w, h) / max_dimension
                
                if size_ratio > 0.7:  # Nới lỏng từ 0.5 lên 0.7
                    continue
                
                if size_ratio > 0.7:  # Nới lỏng từ 0.5 lên 0.7
                    continue
                
                # Nới lỏng điều kiện tỷ lệ cạnh
                if aspect_ratio < 0.2 or aspect_ratio > 5.0:  # Nới lỏng thêm (cũ: 0.25-4.0)
                    continue
                
                # Nới lỏng yêu cầu convexity
                if convexity < 0.4:  # Giảm từ 0.5 xuống 0.4
                    continue
                
                # Đánh giá ban đầu về loại hình dạng và độ tin cậy
                sign_type = None
                confidence = 0.0
                
                # Xác định loại hình dạng - nới lỏng các điều kiện
                if len(approx) == 3:
                    sign_type = f"{color_name}_triangle"
                    confidence = 0.7 * convexity * extent
                
                elif len(approx) == 4:
                    # Phân biệt hình vuông và chữ nhật
                    if 0.7 < aspect_ratio < 1.3:  # Nới lỏng điều kiện vuông
                        sign_type = f"{color_name}_square"
                    else:
                        sign_type = f"{color_name}_rectangle"
                    confidence = 0.8 * convexity * extent  # Tăng confidence cho biển chữ nhật
                
                elif len(approx) == 8 and color_name == "red":
                    sign_type = f"{color_name}_octagon"  # Biển STOP
                    confidence = 0.9 * convexity * extent  # Tăng confidence cho biển STOP
                
                elif (len(approx) >= 8 or (circularity > 0.65)):  # Giảm yêu cầu circularity
                    sign_type = f"{color_name}_circle"
                    confidence = 0.75 * circularity * convexity  # Tăng confidence cho biển tròn
                
                # Thêm nhận diện cho biển báo chỉ dẫn (thường là chữ nhật màu xanh với kích thước lớn)
                elif color_name == "blue" and extent > 0.65 and aspect_ratio > 1.2:
                    sign_type = f"{color_name}_direction"
                    confidence = 0.85 * extent  # Tăng confidence cho biển chỉ dẫn
                
                # Trường hợp biển mũi tên
                elif color_name == "blue" and 0.3 < convexity < 0.7:  # Giảm convexity tối thiểu
                    sign_type = f"{color_name}_arrow"
                    confidence = 0.8 * convexity  # Tăng confidence
                
                else:
                    # Không xác định rõ hình dạng nhưng vẫn có thể là biển báo
                    sign_type = f"{color_name}_sign"
                    confidence = 0.65 * convexity * extent  # Tăng confidence
                
                # Thưởng điểm cho các biển màu đỏ và xanh (thường là biển báo quan trọng)
                if color_name in ["red", "blue"]:
                    confidence *= 1.2
                
                # Tăng confidence cho các biển có kích thước lớn hơn
                normalized_area = min(1.0, area / 10000)
                confidence *= (1.0 + 0.3 * normalized_area)
                
                # Chỉ giữ lại những phát hiện có độ tin cậy cao
                if confidence > self.conf_threshold:
                    candidate_signs.append({
                        'bbox': (x, y, w, h),
                        'contour': contour,
                        'approx': approx,
                        'type': sign_type,
                        'confidence': min(1.0, confidence),  # Giới hạn tối đa là 1.0
                        'color': color_name,
                        'circularity': circularity,
                        'convexity': convexity,
                        'aspect_ratio': aspect_ratio
                    })
        
        return candidate_signs
    
    def _verify_road_signs(self, image, candidate_signs):
        """
        Xác nhận và lọc các biển báo thực sự từ các ứng viên.
        
        Args:
            image: Ảnh gốc
            candidate_signs: Danh sách các biển báo tiềm năng
            
        Returns:
            verified_signs: Danh sách các biển báo đã xác nhận
        """
        verified_signs = []
        
        for sign in candidate_signs:
            x, y, w, h = sign['bbox']
            sign_type = sign['type']
            
            # Kiểm tra thêm các đặc điểm của biển báo thực
            is_valid = True  # Flag để kiểm tra nếu biển báo hợp lệ
            
            # 1. Kiểm tra hình dạng
            # Biển báo thường có hình dạng chuẩn (tròn, tam giác, chữ nhật)
            if 'circle' in sign_type:
                if sign['circularity'] < 0.75:  # Yêu cầu độ tròn cao hơn
                    is_valid = False
            elif 'triangle' in sign_type:
                # Tam giác phải đủ "tam giác"
                approx = sign['approx']
                if len(approx) != 3:
                    is_valid = False
            elif 'square' in sign_type or 'rectangle' in sign_type:
                # Đối với hình chữ nhật, kiểm tra góc
                aspect_ratio = sign['aspect_ratio']
                if aspect_ratio < 0.4 or aspect_ratio > 2.5:  # Giới hạn tỷ lệ cạnh hợp lý
                    is_valid = False
            
            # 2. Biển báo thường có tương phản với nền (biển báo thường có tương phản cao)
            # Cắt vùng biển báo với mép rộng ra 10%
            padding_x = int(w * 0.1)
            padding_y = int(h * 0.1)
            
            y1 = max(0, y - padding_y)
            y2 = min(image.shape[0], y + h + padding_y)
            x1 = max(0, x - padding_x)
            x2 = min(image.shape[1], x + w + padding_x)
            
            roi = image[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            
            # Chuyển sang grayscale
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Tính độ tương phản (độ lệch chuẩn của độ sáng)
            std_dev = np.std(gray_roi)
            
            # Biển báo thường có tương phản cao
            if std_dev < 25:  # Tăng ngưỡng tương phản tối thiểu
                is_valid = False
            
            # 3. Kiểm tra kích thước hợp lý
            # Biển báo không quá nhỏ hoặc quá lớn
            img_area = image.shape[0] * image.shape[1]
            sign_area = w * h
            area_ratio = sign_area / img_area
            
            if area_ratio < 0.002 or area_ratio > 0.2:  # Biển báo thường chiếm 0.2%-20% diện tích ảnh
                is_valid = False
            
            # 4. Kiểm tra tính cân đối và vị trí hợp lý
            # Biển báo thường nằm ở vùng có ý nghĩa trong ảnh
            if y > image.shape[0] * 0.8:  # Biển báo hiếm khi ở dưới cùng của ảnh (80% từ trên xuống)
                is_valid = False
                
            # 5. Đối với biển báo màu xanh, kiểm tra thêm
            if 'blue' in sign_type:
                # Kiểm tra tỷ lệ màu xanh trong ROI
                hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                blue_mask = cv2.inRange(hsv_roi, np.array([90, 80, 80]), np.array([150, 255, 255]))
                blue_ratio = cv2.countNonZero(blue_mask) / (roi.shape[0] * roi.shape[1])
                
                if blue_ratio < 0.3:  # Tăng tỷ lệ yêu cầu
                    is_valid = False
            
            # 6. Đối với biển báo đỏ, kiểm tra tỷ lệ màu đỏ
            if 'red' in sign_type:
                hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                # Màu đỏ (2 khoảng trong HSV)
                lower_red1 = np.array([0, 100, 100])    # Tăng độ bão hòa và giá trị
                upper_red1 = np.array([10, 255, 255])
                lower_red2 = np.array([160, 100, 100])  # Tăng độ bão hòa và giá trị
                upper_red2 = np.array([180, 255, 255])
                
                mask_red1 = cv2.inRange(hsv_roi, lower_red1, upper_red1)
                mask_red2 = cv2.inRange(hsv_roi, lower_red2, upper_red2)
                mask_red = cv2.bitwise_or(mask_red1, mask_red2)
                
                red_ratio = cv2.countNonZero(mask_red) / (roi.shape[0] * roi.shape[1])
                
                if red_ratio < 0.2:  # Yêu cầu tối thiểu 20% là màu đỏ
                    is_valid = False
            
            # 7. Đối với biển báo màu vàng, kiểm tra tỷ lệ màu vàng
            if 'yellow' in sign_type:
                hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                yellow_mask = cv2.inRange(hsv_roi, np.array([15, 100, 100]), np.array([35, 255, 255]))
                yellow_ratio = cv2.countNonZero(yellow_mask) / (roi.shape[0] * roi.shape[1])
                
                if yellow_ratio < 0.2:  # Yêu cầu tối thiểu 20% là màu vàng
                    is_valid = False
            
            # Kiểm tra thêm các điều kiện và cải thiện độ tin cậy nếu cần
            if is_valid:
                # Thưởng nếu tương phản cao
                sign['confidence'] *= (1 + 0.1 * min(std_dev / 50, 1.0))
                
                # Nếu vẫn vượt qua ngưỡng sau các kiểm tra thêm
                if sign['confidence'] > self.conf_threshold:
                    # Cập nhật bounding box để chỉ giữ lại thông tin cần thiết
                    verified_signs.append({
                        'bbox': sign['bbox'],
                        'type': sign['type'],
                        'confidence': sign['confidence'],
                        'color': sign['color']
                    })
        
        return verified_signs
    
    def _temporal_filtering(self):
        """
        Lọc các phát hiện qua nhiều frame để giảm false positives.
        
        Returns:
            consistent_signs: Danh sách các biển báo nhất quán qua thời gian
        """
        if not self.history:
            return []
        
        # Nếu chỉ có một frame trong lịch sử
        if len(self.history) == 1:
            return self.history[0]
        
        # Tạo dictionary để theo dõi biển báo qua thời gian
        consistent_signs = {}
        
        # Duyệt qua từng frame trong lịch sử
        for frame_idx, frame_signs in enumerate(self.history):
            # Trọng số càng cao cho frame gần đây
            frame_weight = (frame_idx + 1) / len(self.history)
            
            for sign in frame_signs:
                x, y, w, h = sign['bbox']
                center_x, center_y = x + w//2, y + h//2
                
                # Tạo key dựa trên vị trí (lượng tử hóa vị trí)
                position_key = f"{center_x//30}_{center_y//30}_{w//30}_{h//30}"
                
                if position_key not in consistent_signs:
                    # Khởi tạo với trọng số thấp
                    consistent_signs[position_key] = {
                        'sign': sign,
                        'weight': frame_weight,
                        'type_counts': {sign['type']: frame_weight}
                    }
                else:
                    # Cập nhật trọng số và đếm loại
                    consistent_signs[position_key]['weight'] += frame_weight
                    
                    if sign['type'] in consistent_signs[position_key]['type_counts']:
                        consistent_signs[position_key]['type_counts'][sign['type']] += frame_weight
                    else:
                        consistent_signs[position_key]['type_counts'][sign['type']] = frame_weight
                    
                    # Cập nhật confidence
                    if sign['confidence'] > consistent_signs[position_key]['sign']['confidence']:
                        consistent_signs[position_key]['sign'] = sign
        
        # Lọc biển báo dựa trên độ nhất quán qua thời gian
        filtered_signs = []
        
        for key, info in consistent_signs.items():
            # Yêu cầu độ nhất quán cao hơn - tăng từ 1.2 lên 1.5
            if info['weight'] >= 1.5:  # Ít nhất phải xuất hiện trong nhiều frame gần đây
                sign = info['sign'].copy()
                
                # Cập nhật loại biển báo dựa trên loại phổ biến nhất
                most_common_type = max(info['type_counts'], key=info['type_counts'].get)
                sign['type'] = most_common_type
                
                # Tăng confidence dựa trên độ nhất quán
                sign['confidence'] = min(1.0, sign['confidence'] * (1 + 0.2 * info['weight']))
                
                filtered_signs.append(sign)
        
        return filtered_signs

# Hàm test để kiểm tra detector
def test_road_sign_detector():
    import cv2
    
    # Khởi tạo detector với ngưỡng thấp hơn
    detector = RoadSignDetector(conf_threshold=0.5)
    
    # Mở webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Không thể mở webcam!")
        return
    
    print("Đang test detector biển báo với webcam. Nhấn 'q' để thoát.")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Phát hiện biển báo
        detections, annotated_frame = detector.detect(frame)
        
        # Hiển thị số lượng biển báo được phát hiện
        cv2.putText(annotated_frame, f"Biển báo: {len(detections)}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Hiển thị kết quả
        cv2.imshow('Phát hiện biển báo cải tiến', annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_road_sign_detector()