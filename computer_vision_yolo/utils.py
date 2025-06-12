import cv2
import numpy as np
import os
from config import TRAFFIC_CLASSES

def get_class_name(class_id):
    """Lấy tên class từ ID"""
    return TRAFFIC_CLASSES.get(class_id, f'class_{class_id}')

def detect_traffic_light_orientation(image, x1, y1, x2, y2):
    """Phát hiện hướng của đèn giao thông (dọc hay ngang)"""
    width = x2 - x1
    height = y2 - y1
    aspect_ratio = width / height if height > 0 else 1
    
    # Nếu width > height * 1.5 thì có thể là đèn ngang
    if aspect_ratio > 1.5:
        return "horizontal"
    elif aspect_ratio < 0.7:
        return "vertical"
    else:
        return "square"  # Đèn vuông hoặc không rõ

def detect_traffic_light_color_horizontal(image, x1, y1, x2, y2):
    """Phát hiện màu đèn giao thông NGANG"""
    traffic_light_roi = image[y1:y2, x1:x2]
    
    if traffic_light_roi.size == 0:
        return "unknown", (128, 128, 128)
    
    h, w = traffic_light_roi.shape[:2]
    
    # Chia đèn ngang thành 3 phần: TRÁI-GIỮA-PHẢI
    margin = max(1, min(h, w) // 10)
    
    regions = [
        traffic_light_roi[margin:h-margin, margin:w//3-margin] if w//3-margin > margin else traffic_light_roi[:, 0:w//3],     # Trái (đỏ)
        traffic_light_roi[margin:h-margin, w//3+margin:2*w//3-margin] if 2*w//3-margin > w//3+margin else traffic_light_roi[:, w//3:2*w//3],  # Giữa (vàng)
        traffic_light_roi[margin:h-margin, 2*w//3+margin:w-margin] if w-margin > 2*w//3+margin else traffic_light_roi[:, 2*w//3:w]       # Phải (xanh)
    ]
    
    colors = ["red", "green", "green"]
    color_bgr = [(0, 0, 255), (0, 255, 0), (0, 255, 0)]
    
    max_brightness = 0
    detected_color = "off"
    detected_bgr = (128, 128, 128)
    
    # Chuyển sang HSV để phân tích màu chính xác hơn
    hsv_roi = cv2.cvtColor(traffic_light_roi, cv2.COLOR_BGR2HSV)
    
    for i, region in enumerate(regions):
        if region.size == 0:
            continue
            
        # Phân tích độ sáng
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        region_hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        
        max_val = np.max(gray)
        mean_val = np.mean(gray)
        bright_pixels = np.sum(gray > 150)  # Đếm pixel sáng
        
        # Phân tích màu sắc trong HSV
        h_channel = region_hsv[:, :, 0]
        s_channel = region_hsv[:, :, 1] 
        v_channel = region_hsv[:, :, 2]
        
        color_score = 0
        if i == 0:  # Vùng đỏ (trái)
            red_mask1 = (h_channel <= 10) & (s_channel > 100) & (v_channel > 100)
            red_mask2 = (h_channel >= 170) & (s_channel > 100) & (v_channel > 100)
            color_score = np.sum(red_mask1) + np.sum(red_mask2)
        elif i == 1:  # Vùng vàng (giữa)
            green_mask = (h_channel >= 15) & (h_channel <= 35) & (s_channel > 100) & (v_channel > 100)
            color_score = np.sum(green_mask)
        elif i == 2:  # Vùng xanh (phải)
            green_mask = (h_channel >= 40) & (h_channel <= 80) & (s_channel > 100) & (v_channel > 100)
            color_score = np.sum(green_mask)
        
        # Tổng hợp điểm số
        total_score = max_val * 0.4 + mean_val * 0.3 + bright_pixels * 0.2 + color_score * 0.1
        
        # Kiểm tra điều kiện phát hiện
        if (max_val > 100 and mean_val > 50 and bright_pixels > 2) or color_score > 15:
            if total_score > max_brightness:
                max_brightness = total_score
                detected_color = colors[i]
                detected_bgr = color_bgr[i]
    
    return detected_color, detected_bgr

def detect_traffic_light_color(image, x1, y1, x2, y2):
    """Phát hiện màu đèn giao thông - Tự động xử lý cả dọc và ngang"""
    
    # Bước 1: Xác định hướng đèn
    orientation = detect_traffic_light_orientation(image, x1, y1, x2, y2)
    
    # Bước 2: Sử dụng thuật toán phù hợp
    if orientation == "horizontal":
        return detect_traffic_light_color_horizontal(image, x1, y1, x2, y2)
    else:
        # Sử dụng thuật toán cũ cho đèn dọc
        return detect_traffic_light_color_vertical(image, x1, y1, x2, y2)

def detect_traffic_light_color_vertical(image, x1, y1, x2, y2):
    """Phát hiện màu đèn giao thông DỌC (thuật toán gốc được cải thiện)"""
    traffic_light_roi = image[y1:y2, x1:x2]
    
    if traffic_light_roi.size == 0:
        return "unknown", (128, 128, 128)
    
    # Phân tích BGR trực tiếp
    b, g, r = cv2.split(traffic_light_roi)
    
    avg_b, avg_g, avg_r = np.mean(b), np.mean(g), np.mean(r)
    max_b, max_g, max_r = np.max(b), np.max(g), np.max(r)
    
    # Điều kiện cho màu XANH LÁ (GREEN)
    green_condition1 = avg_g > avg_r * 1.15 and avg_g > avg_b * 1.3
    green_condition2 = max_g > max_r * 1.1 and max_g > max_b * 1.2
    green_ratio = avg_g / (avg_r + avg_b + 1)
    
    # Điều kiện cho màu VÀNG (green)  
    green_condition1 = abs(avg_r - avg_g) < 25 and avg_b < min(avg_r, avg_g) * 0.7
    green_condition2 = max_r > max_b * 1.5 and max_g > max_b * 1.5
    
    # Điều kiện cho màu ĐỎ (RED)
    red_condition1 = avg_r > avg_g * 1.2 and avg_r > avg_b * 1.3
    red_condition2 = max_r > max_g * 1.1 and max_r > max_b * 1.2
    
    # Kiểm tra độ sáng tối thiểu
    min_brightness = 30
    if max(avg_r, avg_g, avg_b) < min_brightness:
        return "off", (128, 128, 128)
    
    # Quyết định cuối cùng
    if (green_condition1 and green_condition2) or green_ratio > 0.65:
        return "green", (0, 255, 0)
    elif red_condition1 and red_condition2:
        return "red", (0, 0, 255)  
    elif green_condition1 and green_condition2:
        return "green", (0, 255, 0)
    else:
        # Fallback
        if avg_g > avg_r and avg_g > avg_b:
            return "green", (0, 255, 0)
        elif avg_r > avg_g and avg_r > avg_b:
            return "red", (0, 0, 255)
        elif abs(avg_r - avg_g) < 20 and avg_b < min(avg_r, avg_g):
            return "green", (0, 255, 0)
        else:
            return "unknown", (128, 128, 128)

def detect_traffic_light_color_advanced(image, x1, y1, x2, y2):
    """Phát hiện màu đèn giao thông nâng cao - Tự động phát hiện hướng"""
    
    # Xác định hướng đèn trước
    orientation = detect_traffic_light_orientation(image, x1, y1, x2, y2)
    
    if orientation == "horizontal":
        return detect_traffic_light_color_horizontal(image, x1, y1, x2, y2)
    else:
        # Thuật toán cho đèn dọc (giữ nguyên logic cũ)
        traffic_light_roi = image[y1:y2, x1:x2]
        
        if traffic_light_roi.size == 0:
            return "unknown", (128, 128, 128)
        
        h, w = traffic_light_roi.shape[:2]
        margin = max(1, min(h, w) // 10)
        
        # Chia đèn dọc thành 3 phần: TRÊN-GIỮA-DƯỚI
        regions = [
            traffic_light_roi[margin:h//3-margin, margin:w-margin] if h//3-margin > margin else traffic_light_roi[0:h//3, :],
            traffic_light_roi[h//3+margin:2*h//3-margin, margin:w-margin] if 2*h//3-margin > h//3+margin else traffic_light_roi[h//3:2*h//3, :],
            traffic_light_roi[2*h//3+margin:h-margin, margin:w-margin] if h-margin > 2*h//3+margin else traffic_light_roi[2*h//3:h, :]
        ]
        
        colors = ["red", "green", "green"]
        color_bgr = [(0, 0, 255), (0, 255, 0), (0, 255, 0)]
        
        max_brightness = 0
        detected_color = "off"
        detected_bgr = (128, 128, 128)
        
        # Phân tích HSV
        hsv_roi = cv2.cvtColor(traffic_light_roi, cv2.COLOR_BGR2HSV)
        
        for i, region in enumerate(regions):
            if region.size == 0:
                continue
                
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            region_hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
            
            max_val = np.max(gray)
            mean_val = np.mean(gray)
            bright_pixels = np.sum(gray > 180)
            
            h_channel = region_hsv[:, :, 0]
            s_channel = region_hsv[:, :, 1]
            v_channel = region_hsv[:, :, 2]
            
            color_score = 0
            if i == 0:  # Vùng đỏ (trên)
                red_mask1 = (h_channel <= 10) & (s_channel > 100) & (v_channel > 100)
                red_mask2 = (h_channel >= 170) & (s_channel > 100) & (v_channel > 100)
                color_score = np.sum(red_mask1) + np.sum(red_mask2)
            elif i == 1:  # Vùng vàng (giữa)
                green_mask = (h_channel >= 15) & (h_channel <= 35) & (s_channel > 100) & (v_channel > 100)
                color_score = np.sum(green_mask)
            elif i == 2:  # Vùng xanh (dưới)
                green_mask = (h_channel >= 40) & (h_channel <= 80) & (s_channel > 100) & (v_channel > 100)
                color_score = np.sum(green_mask)
            
            total_score = max_val * 0.4 + mean_val * 0.3 + bright_pixels * 0.2 + color_score * 0.1
            
            if (max_val > 120 and mean_val > 60 and bright_pixels > 3) or color_score > 20:
                if total_score > max_brightness:
                    max_brightness = total_score
                    detected_color = colors[i]
                    detected_bgr = color_bgr[i]
        
        return detected_color, detected_bgr

def detect_arrow_direction(image, x1, y1, x2, y2):
    """Phát hiện hướng mũi tên trong đèn giao thông"""
    try:
        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            return "none"
        
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 50:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
                if aspect_ratio > 1.2:
                    moments = cv2.moments(contour)
                    if moments["m00"] != 0:
                        cx = int(moments["m10"] / moments["m00"])
                        if cx > w * 0.6:
                            return "right"
                        elif cx < w * 0.4:
                            return "left"
        
        return "none"
    except:
        return "none"

def draw_boxes(image, results):
    """Vẽ bounding box lên ảnh với phát hiện màu đèn giao thông cải thiện"""
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                
                class_name = TRAFFIC_CLASSES.get(class_id, f'object_{class_id}')
                box_color = (0, 255, 0)
                
                # Nếu là đèn giao thông
                if class_id == 9:  # traffic light
                    try:
                        # Phát hiện hướng đèn
                        orientation = detect_traffic_light_orientation(image, x1, y1, x2, y2)
                        
                        # Sử dụng thuật toán phù hợp
                        light_color1, color_bgr1 = detect_traffic_light_color(image, x1, y1, x2, y2)
                        light_color2, color_bgr2 = detect_traffic_light_color_advanced(image, x1, y1, x2, y2)
                        
                        # Ưu tiên kết quả của phương pháp nâng cao
                        if light_color2 != "off" and light_color2 != "unknown":
                            light_color, color_bgr = light_color2, color_bgr2
                        else:
                            light_color, color_bgr = light_color1, color_bgr1
                        
                        # Phát hiện hướng mũi tên
                        arrow_direction = detect_arrow_direction(image, x1, y1, x2, y2)
                        
                        # Tạo label với thông tin hướng đèn
                        if arrow_direction != "none":
                            class_name = f"Traffic Light ({orientation.upper()}) ({light_color.upper()}) - {arrow_direction.upper()} ARROW"
                        else:
                            class_name = f"Traffic Light ({orientation.upper()}) ({light_color.upper()})"
                            
                        box_color = color_bgr
                        
                    except Exception as e:
                        light_color = "unknown"
                        class_name = "Traffic Light (UNKNOWN)"
                
                # Vẽ box
                if confidence > 0.1:
                    thickness = 3 if class_id == 9 else 2
                    cv2.rectangle(image, (x1, y1), (x2, y2), box_color, thickness)
                    
                    label = f'{class_name}: {confidence:.2f}'
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    
                    # Vẽ nền cho text
                    cv2.rectangle(image, (x1, y1-label_size[1]-10), 
                                (x1+label_size[0]+5, y1), box_color, -1)
                    
                    # Vẽ text
                    text_color = (0, 0, 0) if class_id == 9 and box_color == (0, 255, 0) else (255, 255, 255)
                    cv2.putText(image, label, (x1+2, y1-5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
    
    return image

def get_files_in_folder(folder_path, extensions):
    """Lấy danh sách file theo extension"""
    files = []
    if os.path.exists(folder_path):
        for file in os.listdir(folder_path):
            if any(file.lower().endswith(ext) for ext in extensions):
                files.append(os.path.join(folder_path, file))
    return files

def analyze_traffic_light_stats(image, results):
    """Phân tích thống kê đèn giao thông trong ảnh"""
    traffic_lights = []
    
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                class_id = int(box.cls[0])
                if class_id == 9:  # traffic light
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    
                    # Phát hiện màu và hướng
                    orientation = detect_traffic_light_orientation(image, x1, y1, x2, y2)
                    light_color, _ = detect_traffic_light_color_advanced(image, x1, y1, x2, y2)
                    
                    traffic_lights.append({
                        'position': (x1, y1, x2, y2),
                        'color': light_color,
                        'orientation': orientation,
                        'confidence': confidence
                    })
    
    return traffic_lights