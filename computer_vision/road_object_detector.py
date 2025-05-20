# Tạo file road_object_detector.py
import cv2
import numpy as np
import torch

class RoadObjectDetector:
    """Phát hiện các đối tượng trên đường như xe, người, động vật sử dụng YOLO."""
    
    def __init__(self, model_path=None, conf_threshold=0.5):
        """
        Khởi tạo detector đối tượng trên đường.
        
        Args:
            model_path: Đường dẫn đến model YOLO đã pre-trained
            conf_threshold: Ngưỡng confidence để phát hiện
        """
        self.conf_threshold = conf_threshold
        
        # Nạp model YOLO
        if model_path is None:
            # Sử dụng YOLOv5 từ torch hub
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        else:
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        
        # Các lớp từ COCO dataset
        # Người: 0
        # Xe: 2(car), 3(motorcycle), 5(bus), 7(truck)
        # Động vật: 15-23 (cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe)
        # Xe đạp: 1
        self.person_classes = [0]
        self.vehicle_classes = [1, 2, 3, 5, 7]
        self.animal_classes = list(range(15, 24))
        
        self.all_classes = self.person_classes + self.vehicle_classes + self.animal_classes
        
        print("Road Object Detector đã được khởi tạo!")
        
    def detect(self, image):
        """
        Phát hiện đối tượng trên đường trong ảnh.
        
        Args:
            image: Ảnh đầu vào (định dạng OpenCV)
            
        Returns:
            detections: List các đối tượng đã phát hiện
            annotated_img: Ảnh với kết quả phát hiện
        """
        # Thực hiện dự đoán với model YOLO
        results = self.model(image)
        
        # Lọc ra các đối tượng phát hiện được
        detections = []
        
        # Xử lý kết quả
        for pred in results.xyxy[0]:  # Output format: xmin, ymin, xmax, ymax, confidence, class
            x1, y1, x2, y2, conf, cls = pred
            
            cls_id = int(cls)
            
            # Chỉ quan tâm đến các lớp đã được chỉ định và vượt qua ngưỡng confidence
            if cls_id in self.all_classes and conf > self.conf_threshold:
                # Phân loại đối tượng
                if cls_id in self.person_classes:
                    obj_type = "person"
                    category = "person"
                elif cls_id in self.vehicle_classes:
                    if cls_id == 1:
                        obj_type = "bicycle"
                    elif cls_id == 2:
                        obj_type = "car"
                    elif cls_id == 3:
                        obj_type = "motorcycle"
                    elif cls_id == 5:
                        obj_type = "bus"
                    elif cls_id == 7:
                        obj_type = "truck"
                    else:
                        obj_type = "vehicle"
                    category = "vehicle"
                elif cls_id in self.animal_classes:
                    animal_names = ["cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"]
                    obj_type = animal_names[cls_id - 15] if 15 <= cls_id < 15 + len(animal_names) else "animal"
                    category = "animal"
                else:
                    obj_type = f"object_class_{cls_id}"
                    category = "other"
                
                detections.append({
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'confidence': float(conf),
                    'type': obj_type,
                    'category': category,
                    'class_id': cls_id
                })
        
        # Vẽ kết quả lên ảnh
        annotated_img = image.copy()
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            obj_type = det['type']
            category = det['category']
            
            # Chọn màu dựa trên loại đối tượng
            if category == "person":
                color = (0, 255, 0)  # Green for people
            elif category == "vehicle":
                color = (255, 0, 0)  # Blue for vehicles
            elif category == "animal":
                color = (0, 165, 255)  # Orange for animals
            else:
                color = (255, 255, 255)  # White for others
            
            # Vẽ bounding box
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
            
            # Tính khoảng cách ước lượng (dựa trên kích thước bounding box)
            # Giả sử, kích thước càng nhỏ thì càng xa
            width = x2 - x1
            height = y2 - y1
            size = width * height
            img_size = image.shape[0] * image.shape[1]
            distance_factor = 1.0 - min(1.0, size / (img_size * 0.5))
            distance_meters = int(50 * distance_factor)  # Ước lượng thô
            
            # Hiển thị loại đối tượng và khoảng cách
            label = f"{obj_type}: {det['confidence']:.2f}, ~{distance_meters}m"
            cv2.putText(annotated_img, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return detections, annotated_img
        
    def estimate_collision_risk(self, detections, image_width, image_height):
        """
        Ước lượng rủi ro va chạm dựa trên vị trí và kích thước của đối tượng.
        
        Args:
            detections: Các đối tượng đã phát hiện
            image_width: Chiều rộng ảnh
            image_height: Chiều cao ảnh
            
        Returns:
            risks: Danh sách các rủi ro va chạm
        """
        risks = []
        center_x = image_width // 2
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            obj_center_x = (x1 + x2) // 2
            obj_width = x2 - x1
            obj_height = y2 - y1
            obj_size = obj_width * obj_height
            
            # Ước lượng rủi ro dựa trên:
            # 1. Khoảng cách đến tâm hình (đối tượng ở giữa nguy hiểm hơn)
            # 2. Kích thước đối tượng (đối tượng càng lớn càng gần, càng nguy hiểm)
            # 3. Vị trí theo chiều y (càng thấp trong hình càng gần, càng nguy hiểm)
            
            distance_to_center = abs(obj_center_x - center_x) / center_x
            size_factor = min(1.0, obj_size / (image_width * image_height * 0.1))
            vertical_position = y2 / image_height  # 0 ở trên cùng, 1 ở dưới cùng
            
            # Công thức rủi ro (có thể điều chỉnh)
            risk_score = (1.0 - distance_to_center) * 0.4 + size_factor * 0.3 + vertical_position * 0.3
            
            # Phân loại mức độ rủi ro
            if risk_score < 0.3:
                risk_level = "low"
            elif risk_score < 0.6:
                risk_level = "medium"
            else:
                risk_level = "high"
                
            risks.append({
                'object_id': id(det),  # ID duy nhất cho đối tượng
                'object_type': det['type'],
                'risk_score': risk_score,
                'risk_level': risk_level
            })
            
        return risks

# Hàm test đơn giản để kiểm tra detector
def test_road_object_detector():
    import cv2
    
    # Khởi tạo detector
    detector = RoadObjectDetector()
    
    # Mở webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Không thể mở webcam!")
        return
    
    print("Đang test detector đối tượng trên đường với webcam. Nhấn 'q' để thoát.")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Phát hiện đối tượng
        detections, annotated_frame = detector.detect(frame)
        
        # Ước lượng rủi ro va chạm
        risks = detector.estimate_collision_risk(detections, frame.shape[1], frame.shape[0])
        
        # Hiển thị số lượng đối tượng được phát hiện
        cv2.putText(annotated_frame, f"Objects: {len(detections)}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Hiển thị mức độ rủi ro cao nhất
        if risks:
            max_risk = max(risks, key=lambda x: x['risk_score'])
            risk_text = f"Max Risk: {max_risk['risk_level'].upper()} ({max_risk['object_type']})"
            
            if max_risk['risk_level'] == "high":
                risk_color = (0, 0, 255)  # Red
            elif max_risk['risk_level'] == "medium":
                risk_color = (0, 165, 255)  # Orange
            else:
                risk_color = (0, 255, 0)  # Green
                
            cv2.putText(annotated_frame, risk_text, (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, risk_color, 2)
        
        # Hiển thị kết quả
        cv2.imshow('Road Object Detection Test', annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_road_object_detector()