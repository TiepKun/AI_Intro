# utils_torchvision.py - Utilities for Torchvision Faster R-CNN
import cv2
import numpy as np
import torch
import os
from config_torchvision import *

class TorchvisionUtils:
    """Utilities for Torchvision Faster R-CNN detection and visualization"""
    
    def __init__(self):
        self.traffic_light_colors = TRAFFIC_LIGHT_COLORS
    
    def draw_detections(self, image, results):
        """Draw detection results on image"""
        if results is None:
            return image
        
        try:
            boxes = results['boxes'].cpu().numpy()
            labels = results['labels'].cpu().numpy()
            scores = results['scores'].cpu().numpy()
            
            for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
                if score < CONFIDENCE_THRESHOLD:
                    continue
                
                x1, y1, x2, y2 = map(int, box)
                
                # Get class name
                class_name = COCO_CLASSES[label] if label < len(COCO_CLASSES) else f'class_{label}'
                
                # Special handling for traffic lights
                if label == 10:  # traffic light
                    image = self._draw_traffic_light(image, x1, y1, x2, y2, score)
                else:
                    # Regular object detection
                    color = CLASS_COLORS.get(label, (255, 255, 255))
                    image = self._draw_detection_box(image, x1, y1, x2, y2, class_name, score, color)
            
            return image
            
        except Exception as e:
            print(f"Error drawing detections: {e}")
            return image
    
    def _draw_detection_box(self, image, x1, y1, x2, y2, class_name, score, color):
        """Draw detection box with label"""
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, OUTPUT_CONFIG['line_thickness'])
        
        # Prepare label
        if OUTPUT_CONFIG['draw_confidence_scores']:
            label = f'{class_name}: {score:.2f}'
        else:
            label = class_name
        
        # Calculate label size
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = OUTPUT_CONFIG['font_scale']
        thickness = 2
        label_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        
        # Draw label background
        label_y = y1 - 10 if y1 - 10 > label_size[1] else y1 + label_size[1] + 10
        cv2.rectangle(image, (x1, label_y - label_size[1] - 5), 
                     (x1 + label_size[0] + 5, label_y + 5), color, -1)
        
        # Draw label text
        text_color = (255, 255, 255) if sum(color) < 400 else (0, 0, 0)
        cv2.putText(image, label, (x1 + 2, label_y - 2), font, font_scale, text_color, thickness)
        
        return image
    
    def _draw_traffic_light(self, image, x1, y1, x2, y2, score):
        """Draw traffic light with color detection"""
        try:
            # Detect traffic light color
            light_color, color_rgb = self._detect_traffic_light_color(image, x1, y1, x2, y2)
            
            # Draw bounding box with detected color
            cv2.rectangle(image, (x1, y1), (x2, y2), color_rgb, 3)
            
            # Create label
            label = f'Traffic Light ({light_color.upper()}): {score:.2f}'
            
            # Draw label
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            label_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
            
            # Label background
            label_y = y1 - 10 if y1 - 10 > label_size[1] else y1 + label_size[1] + 10
            cv2.rectangle(image, (x1, label_y - label_size[1] - 5), 
                         (x1 + label_size[0] + 5, label_y + 5), color_rgb, -1)
            
            # Label text
            text_color = (255, 255, 255) if sum(color_rgb) < 400 else (0, 0, 0)
            cv2.putText(image, label, (x1 + 2, label_y - 2), font, font_scale, text_color, thickness)
            
            return image
            
        except Exception as e:
            print(f"Error drawing traffic light: {e}")
            # Fallback to regular detection box
            return self._draw_detection_box(image, x1, y1, x2, y2, 'Traffic Light', score, (128, 128, 128))
    
    def _detect_traffic_light_color(self, image, x1, y1, x2, y2):
        """Detect traffic light color using computer vision"""
        try:
            # Extract ROI
            roi = image[y1:y2, x1:x2]
            if roi.size == 0:
                return 'unknown', self.traffic_light_colors['unknown']
            
            # Determine orientation
            width = x2 - x1
            height = y2 - y1
            is_horizontal = width > height * 1.3
            
            if is_horizontal:
                return self._detect_horizontal_light_color(roi)
            else:
                return self._detect_vertical_light_color(roi)
                
        except Exception as e:
            print(f"Traffic light color detection error: {e}")
            return 'unknown', self.traffic_light_colors['unknown']
    
    def _detect_vertical_light_color(self, roi):
        """Detect color in vertical traffic light"""
        h, w = roi.shape[:2]
        
        # Divide into 3 regions: top (red), middle (yellow), bottom (green)
        margin = max(1, min(h, w) // 10)
        
        regions = [
            roi[margin:h//3-margin, margin:w-margin],           # Top - Red
            roi[h//3+margin:2*h//3-margin, margin:w-margin],    # Middle - Yellow  
            roi[2*h//3+margin:h-margin, margin:w-margin]        # Bottom - Green
        ]
        
        colors = ['red', 'yellow', 'green']
        color_values = [self.traffic_light_colors[c] for c in colors]
        
        return self._analyze_light_regions(regions, colors, color_values)
    
    def _detect_horizontal_light_color(self, roi):
        """Detect color in horizontal traffic light"""
        h, w = roi.shape[:2]
        
        # Divide into 3 regions: left (red), middle (yellow), right (green)
        margin = max(1, min(h, w) // 10)
        
        regions = [
            roi[margin:h-margin, margin:w//3-margin],           # Left - Red
            roi[margin:h-margin, w//3+margin:2*w//3-margin],    # Middle - Yellow
            roi[margin:h-margin, 2*w//3+margin:w-margin]       # Right - Green
        ]
        
        colors = ['red', 'yellow', 'green']
        color_values = [self.traffic_light_colors[c] for c in colors]
        
        return self._analyze_light_regions(regions, colors, color_values)
    
    def _analyze_light_regions(self, regions, colors, color_values):
        """Analyze light regions to determine active color"""
        max_brightness = 0
        detected_color = 'off'
        detected_rgb = self.traffic_light_colors['off']
        
        for i, region in enumerate(regions):
            if region.size == 0:
                continue
            
            # Convert to HSV and grayscale for analysis
            try:
                hsv_region = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
                gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            except:
                continue
            
            # Analyze brightness
            max_val = np.max(gray_region)
            mean_val = np.mean(gray_region)
            bright_pixels = np.sum(gray_region > 150)
            
            # Analyze color saturation and hue
            h_channel = hsv_region[:, :, 0]
            s_channel = hsv_region[:, :, 1]
            v_channel = hsv_region[:, :, 2]
            
            # Color-specific analysis
            color_score = 0
            if i == 0:  # Red region
                red_mask1 = (h_channel <= 10) & (s_channel > 100) & (v_channel > 100)
                red_mask2 = (h_channel >= 170) & (s_channel > 100) & (v_channel > 100)
                color_score = np.sum(red_mask1) + np.sum(red_mask2)
            elif i == 1:  # Yellow region
                yellow_mask = (h_channel >= 15) & (h_channel <= 35) & (s_channel > 100) & (v_channel > 100)
                color_score = np.sum(yellow_mask)
            elif i == 2:  # Green region
                green_mask = (h_channel >= 40) & (h_channel <= 80) & (s_channel > 100) & (v_channel > 100)
                color_score = np.sum(green_mask)
            
            # Calculate total score
            total_score = max_val * 0.4 + mean_val * 0.3 + bright_pixels * 0.2 + color_score * 0.1
            
            # Check if this region is active
            if (max_val > 120 and mean_val > 60 and bright_pixels > 3) or color_score > 20:
                if total_score > max_brightness:
                    max_brightness = total_score
                    detected_color = colors[i]
                    detected_rgb = color_values[i]
        
        return detected_color, detected_rgb
    
    def draw_vn_signs(self, image, vn_results, vn_detector):
        """Draw Vietnamese traffic signs"""
        if vn_results is None or not hasattr(vn_detector, 'vn_classes'):
            return image
        
        try:
            boxes = vn_results['boxes'].cpu().numpy()
            labels = vn_results['labels'].cpu().numpy()
            scores = vn_results['scores'].cpu().numpy()
            
            for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
                if score < 0.4:  # VN signs threshold
                    continue
                
                x1, y1, x2, y2 = map(int, box)
                
                # Get sign info
                # Convert to 0-indexed if needed
                class_id = label - 1 if label > 0 else label
                sign_name = vn_detector.vn_classes.get(class_id, f'vn_sign_{class_id}')
                category = vn_detector.get_sign_category(class_id)
                color = vn_detector.category_colors.get(category, (128, 128, 128))
                
                # Draw bounding box
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
                
                # Create label
                label_text = f'[{category.upper()}] {sign_name}: {score:.2f}'
                
                # Draw label
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 2
                label_size = cv2.getTextSize(label_text, font, font_scale, thickness)[0]
                
                # Label background
                cv2.rectangle(image, (x1, y1-label_size[1]-10), 
                             (x1+label_size[0]+5, y1), color, -1)
                
                # Label text
                text_color = (255, 255, 255) if sum(color) < 400 else (0, 0, 0)
                cv2.putText(image, label_text, (x1+2, y1-5), font, font_scale, text_color, thickness)
            
            return image
            
        except Exception as e:
            print(f"Error drawing VN signs: {e}")
            return image

class ModelEvaluator:
    """Evaluate model performance"""
    
    def __init__(self, model):
        self.model = model
        self.device = DEVICE
    
    def evaluate_on_dataset(self, dataloader):
        """Evaluate model on dataset"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for images, targets in dataloader:
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                # Forward pass
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                
                total_loss += losses.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        return avg_loss
    
    def benchmark_speed(self, image_size=(640, 480), num_iterations=100):
        """Benchmark model inference speed"""
        self.model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, image_size[1], image_size[0]).to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                self.model(dummy_input)
        
        # Benchmark
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_iterations):
                self.model(dummy_input)
        
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_iterations
        fps = 1.0 / avg_time
        
        return {
            'avg_inference_time': avg_time,
            'fps': fps,
            'image_size': image_size,
            'iterations': num_iterations
        }

def create_demo_images():
    """Create demo images for testing"""
    demo_dir = os.path.join(INPUT_DIR, 'demo')
    os.makedirs(demo_dir, exist_ok=True)
    
    # Create simple demo images
    for i in range(3):
        # Create colored image
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Add some simple shapes
        cv2.rectangle(img, (100, 100), (200, 200), (0, 255, 0), -1)  # Green square
        cv2.circle(img, (400, 200), 50, (0, 0, 255), -1)  # Red circle
        
        # Save
        filename = f'demo_image_{i+1}.jpg'
        filepath = os.path.join(demo_dir, filename)
        cv2.imwrite(filepath, img)
    
    print(f"Demo images created in: {demo_dir}")
    return demo_dir

def get_system_info():
    """Get system information"""
    import platform
    
    info = {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'pytorch_version': torch.__version__,
        'torchvision_version': torchvision.__version__,
        'cuda_available': torch.cuda.is_available(),
        'device': str(DEVICE)
    }
    
    if torch.cuda.is_available():
        info['cuda_version'] = torch.version.cuda
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['gpu_memory'] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
    
    return info

def print_system_info():
    """Print system information"""
    info = get_system_info()
    
    print("\nSystem Information:")
    print("=" * 30)
    for key, value in info.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    print("=" * 30)

# Utility functions for file handling
def get_image_files(directory, extensions=('.jpg', '.jpeg', '.png', '.bmp')):
    """Get all image files in directory"""
    image_files = []
    if os.path.exists(directory):
        for file in os.listdir(directory):
            if file.lower().endswith(extensions):
                image_files.append(os.path.join(directory, file))
    return sorted(image_files)

def get_video_files(directory, extensions=('.mp4', '.avi', '.mov', '.mkv')):
    """Get all video files in directory"""
    video_files = []
    if os.path.exists(directory):
        for file in os.listdir(directory):
            if file.lower().endswith(extensions):
                video_files.append(os.path.join(directory, file))
    return sorted(video_files)

def resize_image_if_needed(image, max_size=None):
    """Resize image if needed"""
    if max_size is None:
        max_size = PERFORMANCE_CONFIG['max_image_size']
    
    h, w = image.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        image = cv2.resize(image, (new_w, new_h))
        print(f"Image resized from {w}x{h} to {new_w}x{new_h}")
    
    return image