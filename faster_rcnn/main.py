import cv2
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from pathlib import Path
import time

# Torchvision Faster R-CNN imports
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, fasterrcnn_mobilenet_v3_large_fpn
from torchvision.transforms import functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

print(f"Using Torchvision {torchvision.__version__}")
print(f"PyTorch {torch.__version__}")

from config_torchvision import *
from utils_torchvision import TorchvisionUtils

class FasterRCNNDetector:
    """Traffic Detection using Torchvision Faster R-CNN"""
    
    def __init__(self, model_type='resnet50', custom_weights=None):
        self.model = None
        self.is_loaded = False
        self.model_type = model_type
        
        print(f"Initializing Faster R-CNN ({model_type})...")
        
        try:
            # Load model based on type
            if model_type == 'resnet50':
                if custom_weights and os.path.exists(custom_weights):
                    self.model = fasterrcnn_resnet50_fpn(pretrained=False)
                    self.model.load_state_dict(torch.load(custom_weights, map_location=DEVICE))
                    print(f"Loaded custom weights: {custom_weights}")
                else:
                    self.model = fasterrcnn_resnet50_fpn(pretrained=True)
                    print("Loaded pre-trained ResNet50 Faster R-CNN")
                    
            elif model_type == 'mobilenet':
                if custom_weights and os.path.exists(custom_weights):
                    self.model = fasterrcnn_mobilenet_v3_large_fpn(pretrained=False)
                    self.model.load_state_dict(torch.load(custom_weights, map_location=DEVICE))
                    print(f"Loaded custom MobileNet weights: {custom_weights}")
                else:
                    self.model = fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
                    print("Loaded pre-trained MobileNet Faster R-CNN")
            
            # Set device and evaluation mode
            self.model.to(DEVICE)
            self.model.eval()
            
            self.is_loaded = True
            print(f"Faster R-CNN loaded successfully on {DEVICE}!")
            
        except Exception as e:
            print(f"Error loading Faster R-CNN: {e}")
            self.is_loaded = False
    
    def detect(self, image, confidence_threshold=0.5):
        """Detect objects in image"""
        if not self.is_loaded:
            return None
        
        try:
            # Preprocess image
            if isinstance(image, np.ndarray):
                # Convert BGR to RGB if needed
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image_rgb = image
                
                # Convert to tensor
                image_tensor = F.to_tensor(image_rgb).unsqueeze(0).to(DEVICE)
            else:
                image_tensor = image.unsqueeze(0).to(DEVICE)
            
            # Inference
            with torch.no_grad():
                predictions = self.model(image_tensor)
            
            # Filter by confidence
            result = predictions[0]
            keep_indices = result['scores'] > confidence_threshold
            
            filtered_result = {
                'boxes': result['boxes'][keep_indices],
                'labels': result['labels'][keep_indices],
                'scores': result['scores'][keep_indices]
            }
            
            return filtered_result
            
        except Exception as e:
            print(f"Detection error: {e}")
            return None

class VNSignsDetector:
    """Vietnamese Traffic Signs Detector"""
    
    def __init__(self, model_path=None):
        self.model = None
        self.is_loaded = False
        
        # VN Signs classes (52 classes)
        self.vn_classes = {
            0: 'w224', 1: 'w205c', 2: 'p102', 3: 'r302a', 4: 'w205a',
            5: 'w207', 6: 'w201a', 7: 'p123a', 8: 'i434a', 9: 'r303',
            10: 'p130', 11: 'i409', 12: 'r415a', 13: 'w245a', 14: 'p106axe_tai',
            15: 'w203c', 16: 'p117', 17: 'p124a', 18: 'p107', 19: 'p124d',
            20: 'p103a', 21: 'w203b', 22: 'w221b', 23: 'p111', 24: 'p129',
            25: 's505axe_may', 26: 'w246a', 27: 'w225', 28: 's505axe_tai_va_cong',
            29: 'p104', 30: 's505axe_tai', 31: 'camera', 32: 'p123b', 33: 'w202b',
            34: 'b8a', 35: 'p137', 36: 'p139', 37: 'w205b', 38: 'p12750',
            39: 'p12760', 40: 'p12780', 41: 'p12740', 42: 'r301e', 43: 'w239b',
            44: 'w233', 45: 'i407a', 46: 'p131a', 47: 'p124b1', 48: 'w210',
            49: 'p124c', 50: 'w201b', 51: 'w246c'
        }
        
        # Sign categories and colors
        self.sign_categories = {
            'prohibition': [2, 7, 10, 14, 17, 18, 19, 20, 23, 29, 32, 34, 35, 36, 46, 47, 49],
            'warning': [0, 1, 4, 5, 6, 15, 16, 21, 22, 26, 27, 33, 43, 44, 48, 50, 51],
            'mandatory': [3, 9, 11, 12, 25, 30, 42, 45],
            'information': [8, 13, 24, 28, 31, 37],
            'speed_limit': [38, 39, 40, 41]
        }
        
        self.category_colors = {
            'prohibition': (0, 0, 255),     # Red
            'warning': (0, 165, 255),       # Orange
            'mandatory': (255, 0, 0),       # Blue
            'information': (0, 255, 0),     # Green
            'speed_limit': (0, 255, 255),   # Yellow
            'default': (128, 128, 128)      # Gray
        }
        
        if model_path and os.path.exists(model_path):
            try:
                # Load custom VN signs model
                self.model = fasterrcnn_resnet50_fpn(pretrained=False)
                
                # Modify classifier for VN signs
                in_features = self.model.roi_heads.box_predictor.cls_score.in_features
                self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(self.vn_classes) + 1)
                
                # Load weights
                self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
                self.model.to(DEVICE)
                self.model.eval()
                
                self.is_loaded = True
                print("VN Signs detector loaded!")
                
            except Exception as e:
                print(f"VN Signs loading error: {e}")
        else:
            print("VN Signs model not found (vehicles only)")
    
    def get_sign_category(self, class_id):
        """Get sign category"""
        for category, class_ids in self.sign_categories.items():
            if class_id in class_ids:
                return category
        return 'default'
    
    def detect(self, image, confidence_threshold=0.4):
        """Detect VN signs"""
        if not self.is_loaded:
            return None
        
        try:
            # Preprocess
            if isinstance(image, np.ndarray):
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_tensor = F.to_tensor(image_rgb).unsqueeze(0).to(DEVICE)
            else:
                image_tensor = image.unsqueeze(0).to(DEVICE)
            
            # Inference
            with torch.no_grad():
                predictions = self.model(image_tensor)
            
            # Filter results
            result = predictions[0]
            keep_indices = result['scores'] > confidence_threshold
            
            filtered_result = {
                'boxes': result['boxes'][keep_indices],
                'labels': result['labels'][keep_indices],
                'scores': result['scores'][keep_indices]
            }
            
            return filtered_result
            
        except Exception as e:
            print(f"VN signs detection error: {e}")
            return None

class TrafficDetectionSystem:
    """Complete Traffic Detection System"""
    
    def __init__(self, model_type='resnet50'):
        print("Initializing Traffic Detection System...")
        print("=" * 50)
        
        # Initialize main detector
        self.main_detector = FasterRCNNDetector(model_type)
        
        # Initialize VN signs detector
        vn_model_path = os.path.join(MODEL_DIR, 'vn_signs_faster_rcnn.pth')
        self.vn_detector = VNSignsDetector(vn_model_path)
        
        # Initialize utilities
        self.utils = TorchvisionUtils()
        
        if self.main_detector.is_loaded:
            print("Traffic Detection System Ready!")
            print(f"Main Model: Faster R-CNN ({model_type})")
            print(f"VN Signs: {'Loaded' if self.vn_detector.is_loaded else 'Not available'}")
        else:
            raise RuntimeError("Failed to initialize detection system!")
    
    def detect_image(self, image_path, save_result=True):
        """Detect objects in image"""
        print(f"Processing image: {os.path.basename(image_path)}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Cannot read image: {image_path}")
            return None
        
        print(f"Image size: {image.shape[1]}x{image.shape[0]}")
        
        # Resize if too large
        image = self._resize_if_needed(image)
        
        # Main detection
        print("Detecting vehicles and traffic lights...")
        start_time = time.time()
        
        main_results = self.main_detector.detect(image, CONFIDENCE_THRESHOLD)
        
        detection_time = time.time() - start_time
        print(f"Detection time: {detection_time:.2f}s")
        
        # VN signs detection
        vn_results = None
        if self.vn_detector.is_loaded:
            print("Detecting VN traffic signs...")
            vn_results = self.vn_detector.detect(image, 0.4)
        
        # Draw results
        result_image = self._draw_all_results(image.copy(), main_results, vn_results)
        
        # Print statistics
        self._print_statistics(main_results, vn_results)
        
        # Save result
        if save_result:
            output_path = os.path.join(OUTPUT_DIR, 'images', f'frcnn_detected_{os.path.basename(image_path)}')
            cv2.imwrite(output_path, result_image)
            print(f"Result saved: {output_path}")
        
        return result_image
    
    def _resize_if_needed(self, image, max_size=1024):
        """Resize image if too large"""
        h, w = image.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            image = cv2.resize(image, (new_w, new_h))
            print(f"Image resized from {w}x{h} to {new_w}x{new_h}")
        return image
    
    def _draw_all_results(self, image, main_results, vn_results):
        """Draw all detection results"""
        
        # Draw main detection results
        if main_results is not None:
            image = self.utils.draw_detections(image, main_results)
        
        # Draw VN signs
        if vn_results is not None and self.vn_detector.is_loaded:
            image = self.utils.draw_vn_signs(image, vn_results, self.vn_detector)
        
        return image
    
    def _print_statistics(self, main_results, vn_results):
        """Print detection statistics"""
        
        # Count main objects
        vehicle_count = 0
        traffic_light_count = 0
        person_count = 0
        
        if main_results is not None:
            labels = main_results['labels'].cpu().numpy()
            scores = main_results['scores'].cpu().numpy()
            
            for label, score in zip(labels, scores):
                if score > CONFIDENCE_THRESHOLD:
                    if label in [2, 3, 4, 6, 8]:  # vehicles (1-indexed: car, motorcycle, airplane, bus, truck)
                        vehicle_count += 1
                    elif label == 10:  # traffic light
                        traffic_light_count += 1
                    elif label == 1:  # person
                        person_count += 1
        
        # Count VN signs
        vn_sign_count = 0
        if vn_results is not None:
            scores = vn_results['scores'].cpu().numpy()
            vn_sign_count = len([s for s in scores if s > 0.4])
        
        print(f"\nDetection Results:")
        print(f"Vehicles: {vehicle_count}")
        print(f"Traffic Lights: {traffic_light_count}")
        print(f"Persons: {person_count}")
        print(f"VN Signs: {vn_sign_count}")
        print(f"Model: Faster R-CNN ({self.main_detector.model_type})")
    
    def detect_video(self, video_path, save_result=True):
        """Process video"""
        print(f"Processing video: {os.path.basename(video_path)}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Cannot open video: {video_path}")
            return
        
        # Video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video: {width}x{height}, {fps}fps, {total_frames} frames")
        
        # Video writer
        if save_result:
            output_path = os.path.join(OUTPUT_DIR, 'videos', f'frcnn_detected_{os.path.basename(video_path)}')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        process_every = 3  # Process every 3rd frame
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Progress update
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {frame_count}/{total_frames} frames ({progress:.1f}%)")
            
            # Process frame
            if frame_count % process_every == 0:
                # Resize frame for faster processing
                frame_resized = self._resize_if_needed(frame, 640)
                
                # Detection
                main_results = self.main_detector.detect(frame_resized, 0.6)
                
                vn_results = None
                if self.vn_detector.is_loaded and frame_count % 6 == 0:  # VN signs less frequently
                    vn_results = self.vn_detector.detect(frame_resized, 0.5)
                
                # Scale results back to original frame size if resized
                if frame_resized.shape != frame.shape:
                    scale_x = frame.shape[1] / frame_resized.shape[1]
                    scale_y = frame.shape[0] / frame_resized.shape[0]
                    
                    if main_results is not None:
                        main_results['boxes'][:, [0, 2]] *= scale_x
                        main_results['boxes'][:, [1, 3]] *= scale_y
                    
                    if vn_results is not None:
                        vn_results['boxes'][:, [0, 2]] *= scale_x
                        vn_results['boxes'][:, [1, 3]] *= scale_y
                
                # Draw results
                frame = self._draw_all_results(frame, main_results, vn_results)
            
            # Save frame
            if save_result:
                out.write(frame)
            
            # Display
            display_frame = cv2.resize(frame, (800, 600))
            cv2.imshow('Faster R-CNN Video Processing', display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Video processing stopped by user")
                break
        
        # Cleanup
        cap.release()
        if save_result:
            out.release()
            print(f"Video saved: {output_path}")
        
        cv2.destroyAllWindows()
        print("Video processing complete!")
    
    def detect_webcam(self):
        """Real-time webcam detection"""
        print("Starting webcam detection...")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open webcam!")
            return
        
        # Set webcam resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("Webcam ready! Controls: 'q'=quit, 's'=save image")
        
        frame_count = 0
        fps_start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process every 5th frame for real-time performance
            if frame_count % 5 == 0:
                try:
                    # Main detection
                    main_results = self.main_detector.detect(frame, 0.6)
                    
                    # VN signs (less frequent)
                    vn_results = None
                    if self.vn_detector.is_loaded and frame_count % 15 == 0:
                        vn_results = self.vn_detector.detect(frame, 0.5)
                    
                    # Draw results
                    frame = self._draw_all_results(frame, main_results, vn_results)
                    
                except Exception as e:
                    print(f"Detection error: {e}")
            
            # Calculate and display FPS
            if frame_count % 30 == 0:
                fps_end_time = time.time()
                fps = 30 / (fps_end_time - fps_start_time)
                fps_start_time = fps_end_time
                print(f"FPS: {fps:.1f}")
            
            # Add overlay info
            cv2.putText(frame, 'Faster R-CNN Detection', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f'Frame: {frame_count}', (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow('Faster R-CNN Webcam Detection', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                save_path = os.path.join(OUTPUT_DIR, 'images', f'webcam_frcnn_{frame_count}.jpg')
                cv2.imwrite(save_path, frame)
                print(f"Image saved: {save_path}")
        
        cap.release()
        cv2.destroyAllWindows()
        print("Webcam detection ended")

def main():
    """Main function"""
    
    print("Traffic Detection with Torchvision Faster R-CNN")
    print("=" * 50)
    print("Pure torchvision implementation - no detectron2 needed")
    print("=" * 50)
    
    try:
        # Choose model type
        print("\nSelect model type:")
        print("1. ResNet50 (better accuracy)")
        print("2. MobileNet (faster)")
        
        choice = input("Choice (1-2, default=1): ").strip()
        model_type = 'mobilenet' if choice == '2' else 'resnet50'
        
        detector = TrafficDetectionSystem(model_type)
        
    except Exception as e:
        print(f"System initialization failed: {e}")
        print("\nTo install required packages:")
        print("pip install torch torchvision opencv-python numpy pillow")
        return
    
    while True:
        print(f"\nFaster R-CNN Detection Menu:")
        print("1. Detect Image")
        print("2. Process Video") 
        print("3. Webcam Real-time")
        print("4. Batch Images")
        print("5. Model Information")
        print("0. Exit")
        print("-" * 30)
        
        choice = input("Select option (0-5): ").strip()
        
        if choice == '1':
            image_path = input("Image path: ").strip()
            if os.path.exists(image_path):
                result = detector.detect_image(image_path)
                if result is not None:
                    # Display result
                    h, w = result.shape[:2]
                    if max(h, w) > 1200:
                        scale = 1200 / max(h, w)
                        new_w, new_h = int(w * scale), int(h * scale)
                        result = cv2.resize(result, (new_w, new_h))
                    
                    cv2.imshow('Faster R-CNN Detection Result', result)
                    print("Press any key to close...")
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
            else:
                print("File not found!")
        
        elif choice == '2':
            video_path = input("Video path: ").strip()
            if os.path.exists(video_path):
                detector.detect_video(video_path)
            else:
                print("File not found!")
        
        elif choice == '3':
            detector.detect_webcam()
        
        elif choice == '4':
            folder_path = input("Images folder: ").strip()
            if os.path.exists(folder_path):
                image_files = [f for f in os.listdir(folder_path) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                
                if image_files:
                    print(f"Found {len(image_files)} images")
                    
                    for i, img_file in enumerate(image_files, 1):
                        img_path = os.path.join(folder_path, img_file)
                        print(f"\n[{i}/{len(image_files)}] {img_file}")
                        detector.detect_image(img_path)
                    
                    print("Batch processing complete!")
                else:
                    print("No images found!")
            else:
                print("Folder not found!")
        
        elif choice == '5':
            print(f"\nModel Information:")
            print(f"Framework: Torchvision {torchvision.__version__}")
            print(f"PyTorch: {torch.__version__}")
            print(f"Main Model: Faster R-CNN ({detector.main_detector.model_type})")
            print(f"Status: {'Loaded' if detector.main_detector.is_loaded else 'Failed'}")
            print(f"VN Signs: {'Loaded' if detector.vn_detector.is_loaded else 'Not available'}")
            print(f"Device: {DEVICE}")
            
        elif choice == '0':
            print("Goodbye!")
            break
        
        else:
            print("Invalid option!")

if __name__ == "__main__":
    main()