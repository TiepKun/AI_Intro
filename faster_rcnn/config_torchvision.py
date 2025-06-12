# config_torchvision.py - Configuration for Torchvision Faster R-CNN
import os
import torch

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
INPUT_DIR = os.path.join(BASE_DIR, 'input')
OUTPUT_DIR = r"D:\output"

# Create directories
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(os.path.join(INPUT_DIR, 'images'), exist_ok=True)
os.makedirs(os.path.join(INPUT_DIR, 'videos'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'images'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'videos'), exist_ok=True)

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Detection parameters
CONFIDENCE_THRESHOLD = 0.5  # Confidence threshold for detections
NMS_THRESHOLD = 0.5         # Non-maximum suppression threshold

# COCO classes (1-indexed in torchvision)
COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A',
    'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet',
    'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Traffic-related classes mapping (1-indexed)
TRAFFIC_CLASSES = {
    1: 'person',
    2: 'bicycle', 
    3: 'car',
    4: 'motorcycle',
    6: 'bus',
    8: 'truck',
    10: 'traffic light',
    13: 'stop sign'
}

# Color mapping for different object types
CLASS_COLORS = {
    1: (255, 0, 0),    # person - red
    2: (0, 255, 0),    # bicycle - green
    3: (0, 0, 255),    # car - blue
    4: (255, 255, 0),  # motorcycle - cyan
    6: (255, 0, 255),  # bus - magenta
    8: (0, 255, 255),  # truck - yellow
    10: (128, 128, 128), # traffic light - gray (will be overridden)
    13: (255, 128, 0)   # stop sign - orange
}

# Traffic light colors
TRAFFIC_LIGHT_COLORS = {
    'red': (0, 0, 255),
    'yellow': (0, 255, 255),
    'green': (0, 255, 0),
    'off': (128, 128, 128),
    'unknown': (64, 64, 64)
}

# Model configuration
MODEL_CONFIG = {
    'resnet50': {
        'name': 'fasterrcnn_resnet50_fpn',
        'description': 'Faster R-CNN with ResNet50 backbone',
        'accuracy': 'High',
        'speed': 'Medium'
    },
    'mobilenet': {
        'name': 'fasterrcnn_mobilenet_v3_large_fpn',
        'description': 'Faster R-CNN with MobileNetV3 backbone',
        'accuracy': 'Medium',
        'speed': 'High'
    }
}

# Performance settings
PERFORMANCE_CONFIG = {
    'video_process_every_n_frames': 3,    # Process every N frames in video
    'webcam_process_every_n_frames': 5,   # Process every N frames in webcam
    'max_image_size': 1024,               # Resize images larger than this
    'webcam_detection_size': 640,         # Size for webcam detection
    'batch_size': 1                       # Batch size for processing
}

# Output settings
OUTPUT_CONFIG = {
    'draw_confidence_scores': True,       # Show confidence scores
    'draw_class_names': True,             # Show class names
    'line_thickness': 2,                  # Bounding box thickness
    'font_scale': 0.6,                    # Text font scale
    'save_raw_predictions': False,        # Save raw model outputs
    'save_cropped_detections': False      # Save cropped detection images
}

# Training settings (for VN signs)
TRAINING_CONFIG = {
    'learning_rate': 0.005,
    'momentum': 0.9,
    'weight_decay': 0.0005,
    'step_size': 3,
    'gamma': 0.1,
    'num_epochs': 10,
    'batch_size': 2,
    'num_workers': 2
}