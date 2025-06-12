# Cấu hình dự án
import os

# Đường dẫn thư mục
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
INPUT_DIR = os.path.join(BASE_DIR, 'input')

# Thay đổi OUTPUT_DIR thành D:\output
OUTPUT_DIR = r"D:\output"

# Tạo thư mục nếu chưa tồn tại
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(os.path.join(INPUT_DIR, 'images'), exist_ok=True)
os.makedirs(os.path.join(INPUT_DIR, 'videos'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'images'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'videos'), exist_ok=True)

# Cấu hình model
MODEL_NAME = 'yolov8m.pt'  # Sử dụng model Medium cho độ chính xác cao hơn
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

# Cấu hình detection
CONFIDENCE_THRESHOLD = 0.25  # Giảm ngưỡng để phát hiện nhiều hơn
IOU_THRESHOLD = 0.45       # Ngưỡng IoU

# Classes cần detect (COCO dataset)
TRAFFIC_CLASSES = {
    0: 'person',      # Người
    1: 'bicycle',     # Xe đạp
    2: 'car',         # Xe hơi
    3: 'motorcycle',  # Xe máy
    5: 'bus',         # Xe buýt
    7: 'truck',       # Xe tải
    9: 'traffic light' # Đèn giao thông
}

# Cấu hình nâng cao cho phát hiện đèn giao thông
ENHANCED_DETECTION = True  # Bật chế độ phát hiện nâng cao
MIN_TRAFFIC_LIGHT_SIZE = 20  # Kích thước tối thiểu của đèn (pixel)