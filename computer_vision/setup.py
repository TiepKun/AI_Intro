# Tạo file setup.py
import os
from pathlib import Path

# Tạo cấu trúc thư mục cho dự án
PROJECT_ROOT = Path(".")
MODEL_PATH = PROJECT_ROOT / "models"
DATA_PATH = PROJECT_ROOT / "data"
OUTPUT_PATH = PROJECT_ROOT / "output"

# Đảm bảo thư mục tồn tại
os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(OUTPUT_PATH, exist_ok=True)

print("Cài đặt môi trường thành công!")
print("Thư mục đã được tạo:")
print(f"- Models: {MODEL_PATH}")
print(f"- Data: {DATA_PATH}")
print(f"- Output: {OUTPUT_PATH}")

# Kiểm tra và cài đặt các thư viện cần thiết
try:
    import cv2
    import numpy as np
    import torch
    print("Các thư viện đã được cài đặt!")
except ImportError:
    print("Vui lòng cài đặt các thư viện cần thiết:")
    print("pip install torch opencv-python numpy ultralytics")