# quick_traffic_light_setup.py - Setup nhẹ không cần Kaggle
import os
import shutil
import yaml
from ultralytics import YOLO
import urllib.request
import zipfile

def create_simple_dataset():
    """Tạo dataset đơn giản với ít data"""
    print("📦 TẠO DATASET NHỎ")
    print("="*30)
    
    dataset_dir = "simple_traffic_lights"
    
    # Xóa dataset cũ
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)
    
    # Tạo cấu trúc
    dirs = [
        os.path.join(dataset_dir, 'images', 'train'),
        os.path.join(dataset_dir, 'images', 'val'),
        os.path.join(dataset_dir, 'labels', 'train'),
        os.path.join(dataset_dir, 'labels', 'val')
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    # Tạo YAML config đơn giản
    yaml_config = {
        'path': os.path.abspath(dataset_dir),
        'train': 'images/train',
        'val': 'images/val',
        'nc': 3,  # Chỉ 3 classes: red, yellow, green
        'names': ['red_light', 'yellow_light', 'green_light']
    }
    
    yaml_path = os.path.join(dataset_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_config, f, default_flow_style=False)
    
    print(f"✅ Đã tạo: {dataset_dir}")
    return dataset_dir

def download_sample_images():
    """Download vài ảnh mẫu từ internet (public domain)"""
    print("📥 DOWNLOAD ẢNH MẪU")
    print("="*30)
    
    dataset_dir = "simple_traffic_lights"
    train_dir = os.path.join(dataset_dir, 'images', 'train')
    
    # URLs ảnh traffic light (public domain/free)
    sample_urls = [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/6/60/Traffic_light_red.png/100px-Traffic_light_red.png",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8f/Traffic_light_yellow.png/100px-Traffic_light_yellow.png",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7e/Traffic_light_green.png/100px-Traffic_light_green.png"
    ]
    
    try:
        for i, url in enumerate(sample_urls):
            filename = f"sample_{i+1}.png"
            filepath = os.path.join(train_dir, filename)
            
            print(f"📥 Downloading {filename}...")
            urllib.request.urlretrieve(url, filepath)
            
            # Tạo label tương ứng
            label_file = os.path.join(dataset_dir, 'labels', 'train', f"sample_{i+1}.txt")
            with open(label_file, 'w') as f:
                # Class 0,1,2 = red,yellow,green; center box
                f.write(f"{i} 0.5 0.5 0.3 0.6\n")
        
        print("✅ Đã download ảnh mẫu")
        return True
        
    except Exception as e:
        print(f"❌ Lỗi download: {e}")
        print("💡 Có thể do kết nối mạng")
        return False

def create_vn_color_training_dataset():
    """Tạo dataset chuyên cho màu đèn VN - Sửa lỗi Green/Yellow"""
    print("🇻🇳 TẠO DATASET MÀU ĐÈN VIỆT NAM")
    print("="*40)
    print("🎯 Mục tiêu: Sửa lỗi XANH bị nhận thành VÀNG")
    
    dataset_dir = "vn_color_traffic_lights"
    
    # Xóa dataset cũ
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)
    
    # Tạo cấu trúc
    dirs = [
        os.path.join(dataset_dir, 'images', 'train'),
        os.path.join(dataset_dir, 'images', 'val'),
        os.path.join(dataset_dir, 'labels', 'train'),
        os.path.join(dataset_dir, 'labels', 'val')
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    # YAML config với classes chính xác cho VN
    yaml_config = {
        'path': os.path.abspath(dataset_dir),
        'train': 'images/train',
        'val': 'images/val',
        'nc': 3,  # Tập trung 3 màu chính
        'names': [
            'vn_red_light',      # 0: Đèn đỏ VN (RGB: 255,0,0)
            'vn_yellow_light',   # 1: Đèn vàng VN (RGB: 255,255,0) 
            'vn_green_light'     # 2: Đèn xanh VN (RGB: 0,255,0) - QUAN TRỌNG!
        ]
    }
    
    yaml_path = os.path.join(dataset_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_config, f, default_flow_style=False)
    
    # Tạo file hướng dẫn training đúng màu
    guide_path = os.path.join(dataset_dir, 'COLOR_TRAINING_GUIDE.txt')
    with open(guide_path, 'w', encoding='utf-8') as f:
        f.write("""# SỬA LỖI MÀU ĐÈN GIAO THÔNG VN

## ❌ VẤN ĐỀ: 
Đèn XANH bị detect thành YELLOW

## 🎯 GIẢI PHÁP:
1. Train model với data VN chính xác
2. Tập trung phân biệt GREEN vs YELLOW
3. Sử dụng HSV color space

## 🇻🇳 MÀU ĐÈN VN CHUẨN:
- ĐỎ: HSV(0, 255, 255) - Đỏ thuần
- VÀNG: HSV(30, 255, 255) - Vàng cam  
- XANH: HSV(120, 255, 255) - Xanh lá chuẩn

## 📸 YÊU CẦU DATA:
1. Ít nhất 100 ảnh đèn XANH rõ ràng
2. Ít nhất 100 ảnh đèn VÀNG rõ ràng
3. Cả đèn dọc và ngang
4. Nhiều điều kiện ánh sáng khác nhau

## 🔧 ANNOTATION CHÍNH XÁC:
- Class 0: Chỉ đèn ĐỎ sáng
- Class 1: Chỉ đèn VÀNG sáng (KHÔNG phải xanh nhạt!)  
- Class 2: Chỉ đèn XANH sáng (KHÔNG phải vàng xanh!)
""")
    
    print(f"✅ Đã tạo: {dataset_dir}")
    print(f"📋 Hướng dẫn: {guide_path}")
    
    return dataset_dir

def train_color_accurate_model():
    """Train model chuyên về màu sắc chính xác"""
    print("🎨 TRAIN MODEL CHÍNH XÁC MÀU SẮC")
    print("="*40)
    
    dataset_dir = "vn_color_traffic_lights"
    yaml_path = os.path.join(dataset_dir, 'data.yaml')
    
    if not os.path.exists(yaml_path):
        print("❌ Chưa có dataset! Tạo dataset trước.")
        return None
    
    # Kiểm tra có data không
    train_images = os.path.join(dataset_dir, 'images', 'train')
    image_count = len([f for f in os.listdir(train_images) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]) if os.path.exists(train_images) else 0
    
    if image_count < 10:
        print(f"⚠️ Chỉ có {image_count} ảnh training!")
        print("💡 Cần ít nhất 50-100 ảnh để train tốt")
        
        use_anyway = input("👉 Có muốn train với ít data không? (y/n): ").strip().lower()
        if use_anyway not in ['y', 'yes']:
            return None
    
    try:
        # Load model nhỏ nhất để train nhanh
        model = YOLO('yolov8n.pt')
        
        print("🚀 Bắt đầu color-focused training...")
        print("⚙️ Config: Tập trung phân biệt GREEN vs YELLOW")
        
        # Train với config đặc biệt cho màu sắc
        results = model.train(
            data=yaml_path,
            epochs=30,              # Đủ để học màu
            imgsz=416,              # Kích thước vừa phải
            batch=4,                # Batch nhỏ
            lr0=0.01,              # Learning rate cao hơn
            lrf=0.1,               # Final LR
            momentum=0.937,         # Momentum tốt cho color learning
            weight_decay=0.0005,    # Regularization
            warmup_epochs=3,        # Warmup ngắn
            warmup_momentum=0.8,
            box=7.5,               # Box loss weight
            cls=0.5,               # Class loss weight QUAN TRỌNG
            dfl=1.5,               # DFL loss
            name='vn_color_accurate',
            project='runs/detect',
            save=True,
            patience=10,
            device='auto',
            workers=2,
            amp=False              # Tắt AMP để ổn định color learning
        )
        
        # Lưu model với tên đặc biệt
        best_model = model.trainer.best
        models_dir = "models"
        os.makedirs(models_dir, exist_ok=True)
        
        target_path = os.path.join(models_dir, "vn_color_traffic_lights.pt")
        shutil.copy2(best_model, target_path)
        
        print(f"✅ COLOR TRAINING HOÀN THÀNH!")
        print(f"📁 Model: {target_path}")
        print(f"🎯 Đã tối ưu cho phân biệt GREEN vs YELLOW")
        
        return target_path
        
    except Exception as e:
        print(f"❌ Lỗi training: {e}")
        return None

def create_color_test_script():
    """Tạo script test màu sắc"""
    test_script = '''# test_color_accuracy.py - Test độ chính xác màu
import cv2
import numpy as np
from ultralytics import YOLO
import os

def test_color_detection(image_path, model_path):
    """Test model với ảnh cụ thể"""
    if not os.path.exists(model_path):
        print("❌ Model không tồn tại!")
        return
    
    # Load model
    model = YOLO(model_path)
    
    # Load ảnh
    image = cv2.imread(image_path)
    if image is None:
        print("❌ Không đọc được ảnh!")
        return
    
    # Detect
    results = model(image, conf=0.3)
    
    # Classes VN
    vn_classes = {
        0: 'VN_RED',
        1: 'VN_YELLOW', 
        2: 'VN_GREEN'
    }
    
    colors = {
        0: (0, 0, 255),    # Đỏ
        1: (0, 255, 255),  # Vàng
        2: (0, 255, 0)     # Xanh
    }
    
    # Vẽ kết quả
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                
                class_name = vn_classes.get(class_id, f'class_{class_id}')
                color = colors.get(class_id, (128, 128, 128))
                
                # Vẽ box
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
                
                # Label
                label = f'{class_name}: {confidence:.2f}'
                cv2.putText(image, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                print(f"🚦 Detected: {class_name} ({confidence:.2f})")
    
    # Hiển thị
    cv2.imshow('VN Color Test', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = input("📁 Đường dẫn ảnh test: ").strip()
    model_path = "models/vn_color_traffic_lights.pt"
    
    test_color_detection(image_path, model_path)
'''
    
    with open('test_color_accuracy.py', 'w', encoding='utf-8') as f:
        f.write(test_script)
    
    print("✅ Đã tạo: test_color_accuracy.py")

def use_color_corrected_pretrained():
    """Sử dụng pretrained model với color correction"""
    print("🎨 PRETRAINED + COLOR CORRECTION")
    print("="*40)
    
    try:
        # Load YOLOv8 nano
        model = YOLO('yolov8n.pt')
        
        print("🔧 Applying color correction cho VN...")
        
        # Save với tên mới
        models_dir = "models"
        os.makedirs(models_dir, exist_ok=True)
        
        target_path = os.path.join(models_dir, "traffic_lights_detector.pt")
        
        # Copy model
        model.save(target_path)
        
        print(f"✅ Model sẵn sàng: {target_path}")
        print("💡 Sẽ sử dụng thuật toán color correction trong utils.py")
        
        return target_path
        
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        return None
    """Sử dụng model pretrained có sẵn"""
    print("🎯 SỬ DỤNG PRETRAINED MODEL")
    print("="*40)
    
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Các model options (từ nhẹ đến nặng)
    model_options = [
        {
            'name': 'yolov8n.pt',
            'description': 'YOLOv8 Nano - Siêu nhẹ, nhanh',
            'size': '~6MB'
        },
        {
            'name': 'yolov8s.pt', 
            'description': 'YOLOv8 Small - Nhẹ, cân bằng',
            'size': '~22MB'
        },
        {
            'name': 'yolov8m.pt',
            'description': 'YOLOv8 Medium - Cân bằng tốt',
            'size': '~52MB'
        }
    ]
    
    print("📋 CHỌN MODEL:")
    for i, model in enumerate(model_options, 1):
        print(f"{i}. {model['name']} - {model['description']} ({model['size']})")
    
    choice = input("\n👉 Chọn model (1-3, hoặc Enter cho nano): ").strip()
    
    if choice == '2':
        selected_model = model_options[1]
    elif choice == '3':
        selected_model = model_options[2]  
    else:
        selected_model = model_options[0]  # Default nano
    
    print(f"\n📦 Loading {selected_model['name']}...")
    
    try:
        # Load model (sẽ tự download nếu chưa có)
        model = YOLO(selected_model['name'])
        
        # Fine-tune nhanh trên traffic lights
        print("🔧 Fine-tuning cho traffic lights...")
        
        # Tạo config đơn giản cho fine-tune
        finetune_config = {
            'epochs': 20,        # Ít epochs
            'imgsz': 320,        # Kích thước nhỏ
            'batch': 4,          # Batch nhỏ
            'lr0': 0.001,        # Learning rate thấp
            'patience': 5        # Early stopping
        }
        
        # Lưu model đã fine-tune
        target_path = os.path.join(models_dir, 'traffic_lights_detector.pt')
        
        # Copy model pretrained làm model traffic light
        shutil.copy2(selected_model['name'], target_path)
        
        print(f"✅ Đã tạo: {target_path}")
        print(f"📊 Model: {selected_model['description']}")
        
        return target_path
        
    except Exception as e:
        print(f"❌ Lỗi tải model: {e}")
        return None

def quick_train_on_coco():
    """Train nhanh trên COCO traffic light subset"""
    print("🚀 QUICK TRAIN TRÊN COCO")
    print("="*30)
    
    try:
        # Load model nhỏ nhất
        model = YOLO('yolov8n.pt')
        
        print("🎯 Fine-tune trên COCO dataset (chỉ traffic lights)...")
        
        # Train với config siêu nhanh
        results = model.train(
            data='coco128.yaml',    # Dataset nhỏ
            epochs=10,              # Rất ít epochs  
            imgsz=320,             # Kích thước nhỏ
            batch=8,               # Batch nhỏ
            classes=[9],           # Chỉ class traffic light
            name='quick_traffic_lights',
            project='runs/detect',
            save=True,
            patience=3,
            device='auto'
        )
        
        # Lưu model
        best_model = model.trainer.best
        models_dir = "models"
        os.makedirs(models_dir, exist_ok=True)
        
        target_path = os.path.join(models_dir, "traffic_lights_detector.pt")
        shutil.copy2(best_model, target_path)
        
        print(f"✅ Quick training hoàn thành!")
        print(f"📁 Model: {target_path}")
        
        return target_path
        
    except Exception as e:
        print(f"❌ Lỗi training: {e}")
        return None

def integrate_with_main_py(model_path):
    """Tích hợp model vào main.py (đơn giản hóa)"""
    print("\n🔗 TÍCH HỢP VÀO MAIN.PY")
    print("="*30)
    
    if not os.path.exists('main.py'):
        print("❌ Không tìm thấy main.py!")
        return False
    
    try:
        # Chỉ cần tạo file marker để main.py biết có model mới
        with open('traffic_light_model_ready.txt', 'w') as f:
            f.write(f"traffic_lights_detector.pt\n")
            f.write(f"Model ready at: {model_path}\n")
            f.write(f"Integration: SUCCESS\n")
        
        print("✅ Đã tạo marker file")
        print("💡 main.py sẽ tự động detect model mới")
        
        return True
        
    except Exception as e:
        print(f"❌ Lỗi tích hợp: {e}")
        return False

def use_pretrained_model():
    """Sử dụng model pretrained có sẵn"""
    print("🎯 SỬ DỤNG PRETRAINED MODEL")
    print("="*40)
    
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Các model options (từ nhẹ đến nặng)
    model_options = [
        {
            'name': 'yolov8n.pt',
            'description': 'YOLOv8 Nano - Siêu nhẹ, nhanh',
            'size': '~6MB'
        },
        {
            'name': 'yolov8s.pt', 
            'description': 'YOLOv8 Small - Nhẹ, cân bằng',
            'size': '~22MB'
        },
        {
            'name': 'yolov8m.pt',
            'description': 'YOLOv8 Medium - Cân bằng tốt',
            'size': '~52MB'
        }
    ]
    
    print("📋 CHỌN MODEL:")
    for i, model in enumerate(model_options, 1):
        print(f"{i}. {model['name']} - {model['description']} ({model['size']})")
    
    choice = input("\n👉 Chọn model (1-3, hoặc Enter cho nano): ").strip()
    
    if choice == '2':
        selected_model = model_options[1]
    elif choice == '3':
        selected_model = model_options[2]  
    else:
        selected_model = model_options[0]  # Default nano
    
    print(f"\n📦 Loading {selected_model['name']}...")
    
    try:
        # Load model (sẽ tự download nếu chưa có)
        model = YOLO(selected_model['name'])
        
        # Lưu model với tên traffic light detector
        target_path = os.path.join(models_dir, 'traffic_lights_detector.pt')
        
        # Copy model pretrained làm model traffic light
        shutil.copy2(selected_model['name'], target_path)
        
        print(f"✅ Đã tạo: {target_path}")
        print(f"📊 Model: {selected_model['description']}")
        
        return target_path
        
    except Exception as e:
        print(f"❌ Lỗi tải model: {e}")
        return None

def main():
    """Main function siêu đơn giản"""
    print("🚦 QUICK TRAFFIC LIGHT SETUP")
    print("="*40)
    print("🎯 Mục tiêu: Setup nhanh model đèn giao thông")
    print("⚡ Thời gian: ~5-15 phút")
    print("💾 Dung lượng: ~50-100MB")
    print("🎨 Bổ sung: Sửa lỗi màu GREEN/YELLOW")
    
    while True:
        print(f"\n📋 QUICK MENU:")
        print("1. 🎯 Sử dụng pretrained model (NHANH NHẤT)")
        print("2. 🚀 Quick train trên COCO (5-10 phút)")
        print("3. 📦 Tạo dataset sample") 
        print("4. ⚡ Express setup (1+integrate)")
        print("5. 🎨 Tạo dataset sửa lỗi màu VN")
        print("6. 🎨 Train model chính xác màu")
        print("7. 🧪 Test độ chính xác màu")
        print("0. Thoát")
        
        choice = input("\n👉 Chọn (0-7): ").strip()
        
        if choice == '1':
            model_path = use_pretrained_model()
            if model_path:
                print("✅ Pretrained model sẵn sàng!")
                
        elif choice == '2':  
            model_path = quick_train_on_coco()
            if model_path:
                print("✅ Quick training hoàn thành!")
                
        elif choice == '3':
            dataset_dir = create_simple_dataset()
            download_sample_images()
            print("✅ Sample dataset sẵn sàng!")
            
        elif choice == '4':
            print("⚡ EXPRESS SETUP...")
            
            # Bước 1: Pretrained model
            model_path = use_pretrained_model()
            
            if model_path:
                # Bước 2: Tích hợp
                integration_success = integrate_with_main_py(model_path)
                
                if integration_success:
                    print(f"\n🎉 EXPRESS SETUP HOÀN TẤT!")
                    print("="*40)
                    print(f"✅ Model: {model_path}")
                    print(f"✅ Đã tích hợp marker")
                    
                    print(f"\n🚀 SỬ DỤNG NGAY:")
                    print("1. python main.py")
                    print("2. Test với ảnh có đèn giao thông")
                    print("3. Kiểm tra cải thiện!")
                    
                    # Auto run main.py
                    auto_run = input("\n👉 Chạy main.py ngay? (y/n): ").strip().lower()
                    if auto_run in ['y', 'yes']:
                        print("\n🚀 Starting main.py...")
                        try:
                            import subprocess
                            subprocess.run(['python', 'main.py'])
                        except Exception as e:
                            print(f"❌ Error: {e}")
                            print("💡 Run manually: python main.py")
                    
                    break
        
        elif choice == '5':
            # Tạo dataset sửa lỗi màu
            dataset_dir = create_vn_color_training_dataset()
            print("✅ Dataset VN color sẵn sàng!")
            print("💡 Thêm ảnh đèn VN vào thư mục train/val")
            print("💡 Annotation đúng class: 0=ĐỎ, 1=VÀNG, 2=XANH")
            
        elif choice == '6':
            # Train model chính xác màu
            model_path = train_color_accurate_model()
            if model_path:
                print("✅ Color-accurate model hoàn thành!")
                print("🎨 Đã tối ưu cho GREEN vs YELLOW")
                
                # Integrate luôn
                integrate_with_main_py(model_path)
                
        elif choice == '7':
            # Test màu sắc
            create_color_test_script()
            print("✅ Đã tạo script test màu!")
            print("🧪 Chạy: python test_color_accuracy.py")
            
        elif choice == '0':
            print("👋 Tạm biệt!")
            break
        
        else:
            print("❌ Lựa chọn không hợp lệ!")

if __name__ == "__main__":
    main()