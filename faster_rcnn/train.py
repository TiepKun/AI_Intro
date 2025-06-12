import os
import zipfile
import json
import cv2
import shutil
import torch
import torchvision
import numpy as np
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader, Dataset

print("🚀 Auto Download & Train từ 2 Dataset Kaggle")
print("=" * 60)
print("📊 Dataset 1: VN Traffic Signs YOLO")
print("📊 Dataset 2: LISA Traffic Lights")
print("=" * 60)

def setup_kaggle():
    """Kiểm tra Kaggle API"""
    try:
        import kaggle
        print("✅ Kaggle API sẵn sàng")
        return True
    except ImportError:
        print("❌ Cần cài: pip install kaggle")
        return False
    except OSError:
        print("❌ Cần setup Kaggle credentials:")
        print("1. Vào https://www.kaggle.com/account")
        print("2. Tạo API key → Download kaggle.json")
        print("3. Đặt vào ~/.kaggle/kaggle.json")
        return False

def download_vn_traffic_signs():
    """Download VN Traffic Signs dataset"""
    print("\n📥 Downloading VN Traffic Signs Dataset...")
    
    try:
        import kaggle
        
        dataset_name = "sangnguyenvan7003/vn-traffic-sign-yolo"
        output_path = "vn_dataset"
        
        print(f"📦 Đang download: {dataset_name}")
        kaggle.api.dataset_download_files(
            dataset_name,
            path=output_path,
            unzip=True
        )
        
        print(f"✅ VN dataset downloaded: {output_path}")
        
        # Khám phá cấu trúc
        print("\n🔍 Khám phá cấu trúc VN dataset:")
        explore_folder_structure(output_path)
        
        return output_path
        
    except Exception as e:
        print(f"❌ Download VN dataset failed: {e}")
        return None

def download_lisa_traffic_lights():
    """Download LISA Traffic Lights dataset"""
    print("\n📥 Downloading LISA Traffic Lights Dataset...")
    
    try:
        import kaggle
        
        # LISA dataset có thể từ notebook này hoặc dataset khác
        # https://www.kaggle.com/code/stpeteishii/lisa-traffic-light-data-animation
        
        # Thử tìm LISA dataset
        lisa_datasets = [
            "mbornoe/lisa-traffic-light-dataset",
            "prateekiiest/lisa-traffic-light-dataset", 
            "andrewmvd/lisa-traffic-light-dataset"
        ]
        
        lisa_path = None
        
        for dataset_name in lisa_datasets:
            try:
                output_path = "lisa_dataset"
                print(f"📦 Trying to download: {dataset_name}")
                
                kaggle.api.dataset_download_files(
                    dataset_name,
                    path=output_path,
                    unzip=True
                )
                
                print(f"✅ LISA dataset downloaded: {output_path}")
                lisa_path = output_path
                break
                
            except Exception as e:
                print(f"❌ Failed {dataset_name}: {e}")
                continue
        
        if lisa_path:
            print("\n🔍 Khám phá cấu trúc LISA dataset:")
            explore_folder_structure(lisa_path)
            return lisa_path
        else:
            print("⚠️ Không download được LISA dataset, chỉ dùng VN dataset")
            return None
            
    except Exception as e:
        print(f"❌ Download LISA failed: {e}")
        return None

def explore_folder_structure(folder_path):
    """Khám phá cấu trúc thư mục"""
    
    if not os.path.exists(folder_path):
        print(f"❌ Thư mục không tồn tại: {folder_path}")
        return
    
    print(f"📁 Cấu trúc {folder_path}:")
    
    # Đếm file theo loại
    image_count = 0
    txt_count = 0
    total_files = 0
    
    for root, dirs, files in os.walk(folder_path):
        level = root.replace(folder_path, '').count(os.sep)
        indent = "  " * level
        folder_name = os.path.basename(root) or folder_path
        
        # Đếm file trong thư mục này
        images_here = len([f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
        txts_here = len([f for f in files if f.endswith('.txt')])
        
        print(f"{indent}📂 {folder_name}/ ({len(files)} files, {images_here} images, {txts_here} txts)")
        
        # Hiển thị vài file mẫu
        if files and level < 3:  # Chỉ hiển thị 3 level đầu
            sample_files = files[:3]
            for file in sample_files:
                print(f"{indent}  📄 {file}")
            if len(files) > 3:
                print(f"{indent}  ... và {len(files)-3} files khác")
        
        # Tích lũy số liệu
        image_count += images_here
        txt_count += txts_here
        total_files += len(files)
    
    print(f"\n📊 Tổng cộng:")
    print(f"  📸 Images: {image_count}")
    print(f"  📝 TXT files: {txt_count}")
    print(f"  📄 Total files: {total_files}")

def collect_all_images_and_labels(dataset_paths):
    """Gom tất cả ảnh và labels từ các dataset"""
    print("\n📦 Gom tất cả ảnh và labels...")
    
    all_images = []
    all_labels = []
    
    for dataset_path in dataset_paths:
        if not dataset_path or not os.path.exists(dataset_path):
            continue
            
        print(f"🔍 Xử lý: {dataset_path}")
        
        # Tìm tất cả ảnh
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                file_path = os.path.join(root, file)
                
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    all_images.append(file_path)
                    
                    # Tìm label tương ứng
                    base_name = os.path.splitext(file)[0]
                    label_file = base_name + '.txt'
                    label_path = os.path.join(root, label_file)
                    
                    # Hoặc tìm trong thư mục labels
                    if not os.path.exists(label_path):
                        parent_dir = os.path.dirname(root)
                        labels_dir = os.path.join(parent_dir, "labels")
                        if os.path.exists(labels_dir):
                            label_path = os.path.join(labels_dir, label_file)
                    
                    # Hoặc tìm theo pattern khác
                    if not os.path.exists(label_path):
                        possible_label_paths = [
                            root.replace("images", "labels") + "/" + label_file,
                            root.replace("train", "labels/train") + "/" + label_file,
                            root.replace("val", "labels/val") + "/" + label_file
                        ]
                        
                        for possible_path in possible_label_paths:
                            if os.path.exists(possible_path):
                                label_path = possible_path
                                break
                    
                    if os.path.exists(label_path):
                        all_labels.append(label_path)
                    else:
                        all_labels.append(None)  # Không có label
    
    print(f"✅ Tìm thấy:")
    print(f"  📸 {len(all_images)} ảnh")
    print(f"  📝 {len([l for l in all_labels if l])} labels")
    print(f"  ⚠️ {len([l for l in all_labels if not l])} ảnh không có label")
    
    return all_images, all_labels

class CombinedDataset(Dataset):
    """Dataset kết hợp từ nhiều nguồn"""
    
    def __init__(self, image_paths, label_paths):
        self.image_paths = image_paths
        self.label_paths = label_paths
        
        # Loại bỏ ảnh không có label
        self.valid_pairs = []
        for img_path, label_path in zip(image_paths, label_paths):
            if label_path and os.path.exists(label_path):
                self.valid_pairs.append((img_path, label_path))
        
        print(f"📊 Dataset có {len(self.valid_pairs)} cặp ảnh-label hợp lệ")
        
        # Classes - tạm thời dùng VN classes, sẽ mở rộng cho LISA
        self.classes = {
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
            49: 'p124c', 50: 'w201b', 51: 'w246c',
            # LISA traffic lights classes
            52: 'traffic_light_red', 53: 'traffic_light_yellow', 54: 'traffic_light_green'
        }
        
        self.num_classes = len(self.classes)
    
    def __len__(self):
        return len(self.valid_pairs)
    
    def __getitem__(self, idx):
        img_path, label_path = self.valid_pairs[idx]
        
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            # Fallback image
            image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        image = F.to_tensor(image)
        
        # Load labels
        boxes = []
        labels = []
        
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    # Convert YOLO to absolute coordinates
                    x_center_abs = x_center * w
                    y_center_abs = y_center * h
                    width_abs = width * w
                    height_abs = height * h
                    
                    x1 = x_center_abs - width_abs / 2
                    y1 = y_center_abs - height_abs / 2
                    x2 = x_center_abs + width_abs / 2
                    y2 = y_center_abs + height_abs / 2
                    
                    # Clamp to image bounds
                    x1 = max(0, min(x1, w-1))
                    y1 = max(0, min(y1, h-1))
                    x2 = max(x1+1, min(x2, w))
                    y2 = max(y1+1, min(y2, h))
                    
                    boxes.append([x1, y1, x2, y2])
                    labels.append(class_id + 1)  # +1 vì 0 là background
        
        except Exception as e:
            print(f"⚠️ Lỗi đọc label {label_path}: {e}")
        
        # Ensure we have at least one box
        if len(boxes) == 0:
            # Create dummy box
            boxes.append([w//4, h//4, 3*w//4, 3*h//4])
            labels.append(1)  # Dummy class
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx])
        }
        
        return image, target

def collate_fn(batch):
    return tuple(zip(*batch))

def train_combined_model(dataset, num_epochs=10):
    """Train model với combined dataset"""
    print(f"\n🚂 Training Faster R-CNN với Combined Dataset...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Device: {device}")
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    print(f"📊 Train: {len(train_dataset)} samples")
    print(f"📊 Val: {len(val_dataset)} samples")
    
    # Create Faster R-CNN model
    model = fasterrcnn_resnet50_fpn(weights='DEFAULT')
    
    # Modify for our classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, dataset.num_classes + 1)
    
    model.to(device)
    
    # Optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005
    )
    
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,
        gamma=0.1
    )
    
    # Training loop với auto-save
    print(f"🏃 Starting training for {num_epochs} epochs...")
    print("💡 Model sẽ được save sau mỗi epoch để tránh mất dữ liệu!")
    
    # Check for existing checkpoint
    checkpoint_path = "models/checkpoint.pth"
    start_epoch = 0
    
    if os.path.exists(checkpoint_path):
        resume = input(f"🔄 Tìm thấy checkpoint. Resume training? (y/n): ").strip().lower()
        if resume in ['y', 'yes']:
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                start_epoch = checkpoint['epoch']
                print(f"✅ Resumed from epoch {start_epoch}")
            except Exception as e:
                print(f"⚠️ Không thể load checkpoint: {e}")
                start_epoch = 0
    
    for epoch in range(start_epoch, num_epochs):
        print(f"\n📅 Epoch {epoch+1}/{num_epochs}")
        
        model.train()
        epoch_loss = 0
        num_batches = 0
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            try:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                # Forward pass
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                
                # Backward pass
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                
                epoch_loss += losses.item()
                num_batches += 1
                
                if batch_idx % 10 == 0:
                    print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {losses.item():.4f}")
                    
            except KeyboardInterrupt:
                print(f"\n⚠️ Training bị dừng bởi Ctrl+C")
                print(f"💾 Đang save checkpoint...")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': lr_scheduler.state_dict(),
                    'num_classes': dataset.num_classes,
                    'loss': epoch_loss / num_batches if num_batches > 0 else 0
                }, checkpoint_path)
                print(f"✅ Checkpoint saved: {checkpoint_path}")
                print("🔄 Chạy lại script để resume training!")
                return checkpoint_path
            except Exception as e:
                print(f"⚠️ Error in batch {batch_idx}: {e}")
                continue
        
        lr_scheduler.step()
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        print(f"✅ Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")
        
        # Auto-save checkpoint sau mỗi epoch
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict(),
            'num_classes': dataset.num_classes,
            'loss': avg_loss
        }, checkpoint_path)
        print(f"💾 Checkpoint saved: epoch_{epoch+1}")
        
        # Save best model nếu loss giảm
        best_model_path = "models/best_model.pth"
        if not hasattr(train_combined_model, 'best_loss') or avg_loss < train_combined_model.best_loss:
            train_combined_model.best_loss = avg_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict(),
                'num_classes': dataset.num_classes,
                'loss': avg_loss
            }, best_model_path)
            print(f"🏆 New best model saved! Loss: {avg_loss:.4f}")
        
        # Validation
        if (epoch + 1) % 2 == 0 and len(val_loader) > 0:
            print("🔍 Running validation...")
            model.eval()
            val_loss = 0
            val_batches = 0
            
            with torch.no_grad():
                for images, targets in val_loader:
                    try:
                        images = [img.to(device) for img in images]
                        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                        
                        loss_dict = model(images, targets)
                        losses = sum(loss for loss in loss_dict.values())
                        val_loss += losses.item()
                        val_batches += 1
                    except:
                        continue
            
            avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
            print(f"📊 Validation Loss: {avg_val_loss:.4f}")
    
    # Save model sau mỗi epoch (auto-save)
    os.makedirs("models", exist_ok=True)
    final_model_path = "models/vn_signs_faster_rcnn.pth"
    
    # Save final model
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': lr_scheduler.state_dict(),
        'num_classes': dataset.num_classes
    }, final_model_path)
    
    print(f"💾 Final model saved: {final_model_path}")
    return final_model_path

def resume_training():
    """Resume training từ checkpoint"""
    print("🔄 Resume Training Mode")
    print("=" * 40)
    
    checkpoint_path = "models/checkpoint.pth"
    
    if not os.path.exists(checkpoint_path):
        print("❌ Không tìm thấy checkpoint!")
        print("💡 Chạy training mới với option 1")
        return
    
    try:
        # Load checkpoint info
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        saved_epoch = checkpoint['epoch']
        saved_loss = checkpoint['loss']
        num_classes = checkpoint['num_classes']
        
        print(f"📊 Checkpoint info:")
        print(f"  Last epoch: {saved_epoch}")
        print(f"  Last loss: {saved_loss:.4f}")
        print(f"  Classes: {num_classes}")
        
        # Tìm dataset
        dataset_paths = []
        for folder in ["vn_dataset", "lisa_dataset"]:
            if os.path.exists(folder):
                dataset_paths.append(folder)
        
        if not dataset_paths:
            print("❌ Không tìm thấy dataset!")
            print("💡 Cần download dataset trước")
            return
        
        print(f"✅ Tìm thấy dataset: {dataset_paths}")
        
        # Recreate dataset
        all_images, all_labels = collect_all_images_and_labels(dataset_paths)
        combined_dataset = CombinedDataset(all_images, all_labels)
        
        # Continue training
        additional_epochs = int(input(f"🔢 Số epochs thêm (default=5): ") or "5")
        total_epochs = saved_epoch + additional_epochs
        
        print(f"🚀 Resume training từ epoch {saved_epoch} đến epoch {total_epochs}")
        
        model_path = train_combined_model(combined_dataset, total_epochs)
        
        if model_path:
            print(f"✅ Resume training hoàn thành!")
            print(f"📁 Model: {model_path}")
        
    except Exception as e:
        print(f"❌ Lỗi resume training: {e}")

def main():
    """Main function với resume option"""
    
    print("🎯 Auto Download & Train từ Kaggle")
    print("Dataset 1: https://www.kaggle.com/datasets/sangnguyenvan7003/vn-traffic-sign-yolo")
    print("Dataset 2: https://www.kaggle.com/code/stpeteishii/lisa-traffic-light-data-animation")
    print("=" * 80)
    
    # Check for existing checkpoint
    if os.path.exists("models/checkpoint.pth"):
        print("🔄 Tìm thấy checkpoint từ lần train trước!")
        print("1. Train mới (xóa checkpoint cũ)")
        print("2. Resume training từ checkpoint")
        
        choice = input("👉 Chọn (1/2): ").strip()
        
        if choice == '2':
            resume_training()
            return
        else:
            # Xóa checkpoint cũ
            try:
                os.remove("models/checkpoint.pth")
                print("🗑️ Đã xóa checkpoint cũ")
            except:
                pass
    
    # 1. Setup Kaggle
    if not setup_kaggle():
        print("❌ Cần setup Kaggle API trước!")
        return
    
    # 2. Download datasets (hoặc sử dụng existing)
    vn_path = None
    lisa_path = None
    
    if os.path.exists("vn_dataset"):
        print("✅ VN dataset đã tồn tại")
        vn_path = "vn_dataset"
    else:
        vn_path = download_vn_traffic_signs()
    
    if os.path.exists("lisa_dataset"):
        print("✅ LISA dataset đã tồn tại")
        lisa_path = "lisa_dataset"
    else:
        lisa_path = download_lisa_traffic_lights()
    
    dataset_paths = [path for path in [vn_path, lisa_path] if path]
    
    if not dataset_paths:
        print("❌ Không có dataset nào!")
        return
    
    # 3. Collect all images and labels
    all_images, all_labels = collect_all_images_and_labels(dataset_paths)
    
    if len(all_images) == 0:
        print("❌ Không tìm thấy ảnh nào!")
        return
    
    # 4. Create combined dataset
    combined_dataset = CombinedDataset(all_images, all_labels)
    
    if len(combined_dataset) == 0:
        print("❌ Dataset trống!")
        return
    
    # 5. Train model
    epochs = input(f"\n🔢 Số epochs (default=10): ").strip()
    try:
        epochs = int(epochs) if epochs else 10
    except:
        epochs = 10
    
    print(f"\n🚀 Bắt đầu training...")
    print(f"📊 Total samples: {len(combined_dataset)}")
    print(f"📊 Classes: {combined_dataset.num_classes}")
    print(f"🔢 Epochs: {epochs}")
    print(f"💾 Auto-save: Enabled (mỗi epoch)")
    print(f"🔄 Resume: Có thể Ctrl+C và resume sau")
    
    model_path = train_combined_model(combined_dataset, epochs)
    
    if model_path:
        print(f"\n🎉 Training hoàn thành!")
        print(f"✅ Model: {model_path}")
        print(f"🚀 Sẵn sàng dùng với main.py!")
        
        # Cleanup option
        cleanup = input("\n🗑️ Xóa dataset folders để tiết kiệm dung lượng? (y/n): ").strip().lower()
        if cleanup in ['y', 'yes']:
            for path in dataset_paths:
                if os.path.exists(path):
                    try:
                        shutil.rmtree(path)
                        print(f"🗑️ Đã xóa: {path}")
                    except:
                        print(f"⚠️ Không thể xóa: {path}")
                        
        # Cleanup checkpoint
        if os.path.exists("models/checkpoint.pth"):
            os.remove("models/checkpoint.pth")
            print("🗑️ Đã xóa checkpoint (training hoàn thành)")
    else:
        print("❌ Training failed!")

if __name__ == "__main__":
    main()