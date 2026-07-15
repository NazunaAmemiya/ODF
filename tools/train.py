import argparse
import os
import sys
import yaml
import torch
import collections.abc
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

# BẮT BUỘC: Đẩy thư mục gốc của project vào sys.path để Python nhận diện được thư mục 'src'
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)

from src.utils.logger import setup_logger
from src.datasets.builder import build_dataloader
from src.models.builder import build_model

def parse_args():
    parser = argparse.ArgumentParser(description='🦟 Khởi chạy huấn luyện Mosquito-CV')
    parser.add_argument('--config', type=str, required=True, 
                        help='Đường dẫn tới file cấu hình YAML')
    parser.add_argument('--work-dir', type=str, default=None, 
                        help='Ghi đè thư mục lưu kết quả')
    return parser.parse_args()

def update_dict(d, u):
    """Hàm hỗ trợ gộp (merge) 2 dictionary có lồng nhau (nested)."""
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def load_config(config_path):
    """Hàm đọc YAML có hỗ trợ tính năng kế thừa qua từ khóa _base_"""
    # 1. Đọc file YAML hiện tại
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
        
    if cfg is None:
        return {}

    # 2. Kiểm tra xem có từ khóa _base_ không
    if '_base_' in cfg:
        base_files = cfg.pop('_base_') # Lấy danh sách file base ra và xóa khỏi cfg
        
        # Nếu chỉ truyền 1 file dạng chuỗi, chuyển nó thành list
        if isinstance(base_files, str):
            base_files = [base_files]
            
        merged_cfg = {}
        # Lấy thư mục của file config hiện tại làm gốc để xử lý đường dẫn tương đối (../)
        current_dir = os.path.dirname(os.path.abspath(config_path))
        
        # 3. Lặp qua từng file base, đọc đệ quy và gộp chúng lại
        for base_file in base_files:
            base_file_path = os.path.abspath(os.path.join(current_dir, base_file))
            if not os.path.exists(base_file_path):
                raise FileNotFoundError(f"Không tìm thấy file kế thừa: {base_file_path}")
            
            # Gọi đệ quy để đọc file base
            base_cfg = load_config(base_file_path)
            # Gộp cấu hình base vào cấu hình chung
            merged_cfg = update_dict(merged_cfg, base_cfg)
            
        # 4. Lấy cấu hình của file HIỆN TẠI ghi đè lên cấu hình BASE
        cfg = update_dict(merged_cfg, cfg)
        
    return cfg

def build_optimizer(model, optim_cfg):
    """Khởi tạo Optimizer trực tiếp dựa trên config (do sơ đồ không có module engine)"""
    optim_type = optim_cfg.get('type', 'AdamW')
    lr = optim_cfg.get('lr', 0.001)
    weight_decay = optim_cfg.get('weight_decay', 0.0005)
    
    if optim_type == 'AdamW':
        return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optim_type == 'SGD':
        return SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Chưa hỗ trợ Optimizer: {optim_type}")

def main():
    args = parse_args()
    cfg = load_config(args.config)

    # 1. Khởi tạo thư mục làm việc và Logger
    work_dir = args.work_dir if args.work_dir else cfg.get('work_dir', './work_dirs')
    os.makedirs(work_dir, exist_ok=True)
    logger = setup_logger(name="mosquito_cv", save_dir=work_dir)
    logger.info(f"Đọc cấu hình thành công từ: {args.config}")
    
    # 2. Thiết lập phần cứng (GPU/CPU)
    device = torch.device(cfg.get('device', 'cuda:0' if torch.cuda.is_available() else 'cpu'))
    logger.info(f"Đang chạy trên thiết bị: {device}")
    
    # 3. MỚI: Khởi tạo HAI luồng Dataloader (Train và Val)
    logger.info("Khởi tạo Data Pipeline...")
    train_loader = build_dataloader(cfg['dataset'], cfg['dataloader'], is_train=True)
    
    # Ở tập Val, chúng ta thường không cần shuffle và có thể tăng batch_size lên một chút nếu dư RAM
    val_dataloader_cfg = cfg['dataloader'].copy()
    val_dataloader_cfg['shuffle_train'] = False 
    val_loader = build_dataloader(cfg['dataset'], val_dataloader_cfg, is_train=False)
    
    # 4. Lắp ráp Mô hình
    logger.info("Lắp ráp Mô hình thông qua Registry...")
    model = build_model(cfg['model'])
    model.to(device)
    
    # 5. Khởi tạo Optimizer & Scheduler
    optimizer = build_optimizer(model, cfg['optimizer'] if 'optimizer' in cfg else {})
    scheduler_cfg = cfg.get('scheduler', {})
    scheduler = CosineAnnealingLR(optimizer, T_max=scheduler_cfg.get('T_max', 100))
    
    # 6. VÒNG LẶP HUẤN LUYỆN CHÍNH
    epochs = cfg.get('epochs', 150)
    best_val_loss = float('inf')
    logger.info(f"🔥🔥🔥 BẮT ĐẦU HUẤN LUYỆN {epochs} EPOCHS (CÓ VALIDATION) 🔥🔥🔥")
    
    for epoch in range(1, epochs + 1):
        # ==========================================
        # PHA 1: HUẤN LUYỆN (TRAIN)
        # ==========================================
        model.train()
        total_train_loss = 0.0
        
        for batch_idx, batch_data in enumerate(train_loader):
            images = batch_data['img'].to(device)
            targets = {k: v.to(device) for k, v in batch_data.items() if k != 'img'}
            
            loss = model(images, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
            if batch_idx % cfg.get('logging', {}).get('interval', 10) == 0:
                logger.info(f"Epoch [{epoch}/{epochs}] - Train Batch [{batch_idx}/{len(train_loader)}] - Loss: {loss.item():.4f}")
                
        avg_train_loss = total_train_loss / len(train_loader)
        
        # ==========================================
        # PHA 2: KIỂM THỬ (VALIDATION)
        # ==========================================
        model.eval() # Bật chế độ chấm thi (tắt các lớp Dropout/BatchNorm)
        total_val_loss = 0.0
        
        # Tắt tính toán đạo hàm để chạy nhanh hơn và không tốn VRAM
        with torch.no_grad():
            for batch_data in val_loader:
                images = batch_data['img'].to(device)
                targets = {k: v.to(device) for k, v in batch_data.items() if k != 'img'}
                
                # Chỉ tính Loss, không Backward
                val_loss = model(images, targets)
                total_val_loss += val_loss.item()
                
        # Tránh chia cho 0 nếu tập val trống
        avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        
        # Cập nhật learning rate sau mỗi epoch
        scheduler.step()
        
        # ==========================================
        # PHA 3: TỔNG KẾT VÀ LƯU TRỌNG SỐ
        # ==========================================
        logger.info(f"👉 KẾT THÚC EPOCH {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # 1. LƯU MỚI NHẤT: Lưu đè lên cùng một file (Tối ưu ổ cứng)
        # File này dùng để khôi phục tiến trình nếu lỡ bị cúp điện/đứng máy giữa chừng
        latest_checkpoint_path = os.path.join(work_dir, "latest_checkpoint.pth")
        torch.save(model.state_dict(), latest_checkpoint_path)
        
        # 2. KIỂM TRA KỶ LỤC: Nếu Val Loss hiện hành tốt hơn kỷ lục cũ
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss # Cập nhật mức thấp nhất mới
            
            # Lưu riêng một bản "Vàng"
            best_checkpoint_path = os.path.join(work_dir, "best_checkpoint.pth")
            torch.save(model.state_dict(), best_checkpoint_path)
            
            # Thông báo có kỷ lục mới
            logger.info(f"🌟 ĐÃ LƯU MÔ HÌNH TỐT NHẤT MỚI! (Kỷ lục Val Loss: {best_val_loss:.4f})")

if __name__ == '__main__':
    main()