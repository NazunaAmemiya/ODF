import argparse
import os
import sys
import yaml
import torch
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

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

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
    
    # 3. Gọi Builder khởi tạo Dataloader từ src.datasets
    logger.info("Khởi tạo Data Pipeline...")
    train_loader = build_dataloader(cfg['dataset'], cfg['dataloader'], is_train=True)
    
    # 4. Gọi Builder khởi tạo Kiến trúc Mô hình từ src.models
    logger.info("Lắp ráp Mô hình thông qua Registry...")
    model = build_model(cfg['model'])
    model.to(device)
    
    # 5. Khởi tạo Optimizer & Scheduler
    optimizer = build_optimizer(model, cfg['optimizer'])
    scheduler_cfg = cfg.get('scheduler', {})
    scheduler = CosineAnnealingLR(optimizer, T_max=scheduler_cfg.get('T_max', 100))
    
    # 6. VÒNG LẶP HUẤN LUYỆN CHÍNH
    epochs = cfg.get('epochs', 100)
    logger.info(f"🔥🔥🔥 BẮT ĐẦU HUẤN LUYỆN TRONG {epochs} EPOCHS 🔥🔥🔥")
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        
        for batch_idx, batch_data in enumerate(train_loader):
            # Tuỳ thuộc vào base_dataset.py xuất ra định dạng gì, bạn sẽ unpack ở đây
            # Giả sử dataloader trả về dict(img=..., gt_bboxes=...)
            images = batch_data['img'].to(device)
            targets = {k: v.to(device) for k, v in batch_data.items() if k != 'img'}
            
            # Forward (Mô hình sẽ tự động tính Loss nếu đang ở chế độ train, theo thiết kế của meta_archs)
            loss = model(images, targets)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # In log mỗi 10 batch
            if batch_idx % cfg['logging'].get('interval', 10) == 0:
                logger.info(f"Epoch [{epoch}/{epochs}] - Batch [{batch_idx}/{len(train_loader)}] - Loss: {loss.item():.4f}")
                
        # Cập nhật learning rate sau mỗi epoch
        scheduler.step()
        
        # Tính toán loss trung bình của epoch
        avg_loss = total_loss / len(train_loader)
        logger.info(f"👉 KẾT THÚC EPOCH {epoch} | Average Loss: {avg_loss:.4f}")
        
        # Lưu checkpoint (Giả lập logic lưu file)
        checkpoint_path = os.path.join(work_dir, f"epoch_{epoch}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        logger.info(f"Đã lưu trọng số tại: {checkpoint_path}")

if __name__ == '__main__':
    main()