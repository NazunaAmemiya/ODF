import argparse
import os
import sys
import yaml
import torch
from tqdm import tqdm  # Thư viện tạo thanh tiến trình cực đẹp

# Đẩy thư mục gốc vào sys.path để gọi được module từ 'src'
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)

from src.utils.logger import setup_logger
from src.datasets.builder import build_dataloader
from src.models.builder import build_model
# Giả định bạn có một Builder cho Metrics trong src/evaluation
# from src.evaluation.builder import build_metric 

def parse_args():
    parser = argparse.ArgumentParser(description='🦟 Đánh giá mô hình Mosquito-CV (Validation)')
    parser.add_argument('--config', type=str, required=True, 
                        help='Đường dẫn tới file cấu hình YAML')
    parser.add_argument('--checkpoint', type=str, required=True, 
                        help='Đường dẫn tới file trọng số (.pth / .pt) cần đánh giá')
    parser.add_argument('--work-dir', type=str, default=None, 
                        help='Thư mục lưu kết quả đánh giá')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    args = parse_args()
    cfg = load_config(args.config)
    
    # 1. Khởi tạo thư mục và Logger
    work_dir = args.work_dir if args.work_dir else cfg.get('work_dir', './work_dirs')
    os.makedirs(work_dir, exist_ok=True)
    logger = setup_logger(name="mosquito_cv_val", save_dir=work_dir)
    logger.info(f"Bắt đầu quy trình Đánh giá (Validation).")
    logger.info(f"Cấu hình: {args.config}")
    logger.info(f"Trọng số: {args.checkpoint}")
    
    # 2. Thiết bị chạy
    device = torch.device(cfg.get('device', 'cuda:0' if torch.cuda.is_available() else 'cpu'))
    
    # 3. Khởi tạo Data Pipeline (CHÚ Ý: is_train=False)
    logger.info("Đang nạp tập dữ liệu Validation...")
    # Thường khi val, batch_size có thể x2 so với train vì không tốn bộ nhớ lưu Gradient
    val_dataloader_cfg = cfg['dataloader'].copy()
    val_dataloader_cfg['shuffle'] = False 
    
    val_loader = build_dataloader(cfg['dataset'], val_dataloader_cfg, is_train=False)
    
    # 4. Lắp ráp Mô hình & Nạp trọng số (Checkpoint)
    logger.info("Khởi tạo mô hình và nạp trọng số...")
    model = build_model(cfg['model'])
    
    # Load weights một cách an toàn (tránh lỗi mismatch nếu train trên GPU mà val trên CPU)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    # Nếu lưu cả dict optimizer trong checkpoint thì chỉ lấy 'state_dict' của model
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    
    model.to(device)
    
    # 5. KÍCH HOẠT CHẾ ĐỘ ĐÁNH GIÁ (RẤT QUAN TRỌNG)
    model.eval()
    
    # Giả lập khởi tạo bộ tính Metric (Ví dụ: mAP cho Detection, mIoU cho Seg)
    # metric_calculator = build_metric(cfg['evaluation'])
    logger.info(f"Đang tiến hành suy luận trên {len(val_loader)} batches...")
    
    results = []
    
    # 6. VÒNG LẶP SUY LUẬN (Không tính Gradient)
    with torch.no_grad():
        # Dùng tqdm để bọc dataloader tạo thanh tiến trình loading %
        for batch_idx, batch_data in enumerate(tqdm(val_loader, desc="Evaluating")):
            images = batch_data['img'].to(device)
            targets = batch_data.get('targets', None)
            
            # Forward pass (Đầu ra lúc này sẽ là Box/Mask dự đoán thay vì Loss)
            predictions = model(images)
            
            # Lưu lại kết quả để xíu nữa tính mAP/mIoU tổng thể
            # metric_calculator.update(predictions, targets)
            
    # 7. TÍNH TOÁN VÀ IN KẾT QUẢ
    logger.info("Đang tính toán các chỉ số Metric...")
    # metrics = metric_calculator.compute()
    
    # Đoạn này tôi giả lập kết quả in ra để bạn hình dung giao diện
    logger.info("-" * 40)
    logger.info("📊 KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP VALIDATION")
    logger.info("-" * 40)
    logger.info(f"  * mAP@50    : 0.895")
    logger.info(f"  * mAP@50-95 : 0.652")
    logger.info(f"  * Recall    : 0.912")
    logger.info("-" * 40)
    
    logger.info("Hoàn tất quá trình đánh giá!")

if __name__ == '__main__':
    main()