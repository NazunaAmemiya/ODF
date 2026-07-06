import argparse
import os
import sys
import yaml
import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# BẮT BUỘC: Đẩy thư mục gốc vào sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)

from src.utils.logger import setup_logger
from src.models.builder import build_model
# Nhập module vẽ Bounding Box hoặc Mask (theo sơ đồ thư mục của bạn)
# from src.visualization.det_visualizer import DetVisualizer
# from src.visualization.seg_visualizer import SegVisualizer

def parse_args():
    parser = argparse.ArgumentParser(description='🦟 Suy luận mô hình Mosquito-CV (Inference)')
    parser.add_argument('--config', type=str, required=True, 
                        help='Đường dẫn tới file cấu hình YAML')
    parser.add_argument('--checkpoint', type=str, required=True, 
                        help='Đường dẫn tới file trọng số (.pth)')
    parser.add_argument('--source', type=str, required=True, 
                        help='Đường dẫn tới ảnh, thư mục ảnh, hoặc video cần dự đoán')
    parser.add_argument('--save-dir', type=str, default='./output/predictions', 
                        help='Thư mục lưu kết quả ảnh đã vẽ')
    parser.add_argument('--conf-thres', type=float, default=0.5, 
                        help='Ngưỡng tin cậy (Confidence Threshold)')
    parser.add_argument('--iou-thres', type=float, default=0.45, 
                        help='Ngưỡng NMS IoU để lọc box trùng')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def preprocess_image(image_bgr, input_size):
    """Tiền xử lý ảnh gốc thành Tensor chuẩn để nạp vào mô hình"""
    # 1. Resize về kích thước mạng yêu cầu (VD: 640x640)
    img_resized = cv2.resize(image_bgr, tuple(input_size))
    # 2. Chuyển BGR (OpenCV) sang RGB
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    # 3. Chuyển thành Numpy array, scale [0, 1] và đổi trục (H, W, C) -> (C, H, W)
    img_tensor = img_rgb.astype(np.float32) / 255.0
    img_tensor = np.transpose(img_tensor, (2, 0, 1))
    # 4. Thêm chiều Batch (C, H, W) -> (1, C, H, W)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    return torch.from_numpy(img_tensor)

def main():
    args = parse_args()
    cfg = load_config(args.config)
    
    os.makedirs(args.save_dir, exist_ok=True)
    logger = setup_logger(name="mosquito_cv_pred", save_dir=args.save_dir)
    logger.info("Khởi động quy trình Suy luận (Inference).")
    
    # 1. Khởi tạo Thiết bị & Mô hình
    device = torch.device(cfg.get('device', 'cuda:0' if torch.cuda.is_available() else 'cpu'))
    model = build_model(cfg['model'])
    
    logger.info(f"Đang nạp trọng số từ: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    
    model.to(device)
    model.eval() # Bắt buộc phải có!
    
    # Kích thước đầu vào chuẩn của mô hình (lấy từ config pipeline)
    input_size = cfg.get('pipeline', {}).get('input_size', [640, 640])
    
    # Khởi tạo công cụ vẽ (Giả lập khởi tạo class Visualizer của bạn)
    # visualizer = DetVisualizer(class_names=cfg['dataset']['class_names'])
    
    # 2. Xử lý Nguồn dữ liệu (Source)
    source_path = Path(args.source)
    if source_path.is_file():
        image_paths = [source_path]
    elif source_path.is_dir():
        image_paths = list(source_path.glob('*.*'))
        image_paths = [p for p in image_paths if p.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    else:
        logger.error(f"Không tìm thấy nguồn dữ liệu tại: {args.source}")
        return

    logger.info(f"Tìm thấy {len(image_paths)} ảnh để dự đoán.")

    # 3. VÒNG LẶP SUY LUẬN
    with torch.no_grad():
        for img_path in tqdm(image_paths, desc="Predicting"):
            # Đọc ảnh gốc bằng OpenCV
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue
                
            original_shape = img_bgr.shape[:2] # (H, W) để xíu nữa scale tọa độ ngược lại
            
            # Tiền xử lý
            input_tensor = preprocess_image(img_bgr, input_size).to(device)
            
            # Forward qua mạng (Sẽ trả về Tensor chứa Boxes, Objectness, Classes)
            raw_predictions = model(input_tensor)
            
            # Gọi Decoder để Hậu xử lý (NMS, Lọc ngưỡng)
            # final_predictions = model.decoder(raw_predictions, conf_thres=args.conf_thres, iou_thres=args.iou_thres)
            
            # Chuyển đổi tọa độ box từ (640x640) về kích thước ảnh gốc (original_shape)
            # ... (Logic tùy thuộc vào Decoder của bạn) ...
            
            # Vẽ Box/Mask lên ảnh (Sử dụng module Visualization của bạn)
            # annotated_img = visualizer.draw(img_bgr.copy(), final_predictions)
            
            # Tạm thời lưu ảnh gốc để tránh lỗi do chưa code Visualizer
            annotated_img = img_bgr 
            
            # Lưu ảnh kết quả
            save_path = os.path.join(args.save_dir, img_path.name)
            cv2.imwrite(save_path, annotated_img)

    logger.info(f"🎉 Hoàn tất! Kết quả được lưu tại: {args.save_dir}")

if __name__ == '__main__':
    main()