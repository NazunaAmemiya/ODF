import argparse
import cv2
import yaml
import torch
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.builder import build_model
# --- IMPORT HÀM BUILDER CỦA CHÍNH FRAMEWORK ---
from src.datasets.builder import _build_transforms
from src.visualization.seg_visualizer import SegVisualizer

def parse_args():
    parser = argparse.ArgumentParser(description='🦟 Dự đoán và Phân vùng muỗi')
    parser.add_argument('--config', type=str, required=True, help='Đường dẫn file YAML')
    parser.add_argument('--checkpoint', type=str, required=True, help='Đường dẫn tới file trọng số')
    parser.add_argument('--source', type=str, required=True, help='Đường dẫn tới bức ảnh cần test')
    parser.add_argument('--out', type=str, default='result.jpg', help='Tên file ảnh xuất ra')
    parser.add_argument('--conf-thresh', type=float, default=0.5, help='Ngưỡng tin cậy')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def preprocess_image_auto(image_path, cfg, device):
    """
    Tiền xử lý tự động: Dùng chính code Dataloader của framework để 
    bảo đảm không có 0.001% sai lệch nào so với lúc Train.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"❌ Không thể đọc ảnh: {image_path}")
    
    # BaseMosquitoDataset luôn đọc RGB trước khi đưa vào pipeline
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Lấy thông số từ YAML
    pipeline_cfg = cfg.get('pipeline', {})
    input_size = pipeline_cfg.get('input_size', [640, 640])
    if isinstance(input_size, list):
        input_size = tuple(input_size)
        
    # Gọi hàm nội bộ để build Pipeline y hệt tập Validation
    transforms = _build_transforms(pipeline_cfg, list(input_size), is_train=False)
    
    # Chạy ảnh qua Pipeline
    sample = {"image": image_rgb}
    processed = transforms(sample) 
    
    # Tensor cuối cùng đã hoàn hảo 100%
    img_tensor = processed["img"].unsqueeze(0).to(device)
    
    # Dùng OpenCV bóp ảnh gốc để lát nữa vẽ khung
    img_draw = cv2.resize(image, (input_size[0], input_size[1]))
    
    return img_tensor, img_draw

def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Đang khởi động hệ thống suy luận trên: {device}")

    # ==========================================
    # 1. KHỞI TẠO MÔ HÌNH
    # ==========================================
    model = build_model(cfg['model'])
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    state_dict = checkpoint.get('state_dict', checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval() 

    # --- ÉP MÔ HÌNH VƯỢT RÀO NMS BÊN TRONG ---
    try:
        if hasattr(model, 'head') and hasattr(model.head, 'test_cfg'):
            model.head.test_cfg['score_thr'] = 0.001
            model.head.test_cfg['conf_thr'] = 0.001
    except:
        pass

    # ==========================================
    # 2. CHUẨN BỊ DỮ LIỆU BẰNG PIPELINE TỰ ĐỘNG
    # ==========================================
    class_names = cfg.get('dataset', {}).get('class_names', ['aedes', 'culex', 'anopheles'])
    img_tensor, img_draw = preprocess_image_auto(args.source, cfg, device)

    print("\n[DEBUG PREDICT] TENSOR ẢNH ĐẦU VÀO:")
    print(f"- Shape: {img_tensor.shape}")
    print(f"- Min value: {img_tensor.min().item():.4f}")
    print(f"- Max value: {img_tensor.max().item():.4f}\n")

    # ==========================================
    # 3. TIẾN HÀNH SUY LUẬN
    # ==========================================
    print("🧠 Trí tuệ nhân tạo đang phân tích ảnh...")
    with torch.no_grad():
        outputs = model(img_tensor) 
        
    predictions = outputs[0]
    all_boxes = predictions.get('boxes', torch.tensor([]))
    all_scores = predictions.get('scores', torch.tensor([]))
    all_class_ids = predictions.get('labels', torch.tensor([]))
    all_masks = predictions.get('masks', torch.tensor([]))

    if len(all_scores) > 0:
        print(f"🧐 [SOI ĐIỂM SỐ]: Điểm tự tin CAO NHẤT AI dự đoán là {all_scores.max().item():.4f}")
    else:
        print("🧐 [SOI ĐIỂM SỐ]: AI xuất ra 0 dự đoán (Tức là mù hoàn toàn).")

    if len(all_boxes) == 0:
        print("🤷‍♂️ AI không tìm thấy đối tượng nào trên ảnh.")
        return

    keep_idx = all_scores >= args.conf_thresh
    boxes = all_boxes[keep_idx]
    scores = all_scores[keep_idx]
    class_ids = all_class_ids[keep_idx]

    if len(all_masks) == len(all_boxes):
        masks = all_masks[keep_idx]
    else:
        masks = all_masks 

    if len(boxes) == 0:
        print(f"🤷‍♂️ Tìm thấy đối tượng nhưng điểm số quá thấp (Dưới {args.conf_thresh}).")
        return

    print(f"🎯 Phát hiện {len(boxes)} vật thể!")

    # ==========================================
    # 4. VẼ KHUNG & MẶT NẠ
    # ==========================================
    visualizer = SegVisualizer(class_names=class_names)

    pred_dict = {
        "boxes":boxes, 
        "masks":masks, 
        "classes":class_ids, 
        "scores":scores
    }
    result_img = visualizer.draw(
        image=img_draw, 
        prediction=pred_dict,
        score_thr=0.25
    )

    cv2.imwrite(args.out, result_img)
    print(f"📸 Đã lưu bức ảnh phân vùng thành công tại: {args.out}")

if __name__ == '__main__':
    main()