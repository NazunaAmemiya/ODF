import argparse
import cv2
import yaml
import torch
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.builder import build_model
from src.models.decoders.seg_decoder import SegDecoder
from src.visualization.seg_visualizer import SegVisualizer

def parse_args():
    parser = argparse.ArgumentParser(description='🦟 Dự đoán và Phân vùng muỗi (Segmentation Inference)')
    parser.add_argument('--config', type=str, required=True, help='Đường dẫn file YAML')
    parser.add_argument('--checkpoint', type=str, required=True, help='Đường dẫn tới file trọng số')
    parser.add_argument('--source', type=str, required=True, help='Đường dẫn tới bức ảnh cần test')
    parser.add_argument('--out', type=str, default='result.jpg', help='Tên file ảnh xuất ra sau khi vẽ')
    parser.add_argument('--conf-thresh', type=float, default=0.5, help='Ngưỡng tin cậy tối thiểu (mặc định: 0.5)')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def preprocess_image(image_path, input_size, device):
    """Tiền xử lý khớp 100% với Dataloader lúc huấn luyện"""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"❌ Không thể đọc ảnh: {image_path}")
    
    # 1. Resize ép buộc (Bóp méo ảnh)
    img_padded = cv2.resize(img_bgr, (input_size[0], input_size[1]), interpolation=cv2.INTER_LINEAR)
    
    # 2. Chuyển BGR sang RGB
    img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
    
    # 3. Chuyển thành Tensor và chia 255.0
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)
    return img_tensor, img_padded

def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Đang khởi động hệ thống suy luận trên: {device}")

    # ==========================================
    # 1. KHỞI TẠO MÔ HÌNH VÀ NẠP TRỌNG SỐ
    # ==========================================
    model = build_model(cfg['model'])
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    state_dict = checkpoint.get('state_dict', checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    try:
        # Ép mô hình phải nhả ra mọi thứ dù độ tự tin chỉ có 0.1%
        model.head.test_cfg['score_thr'] = 0.001 
        print("🔓 Đã bẻ khóa ngưỡng tin cậy ngầm của mô hình xuống 0.001")
    except:
        pass

    # ==========================================
    # 2. CHUẨN BỊ DỮ LIỆU
    # ==========================================
    class_names = cfg.get('dataset', {}).get('class_names', ['aedes', 'culex', 'anopheles'])
    input_size = cfg.get('pipeline', {}).get('input_size', [640, 640])
    if isinstance(input_size, list):
        input_size = tuple(input_size)

    img_tensor, img_draw = preprocess_image(args.source, input_size, device)

    print("\n[DEBUG PREDICT] TENSOR ẢNH ĐẦU VÀO:")
    print(f"- Shape: {img_tensor.shape}")
    print(f"- Min value: {img_tensor.min().item():.4f}")
    print(f"- Max value: {img_tensor.max().item():.4f}\n")

    # ==========================================
    # 3. TIẾN HÀNH SUY LUẬN & GIẢI MÃ
    # ==========================================
    print("🧠 Trí tuệ nhân tạo đang phân tích ảnh...")
    with torch.no_grad():
        outputs = model(img_tensor) 
        
    predictions = outputs[0]
    all_boxes = predictions.get('boxes', torch.tensor([]))
    all_scores = predictions.get('scores', torch.tensor([]))
    all_class_ids = predictions.get('labels', torch.tensor([]))
    all_masks = predictions.get('masks', torch.tensor([]))

    # --- ĐOẠN DEBUG QUAN TRỌNG NHẤT ---
    if len(all_scores) > 0:
        print(f"🧐 [SOI ĐIỂM SỐ]: Điểm tự tin CAO NHẤT AI dự đoán là {all_scores.max().item():.4f} (Ngưỡng yêu cầu: {args.conf_thresh})")
    else:
        print("🧐 [SOI ĐIỂM SỐ]: AI xuất ra 0 dự đoán (Tức là mù hoàn toàn).")

    # BƯỚC BẢO VỆ 1
    if len(all_boxes) == 0:
        print("🤷‍♂️ AI không tìm thấy đối tượng nào trên ảnh.")
        return

    # Lọc thủ công
    keep_idx = all_scores >= args.conf_thresh
    boxes = all_boxes[keep_idx]
    scores = all_scores[keep_idx]
    class_ids = all_class_ids[keep_idx]

    # BƯỚC BẢO VỆ 2
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
    result_img = visualizer.draw(
        image=img_draw, 
        boxes=boxes, 
        masks=masks, 
        classes=class_ids, 
        scores=scores
    )

    cv2.imwrite(args.out, result_img)
    print(f"📸 Đã lưu bức ảnh phân vùng thành công tại: {args.out}")

if __name__ == '__main__':
    main()