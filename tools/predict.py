import argparse
import cv2
import yaml
import torch
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Import các module chuẩn xác từ framework ODF
from src.models.builder import build_model
from src.models.decoders.seg_decoder import SegDecoder
from src.visualization.seg_visualizer import SegVisualizer

def parse_args():
    parser = argparse.ArgumentParser(description='🦟 Dự đoán và Phân vùng muỗi (Segmentation Inference)')
    parser.add_argument('--config', type=str, required=True, 
                        help='Đường dẫn file YAML')
    parser.add_argument('--checkpoint', type=str, required=True, 
                        help='Đường dẫn tới file trọng số best_checkpoint.pth')
    parser.add_argument('--source', type=str, required=True, 
                        help='Đường dẫn tới bức ảnh cần test')
    parser.add_argument('--out', type=str, default='result.jpg', 
                        help='Tên file ảnh xuất ra sau khi vẽ')
    parser.add_argument('--conf-thresh', type=float, default=0.5, 
                        help='Ngưỡng tin cậy tối thiểu (mặc định: 0.5)')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def letterbox_image(image, expected_size):
    """Giữ nguyên tỷ lệ ảnh, bù viền xám (114,114,114) chuẩn YOLOv8"""
    ih, iw = image.shape[0:2]
    ew, eh = expected_size
    scale = min(ew / iw, eh / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    
    image_resized = cv2.resize(image, (nw, nh))
    image_padded = np.full((eh, ew, 3), 114, dtype=np.uint8)
    
    dx = (ew - nw) // 2
    dy = (eh - nh) // 2
    image_padded[dy:dy+nh, dx:dx+nw, :] = image_resized
    
    return image_padded

def preprocess_image(image_path, input_size, device):
    """Tiền xử lý: Letterbox -> RGB -> Tensor -> Normalize"""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"❌ Không thể đọc ảnh: {image_path}")
    
    # 1. Letterbox
    img_padded = letterbox_image(img_bgr, (input_size[0], input_size[1]))
    
    # 2. BGR to RGB
    img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
    
    # 3. To Tensor [0, 1]
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    
    # 4. Normalize (ImageNet standard)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_tensor = (img_tensor - mean) / std 
    
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

    # ==========================================
    # 2. CHUẨN BỊ DỮ LIỆU
    # ==========================================
    class_names = cfg.get('dataset', {}).get('class_names', ['aedes', 'culex', 'anopheles'])
    input_size = cfg.get('pipeline', {}).get('input_size', [640, 640])
    
    if isinstance(input_size, list):
        input_size = tuple(input_size)

    img_tensor, img_draw = preprocess_image(args.source, input_size, device)

    # ==========================================
    # 3. TIẾN HÀNH SUY LUẬN & GIẢI MÃ
    # ==========================================
    print("🧠 Trí tuệ nhân tạo đang phân tích ảnh...")
    with torch.no_grad():
        raw_outputs = model(img_tensor)
        
    # KHỞI TẠO DECODER LỌC NHIỄU VÀ ÁP DỤNG NGƯỠNG TIN CẬY
    decoder = SegDecoder(conf_thresh=args.conf_thresh)
    decoded_outputs = decoder(raw_outputs)
    
    predictions = decoded_outputs[0]
    
    boxes = predictions.get('boxes', [])
    scores = predictions.get('scores', [])
    class_ids = predictions.get('labels', []) # Đã sửa thành 'labels' cho chuẩn
    masks = predictions.get('masks', [])

    print("--- DEBUG: AI DỰ ĐOÁN ---")
    print(f"Danh sách các điểm tin cậy (scores): {scores}")
    print(f"Số lượng boxes tìm thấy: {len(boxes)}")

    if len(boxes) == 0:
        print("🤷‍♂️ Không tìm thấy đối tượng nào thỏa mãn độ tin cậy.")
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