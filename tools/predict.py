import argparse
import cv2
import yaml
import torch
import numpy as np

# Import các module chuẩn xác từ framework ODF của bạn
from src.models.builder import build_model
from src.models.decoders.seg_decoder import SegDecoder
from src.visualization.seg_visualizer import SegVisualizer

def parse_args():
    parser = argparse.ArgumentParser(description='🦟 Dự đoán và Phân vùng muỗi (Segmentation Inference)')
    parser.add_argument('--config', type=str, required=True, 
                        help='Đường dẫn file YAML (vd: configs/yolo/yolov8n_seg_mosquito.yaml)')
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
    """Đọc file cấu hình YAML"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def preprocess_image(image_path, input_size=(640, 640), device='cpu'):
    """Đọc, resize và chuẩn hóa ảnh thành Tensor"""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"❌ Không thể đọc ảnh từ {image_path}. Vui lòng kiểm tra lại đường dẫn!")
    
    # Giữ lại ảnh đã resize để xíu nữa đưa vào Visualizer vẽ cho khớp tọa độ
    img_resized = cv2.resize(img_bgr, input_size)
    
    # Chuyển BGR (OpenCV) sang RGB và Normalize (/255.0)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    
    # Thêm batch dimension -> (1, C, H, W)
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    return img_tensor, img_resized

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
    
    # Xử lý trường hợp lưu toàn bộ Checkpoint (có optimizer) hay chỉ lưu State Dict
    state_dict = checkpoint.get('state_dict', checkpoint)
    model.load_state_dict(state_dict, strict=True)
    
    model.to(device)
    model.eval() # Bật chế độ suy luận (Tắt Dropout/BatchNorm updates)

    # ==========================================
    # 2. CHUẨN BỊ DỮ LIỆU
    # ==========================================
    # Lấy thông số từ config, nếu không có thì mặc định
    class_names = cfg.get('dataset', {}).get('class_names', ['aedes', 'culex', 'anopheles'])
    input_size = cfg.get('pipeline', {}).get('input_size', [640, 640])
    
    # Đảm bảo input_size là tuple (W, H)
    if isinstance(input_size, list):
        input_size = tuple(input_size)

    img_tensor, img_draw = preprocess_image(args.source, input_size, device)

    # ==========================================
    # 3. TIẾN HÀNH SUY LUẬN (INFERENCE)
    # ==========================================
    print("🧠 Trí tuệ nhân tạo đang phân tích ảnh...")
    with torch.no_grad():
        raw_outputs = model(img_tensor)
        
    # ==========================================
    # 4. GIẢI MÃ KẾT QUẢ (DECODE)
    # ==========================================
    # Khởi tạo SegDecoder với mask_threshold (mặc định 0.5)
    decoder = SegDecoder(mask_threshold=0.5)
    
    # Chạy hàm decode đúng như cấu trúc class bạn đã viết
    detections = decoder.decode(
        outputs=raw_outputs, 
        image_size=input_size, 
        conf_thres=args.conf_thresh
    )
    
    # Dữ liệu trả về là 1 List, ta lấy phần tử đầu tiên (vì chỉ truyền vào 1 ảnh)
    predictions = detections[0]
    
    boxes = predictions.get('boxes', [])
    scores = predictions.get('scores', [])
    class_ids = predictions.get('classes', [])
    masks = predictions.get('masks', [])

    if len(boxes) == 0:
        print("🤷‍♂️ Không tìm thấy đối tượng nào thỏa mãn độ tin cậy.")
        return

    print(f"🎯 Phát hiện {len(boxes)} vật thể!")

    # ==========================================
    # 5. VẼ KHUNG & MẶT NẠ (VISUALIZE)
    # ==========================================
    visualizer = SegVisualizer(class_names=class_names)
    
    # Đưa ảnh đã resize (để khớp tỉ lệ mask/box) vào visualizer
    result_img = visualizer.draw(
        image=img_draw, 
        boxes=boxes, 
        masks=masks, 
        classes=class_ids, 
        scores=scores
    )

    # ==========================================
    # 6. XUẤT KẾT QUẢ
    # ==========================================
    cv2.imwrite(args.out, result_img)
    print(f"📸 Đã lưu bức ảnh phân vùng thành công tại: {args.out}")

if __name__ == '__main__':
    main()