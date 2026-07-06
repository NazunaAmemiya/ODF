import argparse
import os
import sys
import yaml
import torch
import time

# BẮT BUỘC: Đẩy thư mục gốc vào sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)

from src.utils.logger import setup_logger
from src.models.builder import build_model

# Thử import thư viện tính toán độ phức tạp (FLOPs)
try:
    from thop import profile, clever_format
    HAS_THOP = True
except ImportError:
    HAS_THOP = False
    print("⚠️ Cảnh báo: Thư viện 'thop' chưa được cài đặt. Bỏ qua bước tính FLOPs.")
    print("👉 Hãy chạy lệnh: pip install thop")

def parse_args():
    parser = argparse.ArgumentParser(description='🦟 Benchmark hiệu năng mô hình (FPS, Latency, FLOPs)')
    parser.add_argument('--config', type=str, required=True, 
                        help='Đường dẫn tới file cấu hình YAML')
    parser.add_argument('--checkpoint', type=str, default=None, 
                        help='Đường dẫn tới trọng số (Không bắt buộc, nhưng nên có)')
    parser.add_argument('--batch-size', type=int, default=1, 
                        help='Kích thước batch để đo đạc (Thường set = 1 để đo độ trễ thực tế)')
    parser.add_argument('--img-size', type=int, nargs='+', default=[640, 640], 
                        help='Kích thước ảnh đầu vào mô phỏng, ví dụ: 640 640')
    parser.add_argument('--half', action='store_true', 
                        help='Kích hoạt chế độ FP16 (Half precision) để đo tốc độ TensorRT/Edge')
    parser.add_argument('--warmup', type=int, default=50, 
                        help='Số vòng lặp khởi động GPU trước khi đo')
    parser.add_argument('--iters', type=int, default=200, 
                        help='Số vòng lặp chính thức để lấy trung bình')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    args = parse_args()
    cfg = load_config(args.config)
    
    logger = setup_logger(name="mosquito_cv_bench", save_dir='./work_dirs')
    logger.info("🚀 KHỞI ĐỘNG CÔNG CỤ BENCHMARK PHẦN CỨNG 🚀")
    
    # 1. Thiết bị chạy
    device = torch.device(cfg.get('device', 'cuda:0' if torch.cuda.is_available() else 'cpu'))
    is_gpu = device.type != 'cpu'
    
    # 2. Lắp ráp Mô hình
    model = build_model(cfg['model'])
    
    if args.checkpoint:
        logger.info(f"Đang nạp trọng số từ: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=True)
    
    model.to(device)
    model.eval() # Bắt buộc khi đo đạc!
    
    # Ép kiểu FP16 nếu yêu cầu (Tăng tốc độ trên các GPU đời mới)
    if args.half and is_gpu:
        model.half()
        logger.info("⚡ Đã kích hoạt chế độ FP16 (Half-Precision).")

    # Xử lý input size
    if len(args.img_size) == 1:
        input_size = (args.img_size[0], args.img_size[0])
    else:
        input_size = tuple(args.img_size[:2])

    logger.info("-" * 50)
    logger.info(f"Cấu hình Test: Batch Size = {args.batch_size}, Image Size = {input_size}")
    logger.info(f"Thiết bị: {torch.cuda.get_device_name(device) if is_gpu else 'CPU'}")
    
    # 3. ĐO SỐ LƯỢNG THAM SỐ (PARAMETERS) VÀ FLOPS
    # Dùng Tensor rác (Dummy input) để mồi mô hình
    dummy_input = torch.randn(args.batch_size, 3, input_size[0], input_size[1]).to(device)
    if args.half and is_gpu:
        dummy_input = dummy_input.half()

    if HAS_THOP:
        logger.info("Đang phân tích tính toán (FLOPs)...")
        # thop đo lường trên một forward pass
        flops, params = profile(model, inputs=(dummy_input, ), verbose=False)
        flops_str, params_str = clever_format([flops, params], "%.2f")
        logger.info(f"🧠 Tổng số tham số (Parameters): {params_str}")
        logger.info(f"⚙️ Độ phức tạp toán học (FLOPs) : {flops_str}")
    else:
        # Tính thủ công Parameters nếu không có thop
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"🧠 Tổng số tham số (Parameters): {total_params / 1e6:.2f} M")

    # 4. ĐO TỐC ĐỘ (LATENCY & FPS)
    logger.info("-" * 50)
    logger.info(f"Đang làm nóng (Warm-up) thiết bị trong {args.warmup} iterations...")
    with torch.no_grad():
        for _ in range(args.warmup):
            _ = model(dummy_input)

    logger.info(f"Bắt đầu đo tốc độ trong {args.iters} iterations...")
    time_list = []
    
    with torch.no_grad():
        for _ in range(args.iters):
            # Đồng bộ hoá bộ nhớ đệm GPU trước khi bấm giờ (CỰC KỲ QUAN TRỌNG ĐỂ ĐO CHÍNH XÁC)
            if is_gpu:
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            
            _ = model(dummy_input)
            
            if is_gpu:
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            time_list.append(end_time - start_time)

    # 5. TỔNG HỢP BÁO CÁO
    total_time = sum(time_list)
    avg_latency = (total_time / args.iters) * 1000 # Đổi ra milliseconds (ms)
    # Công thức FPS: 1000 / (Độ trễ của 1 ảnh). Vì 1 batch có thể có nhiều ảnh nên nhân với batch_size
    avg_fps = (1000 / avg_latency) * args.batch_size

    logger.info("-" * 50)
    logger.info("📊 BÁO CÁO HIỆU NĂNG PHẦN CỨNG (BENCHMARK REPORT)")
    logger.info("-" * 50)
    logger.info(f"⏱️ Độ trễ trung bình (Latency): {avg_latency:.2f} ms / batch")
    logger.info(f"🎞️ Tốc độ khung hình (FPS)   : {avg_fps:.2f} frames/sec")
    logger.info("-" * 50)
    logger.info("Hoàn tất Benchmark!")

if __name__ == '__main__':
    main()