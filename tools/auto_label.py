import os
import cv2
import json
import argparse
import numpy as np
from datetime import datetime

# Import framework (Cần khớp với module thực tế trong source code của bạn)
import torch
from src.utils.checkpoint import load_checkpoint
from src.models.builder import build_model
from src.utils.logger import setup_logger

def parse_args():
    parser = argparse.ArgumentParser(description='Auto-labeling cho 1 Folder (Xuất COCO JSON)')
    parser.add_argument('--config', help='File config (vd: configs/yolo/yolov8n_seg_mosquito.yaml)', required=True)
    parser.add_argument('--checkpoint', help='File trọng số (.pth) đã train sơ bộ', required=True)
    parser.add_argument('--folder', help='Thư mục chứa ảnh cần gán nhãn', required=True)
    parser.add_argument('--conf-thres', type=float, default=0.25, help='Ngưỡng tin cậy (Confidence)')
    return parser.parse_args()

class COCOFolderLabeler:
    def __init__(self, config_path, checkpoint_path, conf_thres=0.25):
        self.logger = setup_logger("AutoLabeler")
        self.conf_thres = conf_thres
        
        self.logger.info(f"Đang tải mô hình từ {config_path}...")
        # LƯU Ý: Khởi tạo mô hình tại đây (Dựa theo predict.py của framework)
        # self.model = build_model(config_path) 
        # load_checkpoint(self.model, checkpoint_path)
        # self.model.eval()
        # self.model.cuda()
        
        self.coco_format = {
            "info": {"description": "Auto-labeled Folder", "date_created": datetime.now().strftime("%Y/%m/%d")},
            "images": [],
            "annotations": [],
            "categories": [{"id": 1, "name": "mosquito", "supercategory": "insect"}]
        }
        self.annotation_id = 1
        self.image_id = 1

    def simplify_polygon(self, polygon, tolerance=2.0):
        polygon = np.array(polygon, dtype=np.float32).reshape(-1, 1, 2)
        simplified = cv2.approxPolyDP(polygon, tolerance, closed=True)
        return simplified.flatten().tolist()

    def process_single_folder(self, folder_path):
        self.logger.info(f"Bắt đầu quét thư mục: {folder_path}")
        
        # Đường dẫn file JSON đầu ra sẽ tự động đặt trong folder này
        out_json_path = os.path.join(folder_path, 'annotations.json')
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif')
        
        # Quét toàn bộ file trong 1 folder
        for file_name in os.listdir(folder_path):
            if not file_name.lower().endswith(valid_extensions):
                continue
                
            img_path = os.path.join(folder_path, file_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            height, width = img.shape[:2]
            
            # Ghi nhận ảnh
            self.coco_format["images"].append({
                "id": self.image_id,
                "file_name": file_name,
                "width": width,
                "height": height
            })
            
            # Chạy model dự đoán (Thay bằng logic gọi model thực tế)
            # with torch.no_grad():
            #     results = self.model.predict(img) 
            results = self.model.predict(img)
            
            # Xử lý kết quả trả về
            for res in results:
                score = res.get('score', 0)
                if score < self.conf_thres:
                    continue
                    
                bbox = res.get('bbox', []) 
                mask = res.get('mask') 
                polygon = []
                
                if mask is not None:
                    contours, _ = cv2.findContours((mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if len(contours) > 0:
                        largest_contour = max(contours, key=cv2.contourArea)
                        polygon = self.simplify_polygon(largest_contour)
                
                area = bbox[2] * bbox[3] if len(bbox) == 4 else 0
                
                self.coco_format["annotations"].append({
                    "id": self.annotation_id,
                    "image_id": self.image_id,
                    "category_id": 1,
                    "bbox": bbox,
                    "segmentation": [polygon] if polygon else [],
                    "area": float(area),
                    "iscrowd": 0,
                    "score": float(score)
                })
                self.annotation_id += 1
                
            self.image_id += 1
            
        # Ghi đè file JSON vào folder
        with open(out_json_path, 'w', encoding='utf-8') as f:
            json.dump(self.coco_format, f, ensure_ascii=False, indent=4)
            
        self.logger.info(f"Hoàn thành! Đã gán nhãn {self.image_id - 1} ảnh.")
        self.logger.info(f"File nhãn COCO được lưu tại: {out_json_path}")

def main():
    args = parse_args()
    # Kiểm tra xem folder có tồn tại không
    if not os.path.isdir(args.folder):
        print(f"Lỗi: Thư mục '{args.folder}' không tồn tại!")
        return
        
    labeler = COCOFolderLabeler(args.config, args.checkpoint, args.conf_thres)
    labeler.process_single_folder(args.folder)

if __name__ == '__main__':
    main()