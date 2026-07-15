import os
import sys
import json
import unittest
import cv2
import numpy as np
from unittest.mock import patch, MagicMock

# Thêm thư mục gốc vào biến môi trường sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tools.auto_label import COCOFolderLabeler

class TestAutoLabel(unittest.TestCase):
    def setUp(self):
        # 1. Trỏ trực tiếp vào thư mục tests/test_data
        self.test_dir = os.path.join(os.path.dirname(__file__), 'test_data/Macro invertebrates data v1/Amphipod')
        os.makedirs(self.test_dir, exist_ok=True)
        
        # 2. Đảm bảo có ít nhất 1 ảnh để test không bị lỗi (nếu bạn chưa bỏ ảnh nào vào)
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif')
        has_image = any(f.lower().endswith(valid_exts) for f in os.listdir(self.test_dir))
        if not has_image:
            dummy_img_path = os.path.join(self.test_dir, 'dummy_mosquito.jpg')
            cv2.imwrite(dummy_img_path, np.zeros((100, 100, 3), dtype=np.uint8))
        
        # 3. Tạo file config và checkpoint giả
        self.dummy_config = os.path.join(self.test_dir, 'dummy.yaml')
        self.dummy_ckpt = os.path.join(self.test_dir, 'dummy.pth')
        open(self.dummy_config, 'a').close()
        open(self.dummy_ckpt, 'a').close()

    @patch('tools.auto_label.load_checkpoint')
    @patch('tools.auto_label.build_model')
    def test_process_single_folder(self, mock_build_model, mock_load_checkpoint):
        # Tạo model giả trả về kết quả mặc định
        mock_fake_model = MagicMock()
        mock_fake_model.predict.return_value = [{
            'score': 0.95,
            'bbox': [10, 10, 50, 50],
            'mask': np.ones((100, 100))
        }]
        mock_build_model.return_value = mock_fake_model
        
        # Khởi tạo class Labeler
        labeler = COCOFolderLabeler(
            config_path=self.dummy_config, 
            checkpoint_path=self.dummy_ckpt,
            conf_thres=0.25
        )
        labeler.model = mock_fake_model
        
        # Chạy logic xử lý folder (nó sẽ tự quét mọi ảnh trong test_data)
        labeler.process_single_folder(self.test_dir)
        
        # Kiểm tra file JSON
        json_path = os.path.join(self.test_dir, 'annotations.json')
        self.assertTrue(os.path.exists(json_path), "Lỗi: File annotations.json không được tạo ra!")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Dùng assertGreaterEqual vì nếu bạn để 5 ảnh trong folder, nó sẽ sinh ra 5 annotations
        self.assertGreaterEqual(len(data['annotations']), 1, "Chưa ghi nhận được annotation!")
        
        # Kiểm tra nội dung bbox do model giả tạo ra
        self.assertEqual(data['annotations'][0]['bbox'], [10, 10, 50, 50], "Bbox ghi vào JSON bị sai lệch!")

    def tearDown(self):
        # DỌN DẸP AN TOÀN: Chỉ xóa các file rác sinh ra, KHÔNG XÓA thư mục chứa ảnh
        json_path = os.path.join(self.test_dir, 'annotations.json')
        if os.path.exists(json_path):
            os.remove(json_path)
            
        if os.path.exists(self.dummy_config):
            os.remove(self.dummy_config)
            
        if os.path.exists(self.dummy_ckpt):
            os.remove(self.dummy_ckpt)

if __name__ == '__main__':
    unittest.main()