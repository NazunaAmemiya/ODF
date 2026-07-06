# 🦟 Mosquito-CV Framework

**Mosquito-CV** là một framework Computer Vision mạnh mẽ, linh hoạt và được module hóa hoàn toàn, chuyên phục vụ cho bài toán **Phát hiện (Object Detection)** và **Phân vùng (Instance Segmentation)** đối tượng muỗi.

Được xây dựng dựa trên thiết kế *Registry* và *Builder* (tương tự OpenMMLab), framework cho phép các nhà nghiên cứu dễ dàng thay thế, tinh chỉnh và benchmark nhiều thuật toán khác nhau (YOLO, Mask R-CNN,...) trên cùng một hạ tầng thống nhất.

---

## ✨ Tính năng nổi bật

* **Kiến trúc Module Hóa Cao Độ:** Dễ dàng tháo lắp Backbone, Neck, Head, và Loss như những khối Lego thông qua cơ chế `@Registry`.
* **Config-Driven:** Toàn bộ quá trình huấn luyện, đánh giá và suy luận được điều khiển thông qua các file cấu hình `.yaml` (không cần sửa trực tiếp vào mã nguồn).
* **Đa Bài Toán:** Hỗ trợ song song cả Detection (Bounding Box) và Segmentation (Polygon/Mask).
* **Unified Pipeline:** Một 파이프라인 (pipeline) duy nhất cho mọi kiến trúc: `Train` $\rightarrow$ `Validate` $\rightarrow$ `Predict` $\rightarrow$ `Benchmark`.
* **Dễ Dàng Mở Rộng:** Thêm dataset mới, data augmentation mới hoặc hàm loss mới chỉ với vài dòng code khai báo.

---

## 📂 Tổng quan Cấu trúc Thư mục

Framework được chia thành 3 phần chính: **Cấu hình (Configs)**, **Công cụ (Tools)**, và **Mã nguồn lõi (Src)**.

```text
mosquito_framework/
├── configs/            # Chứa file cấu hình (.yaml) cho các thí nghiệm khác nhau
├── tools/              # Các script thực thi (train, val, predict, benchmark)
├── src/                # Mã nguồn cốt lõi (Core Framework)
│   ├── datasets/       # Dataloader, Data Augmentation và chuẩn hóa đầu vào
│   ├── models/         # Khởi tạo mô hình (Backbone, Neck, Head, Loss, Decoder)
│   ├── evaluation/     # Các metric đánh giá (mAP, mIoU, Dice...)
│   └── visualization/  # Trực quan hóa kết quả (Vẽ BBox, Mask)
├── data/               # (Bỏ qua trên Git) Thư mục chứa dataset thực tế
└── work_dirs/          # (Tự động sinh) Nơi lưu Checkpoint và Logs sau khi train