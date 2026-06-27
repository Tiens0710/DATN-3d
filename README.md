# DATN: Automated Text-to-3D Room Scene Reconstruction

Đồ án tốt nghiệp xây dựng hệ thống tự động tái dựng cảnh phòng ngủ/phòng khách 3D từ mô tả văn bản tiếng Anh bằng cách kết hợp mô hình Sinh ảnh 2D, Phân mảnh vật thể và Dựng hình 3D thưa.

## 🌟 Công nghệ tích hợp
*   **NLP Scene Graph**: Phân tích ngữ nghĩa câu bằng `spaCy` để trích xuất vật thể và quan hệ không gian.
*   **2D Generation**: Sử dụng `SDXL base` tinh chỉnh bằng trọng số `LoRA` (chân thực hóa nội thất) kết hợp `ControlNet Segmentation`.
*   **Phân mảnh tự động**: Tích hợp `GroundingDINO` (định vị) và `SAM2` (bóc tách nền pixel trong suốt).
*   **Dựng 3D đơn vật thể**: Sinh mô hình 3D Mesh dạng `.glb` chất lượng cao từ ảnh 2D bằng `TRELLIS Image-to-3D`.
*   **Ghép cảnh 3D**: Sử dụng thuật toán ánh xạ hình học 2D-to-3D thông qua thư viện `trimesh`.

---

## 📁 Cấu trúc thư mục mã nguồn

```text
├── index.html          # Giao diện Web Node-Graph kéo thả (giống ComfyUI)
├── server.py           # FastAPI Web Server (chạy trên GPU Kaggle)
├── run_pipeline.py     # Script chạy tự động CLI tuần tự (chạy trên Kaggle)
├── requirements.txt    # Danh sách các thư viện Python cần thiết
└── README.md           # Hướng dẫn sử dụng
```

---

## ⚙️ Hướng dẫn cài đặt môi trường (Kaggle)

Tạo virtual environment Python 3.10 và cài các thư viện trong `requirements.txt`:

```bash
# 1. Tạo môi trường ảo
python3.10 -m venv /opt/venv310

# 2. Kích hoạt và cài đặt dependencies
/opt/venv310/bin/pip install -r requirements.txt
```

---

## 🚀 Cách chạy chương trình

Bạn có thể chạy thử nghiệm mã nguồn này theo 2 cách:

### Cách 1: Chạy qua dòng lệnh (CLI Mode) - Tuần tự không giao diện
Chạy trực tiếp script `run_pipeline.py` bằng Python 3.10 trên Kaggle:

```bash
/opt/venv310/bin/python run_pipeline.py \
    --prompt "a wooden dining chair next to a small wooden table" \
    --lora_scale 0.6 \
    --scale_factor 0.01
```

*   **Kết quả đầu ra**: 
    *   Ảnh 2D sinh ra lưu tại `/kaggle/working/outputs/trellis/input_2d.png`
    *   Các ảnh vật thể cắt nền lưu tại `/kaggle/working/crops/`
    *   Các file GLB đơn lẻ lưu tại `/kaggle/working/multi_object_glb/`
    *   File 3D cảnh phòng hoàn chỉnh lưu tại `/kaggle/working/outputs/trellis/scene_combined.glb`

---

### Cách 2: Chạy qua giao diện Web tương tác (Web UI Mode)
1.  **Chạy Backend trên Kaggle**: Khởi động FastAPI server và tạo tunnel Ngrok để lấy Link API.
    ```bash
    /opt/venv310/bin/python server.py
    ```
2.  **Mở Frontend trên Máy tính**: Click đúp mở file `index.html` bằng trình duyệt Web, điền Link API Ngrok vào góc phải màn hình và nhấn **Connect**.
3.  **Thực thi**: Nhập mô tả vào Node đầu vào và bấm **▶ Queue Prompt** để trải nghiệm chạy trực quan.
