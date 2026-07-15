import os
import sys
from huggingface_hub import snapshot_download

# Định nghĩa thư mục lưu trữ đích
TARGET_DIR = r"D:\DATN\down"
os.makedirs(TARGET_DIR, exist_ok=True)

print("🚀 Bắt đầu tiến trình tải weights mô hình về máy cá nhân...")
print(f"📂 Thư mục đích: {TARGET_DIR}")

try:
    print("\n⏳ 1. Đang tải mô hình BERT (khoảng 400MB)...")
    snapshot_download(
        repo_id="bert-base-uncased", 
        local_dir=os.path.join(TARGET_DIR, "bert-base-uncased"),
        ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.bin", "*.bin.meta"] # Chỉ giữ lại các định dạng thích hợp
    )
    # Tải thêm pytorch_model.bin gốc của BERT
    snapshot_download(
        repo_id="bert-base-uncased",
        local_dir=os.path.join(TARGET_DIR, "bert-base-uncased"),
        allow_patterns=["pytorch_model.bin", "config.json", "vocab.txt", "tokenizer.json"]
    )
    print("✅ Đã tải xong BERT!")

    print("\n⏳ 2. Đang tải mô hình TRELLIS-image-large (khoảng 6.2GB)...")
    snapshot_download(
        repo_id="JeffreyXiang/TRELLIS-image-large", 
        local_dir=os.path.join(TARGET_DIR, "TRELLIS-image-large")
    )
    print("✅ Đã tải xong TRELLIS-image-large!")

    print(f"\n🎉 HOÀN THÀNH THÀNH CÔNG! Toàn bộ file đã được lưu tại: {TARGET_DIR}")
    print("👉 Hãy nén thư mục 'down' thành file 'down.zip' rồi upload lên Kaggle Dataset nhé!")

except Exception as e:
    print(f"\n❌ Có lỗi xảy ra trong quá trình tải: {e}")
    sys.exit(1)
