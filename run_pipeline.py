import os
import sys
import argparse
import shutil

# Import modules from src package
from src.parser import parse_scene_graph
from src.layout import compute_layout
from src.generator_2d import generate_2d_image
from src.segmenter import run_grounded_sam2
from src.generator_3d import generate_3d_models
from src.combiner import combine_scene_meshes

KAGGLE_WORKING = "/kaggle/working"
CROPS_DIR = os.path.join(KAGGLE_WORKING, "crops")
MULTI_GLB_DIR = os.path.join(KAGGLE_WORKING, "multi_object_glb")
OUT_DIR = os.path.join(KAGGLE_WORKING, "outputs/trellis")

def parse_args():
    parser = argparse.ArgumentParser(description="DATN End-to-End 3D Scene Reconstruction Pipeline")
    parser.add_argument(
        "--prompt", 
        type=str, 
        required=True,
        help="Câu mô tả văn bản tiếng Anh của cảnh (VD: 'a wooden chair next to a table')"
    )
    parser.add_argument(
        "--lora_scale", 
        type=float, 
        default=0.6,
        help="Hệ số scale LoRA weight cho SDXL (0.0 đến 1.0)"
    )
    parser.add_argument(
        "--scale_factor", 
        type=float, 
        default=0.01,
        help="Hệ số chuyển đổi đơn vị pixel sang mét trong không gian 3D"
    )
    return parser.parse_args()

def setup_directories():
    print("🧹 Đang khởi tạo và làm sạch thư mục đầu ra...")
    for d in [CROPS_DIR, MULTI_GLB_DIR, OUT_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

def main():
    args = parse_args()
    setup_directories()
    
    try:
        # Bước 1: spaCy NLP Parsing
        print("\n--- [BƯỚC 1] Phân tích ngữ nghĩa văn bản (spaCy NLP) ---")
        scene_graph = parse_scene_graph(args.prompt)
        print(f"Nodes: {len(scene_graph['nodes'])} | Edges: {len(scene_graph['edges'])}")
        
        # Bước 2: 2D Layout Box Calculating
        print("\n--- [BƯỚC 2] Tính toán bố cục hình học 2D (Layout Engine) ---")
        layout = compute_layout(scene_graph)
        
        # Bước 3: SDXL + LoRA Image Generation
        print("\n--- [BƯỚC 3] Sinh ảnh 2D (SDXL base + LoRA) ---")
        output_image_path = os.path.join(KAGGLE_WORKING, "input.png")
        generate_2d_image(args.prompt, args.lora_scale, output_image_path)
        shutil.copy(output_image_path, os.path.join(OUT_DIR, "input_2d.png"))
        print(f"Ảnh 2D lưu tại: {output_image_path}")
        
        # Bước 4: Grounded-SAM2 Instance Segmentation
        print("\n--- [BƯỚC 4] Nhận diện & Tách nền đa vật thể (Grounded-SAM2) ---")
        crops = run_grounded_sam2(output_image_path, layout, CROPS_DIR)
        print(f"Đã tách nền {len(crops)} vật thể.")
        
        # Bước 5: TRELLIS Single-Object 3D Generation
        print("\n--- [BƯỚC 5] Tái dựng mô hình 3D thô từng đối tượng (TRELLIS) ---")
        models = generate_3d_models(crops, MULTI_GLB_DIR)
        print(f"Đã dựng xong {len(models)} GLB models.")
        
        # Bước 6: Trimesh Scene Combining
        print("\n--- [BƯỚC 6] Sắp xếp và Ghép các đối tượng thành Cảnh 3D (Trimesh) ---")
        output_scene_path = os.path.join(OUT_DIR, "scene_combined.glb")
        combine_scene_meshes(models, output_scene_path, args.scale_factor)
        print(f"🎉 THÀNH CÔNG! File 3D Scene tổng hợp lưu tại: {output_scene_path}")
        
    except Exception as e:
        print(f"\n❌ Pipeline gặp lỗi nghiêm trọng: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
