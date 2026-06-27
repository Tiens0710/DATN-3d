import os
import json
import subprocess

PY_PATH = "/opt/venv310/bin/python"

def generate_3d_models(crops: list, multi_glb_dir: str) -> list:
    """
    Sử dụng TRELLIS để tạo các mô hình 3D (.glb) riêng lẻ từ ảnh tách nền.
    """
    meta_json_path = "/kaggle/working/objects_meta_api.json"
    objects_dict = {crop["name"]: crop for crop in crops}
    with open(meta_json_path, 'w') as f:
        json.dump(objects_dict, f)

    script = f"""
import sys, os, json, torch
sys.modules['triton'] = None

sys.path.insert(0, "/kaggle/working/TRELLIS")
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import postprocessing_utils
from PIL import Image

with open("{meta_json_path}") as f:
    objects = json.load(f)

pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
pipeline.to("cuda")

for name, info in objects.items():
    print(f"Dựng mô hình 3D cho {{name}}...")
    img = Image.open(info["crop_path"]).convert("RGB")
    image = pipeline.preprocess_image(img)
    
    outputs = pipeline.run(
        image,
        seed=42,
        formats=["gaussian", "mesh"],
        preprocess_image=False,
        sparse_structure_sampler_params={{"steps": 12, "cfg_strength": 7.5}},
        slat_sampler_params={{"steps": 12, "cfg_strength": 3.0}},
    )
    
    glb = postprocessing_utils.to_glb(
        outputs["gaussian"][0], outputs["mesh"][0],
        simplify=0.95, texture_size=1024, verbose=False
    )
    out_path = f"{multi_glb_dir}/{{name}}.glb"
    glb.export(out_path)
    print(f"Saved: {{out_path}}")
    torch.cuda.empty_cache()
"""
    r = subprocess.run([PY_PATH, "-c", script], capture_output=True, text=True, timeout=1200)
    if r.returncode != 0:
        raise RuntimeError(f"Lỗi chạy TRELLIS: {r.stderr}")
        
    models = []
    for crop in crops:
        name = crop["name"]
        models.append({
            "name": name,
            "label": crop["label"],
            "model_url": f"/multi_object_glb/{name}.glb",
            "model_path": f"{multi_glb_dir}/{name}.glb",
            "final_box": crop["final_box"]
        })
    return models
