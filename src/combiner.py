import os
import json
import subprocess

def combine_scene_meshes(models: list, output_scene_path: str, scale_factor: float = 0.01) -> bool:
    """
    Sử dụng trimesh để biến đổi hình học và ghép các vật thể 3D đơn lẻ thành một Cảnh 3D.
    """
    models_json_path = "/kaggle/working/models_to_combine.json"
    with open(models_json_path, 'w') as f:
        json.dump(models, f)

    script = f"""
import sys
sys.path.insert(0, "/opt/venv310/lib/python3.10/site-packages")

import json, trimesh, os
import numpy as np

with open("{models_json_path}") as f:
    models = json.load(f)

scene = trimesh.Scene()
scale_factor = {scale_factor}
img_w, img_h = 1024, 1024

for m in models:
    glb_path = m["model_path"]
    if not os.path.exists(glb_path):
        continue
    
    mesh = trimesh.load(glb_path)
    x1, y1, x2, y2 = m["final_box"]
    box_w = x2 - x1
    box_h = y2 - y1
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    mesh_extent = mesh.bounding_box.extents
    target_size = max(box_w, box_h) * scale_factor
    current_size = max(mesh_extent)
    
    if current_size > 0:
        scale = target_size / current_size
        mesh.apply_scale(scale)
        
    tx = (center_x - img_w / 2) * scale_factor
    tz = (center_y - img_h / 2) * scale_factor
    mesh.apply_translation([tx, 0, -tz])
    
    scene.add_geometry(mesh, node_name=m["name"])

scene.export("{output_scene_path}")
print("✅ Ghép cảnh xong.")
"""
    PY_PATH = "/opt/venv310/bin/python"
    r = subprocess.run([PY_PATH, "-c", script], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"Lỗi ghép cảnh trimesh: {r.stderr}")
    return os.path.exists(output_scene_path)
