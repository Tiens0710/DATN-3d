import os
import json
import subprocess

PY_PATH = "/opt/venv310/bin/python"

def run_grounded_sam2(input_image_path: str, layout: dict, crops_dir: str) -> list:
    """
    Chạy GroundingDINO nhận diện và SAM2 bóc tách nền vật thể.
    """
    labels_list = []
    for nid in layout.get("layout", {}).keys():
        labels_list.append("chair" if "chair" in nid else "table")
    if not labels_list:
        labels_list = ["chair", "table"]

    script = f"""
import sys, json, os, torch, numpy as np
sys.modules['triton'] = None

sys.path.insert(0, "/opt/venv310/lib/python3.10/site-packages")
from groundingdino.util.inference import load_model, load_image, predict
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from PIL import Image

CKPT_PATH = "/kaggle/working/groundingdino_ckpt/groundingdino_swint_ogc.pth"
CONFIG_PATH = "/kaggle/working/groundingdino_ckpt/GroundingDINO_SwinT_OGC.py"
model_dino = load_model(CONFIG_PATH, CKPT_PATH)

image_source, image_tensor = load_image("{input_image_path}")
labels = {labels_list}
text_prompt = " . ".join(labels)

boxes, logits, phrases = predict(
    model=model_dino,
    image=image_tensor,
    caption=text_prompt,
    box_threshold=0.35,
    text_threshold=0.25
)

SAM2_CKPT = "/kaggle/working/sam2_ckpt/sam2_hiera_small.pt"
sam2_model = build_sam2("sam2_hiera_s.yaml", SAM2_CKPT, device="cuda")
predictor = SAM2ImagePredictor(sam2_model)

img_rgb = np.array(Image.open("{input_image_path}").convert("RGB"))
predictor.set_image(img_rgb)
H, W, _ = img_rgb.shape

results = []
for i, (box, logit, phrase) in enumerate(zip(boxes, logits, phrases)):
    cx, cy, bw, bh = box.tolist()
    x1 = int((cx - bw/2) * W)
    y1 = int((cy - bh/2) * H)
    x2 = int((cx + bw/2) * W)
    y2 = int((cy + bh/2) * H)
    
    input_box = np.array([[x1, y1, x2, y2]])
    masks, scores, _ = predictor.predict(
        point_coords=None, point_labels=None, box=input_box, multimask_output=False
    )
    
    mask = masks[0]
    img_rgba = Image.fromarray(img_rgb).convert("RGBA")
    alpha = Image.fromarray((mask * 255).astype(np.uint8))
    img_rgba.putalpha(alpha)
    
    PAD = 15
    cx1 = max(0, x1 - PAD)
    cy1 = max(0, y1 - PAD)
    cx2 = min(W, x2 + PAD)
    cy2 = min(H, y2 + PAD)
    crop = img_rgba.crop((cx1, cy1, cx2, cy2))
    
    name = f"object_{{i+1}}"
    crop_path = f"{crops_dir}/{{name}}.png"
    crop.save(crop_path)
    
    results.append({{
        "name": name,
        "label": phrase,
        "box": [x1, y1, x2, y2],
        "final_box": [cx1, cy1, cx2, cy2],
        "crop_url": f"/crops/{{name}}.png",
        "crop_path": crop_path
    }})

with open("{crops_dir}/sam2_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("✅ Tách nền SAM2 hoàn tất.")
"""
    r = subprocess.run([PY_PATH, "-c", script], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"Lỗi chạy Grounded-SAM2: {r.stderr}")
        
    result_json_path = os.path.join(crops_dir, "sam2_results.json")
    with open(result_json_path, 'r') as f:
        crops_data = json.load(f)
    return crops_data
