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
        label = nid.split("_")[0] if "_" in nid else nid
        labels_list.append(label)
    if not labels_list:
        labels_list = ["chair", "table"]

    # Use .replace() instead of f-string to avoid "Invalid format specifier" errors
    # caused by Python dict/list literals inside the script body
    script = """
import sys
sys.path.insert(0, "/opt/venv310/lib/python3.10/site-packages")

import json, os, torch, numpy as np
sys.modules['triton'] = None
from groundingdino.util.inference import load_model, load_image, predict
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from PIL import Image

CKPT_PATH = "/kaggle/working/groundingdino_ckpt/groundingdino_swint_ogc.pth"
CONFIG_PATH = "/kaggle/working/groundingdino_ckpt/GroundingDINO_SwinT_OGC.py"
model_dino = load_model(CONFIG_PATH, CKPT_PATH)

image_source, image_tensor = load_image("__INPUT_IMAGE_PATH__")
labels = __LABELS_LIST__
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

img_rgb = np.array(Image.open("__INPUT_IMAGE_PATH__").convert("RGB"))
predictor.set_image(img_rgb)
H, W, _ = img_rgb.shape

# Tạo ảnh lưu trữ mặt nạ trực quan hóa (visual mask overlay)
visual_img = img_rgb.copy()
colors = [
    [251, 113, 133],  # #fb7185 (Rose)
    [52, 211, 153],   # #34d399 (Emerald)
    [56, 189, 248],   # #38bdf8 (Sky)
    [250, 204, 21],   # #facc15 (Amber)
    [244, 114, 182]   # #f472b6 (Pink)
]

results = []
for i, (box, logit, phrase) in enumerate(zip(boxes, logits, phrases)):
    cx, cy, bw, bh = box.tolist()
    x1 = int((cx - bw/2) * W)
    y1 = int((cy - bh/2) * H)
    x2 = int((cx + bw/2) * W)
    y2 = int((cy + bh/2) * H)
    
    input_box = np.array([[x1, y1, x2, y2]])
    
    # Gợi ý thêm điểm dương ở tâm để giữ phần ruột kính phản chiếu của gương
    cx_px = (x1 + x2) // 2
    cy_px = (y1 + y2) // 2
    point_coords = np.array([[cx_px, cy_px]])
    point_labels = np.array([1]) # 1 là foreground point
    
    masks, scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box=input_box,
        multimask_output=False
    )
    
    import scipy.ndimage as ndimage
    # 1. Phép đóng (Closing) để kết nối các phần bị đứt nét và lấp khe hở nhỏ ở viền
    closed_mask = ndimage.binary_closing(masks[0], structure=np.ones((7, 7)))
    # 2. Điền đầy lỗ hổng (Fill holes) để lấp kín phần kính phản chiếu
    filled_mask = ndimage.binary_fill_holes(closed_mask)
    
    # Vẽ đè mặt nạ lên visual_img (độ mờ 40%)
    color = colors[i % len(colors)]
    mask_pixels = (filled_mask > 0)
    visual_img[mask_pixels] = (visual_img[mask_pixels] * 0.6 + np.array(color) * 0.4).astype(np.uint8)
    
    # 3. Làm mịn cạnh (Anti-aliasing) bằng Gaussian Filter để tránh răng cưa
    alpha_array = (filled_mask * 255).astype(np.uint8)
    alpha_smooth = ndimage.gaussian_filter(alpha_array.astype(float), sigma=1.2)
    alpha_smooth = np.clip(alpha_smooth, 0, 255).astype(np.uint8)
    
    img_rgba = Image.fromarray(img_rgb).convert("RGBA")
    alpha = Image.fromarray(alpha_smooth)
    img_rgba.putalpha(alpha)
    
    PAD = 15
    cx1 = max(0, x1 - PAD)
    cy1 = max(0, y1 - PAD)
    cx2 = min(W, x2 + PAD)
    cy2 = min(H, y2 + PAD)
    crop = img_rgba.crop((cx1, cy1, cx2, cy2))
    
    name = "object_" + str(i+1)
    crop_path = "__CROPS_DIR__/" + name + ".png"
    crop.save(crop_path)
    
    results.append({
        "name": name,
        "label": phrase,
        "box": [x1, y1, x2, y2],
        "final_box": [cx1, cy1, cx2, cy2],
        "confidence": float(logit),
        "mask_score": float(scores[0]),
        "crop_url": "/crops/" + name + ".png",
        "crop_path": crop_path
    })

# Lưu ảnh trực quan hóa mặt nạ SAM2
Image.fromarray(visual_img).save("__CROPS_DIR__/sam2_visual.png")

with open("__CROPS_DIR__/sam2_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("✅ Tách nền SAM2 hoàn tất.")
"""

    # Safe string substitution - no f-string parsing issues
    script = script.replace("__INPUT_IMAGE_PATH__", input_image_path)
    script = script.replace("__LABELS_LIST__", repr(labels_list))
    script = script.replace("__CROPS_DIR__", crops_dir)

    r = subprocess.run([PY_PATH, "-c", script], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"Lỗi chạy Grounded-SAM2: {r.stderr}")
        
    result_json_path = os.path.join(crops_dir, "sam2_results.json")
    with open(result_json_path, 'r') as f:
        crops_data = json.load(f)
    return crops_data
