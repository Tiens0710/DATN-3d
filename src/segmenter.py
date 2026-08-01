import json
import os
import subprocess
from pathlib import Path


PY_PATH = "/opt/venv310/bin/python"
OBJECT_MANIFEST = "/kaggle/working/object_images_manifest.json"
UPLOAD_SEGMENTATION_META = "/kaggle/working/upload_segmentation_meta.json"


def _run_uploaded_grounded_sam2(input_image_path: str, layout: dict, crops_dir: str) -> list:
    """Detect objects in a user image with GroundingDINO, then mask them with SAM2."""
    objects = list(layout.get("objects", []))
    if not objects:
        raise ValueError("Upload segmentation needs at least one labeled object")

    with open(UPLOAD_SEGMENTATION_META, "w", encoding="utf-8") as file:
        json.dump(
            {
                "image_path": input_image_path,
                "crops_dir": crops_dir,
                "objects": objects,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    script_path = "/kaggle/working/run_uploaded_grounded_sam2.py"
    script = r'''
import json
import os
import sys
sys.path.insert(0, "/opt/venv310/lib/python3.10/site-packages")
sys.modules["triton"] = None

import numpy as np
import scipy.ndimage as ndimage
import torch
from PIL import Image, ImageOps
from groundingdino.util.inference import load_image, load_model, predict
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

with open("__META__", encoding="utf-8") as file:
    meta = json.load(file)

image_path = meta["image_path"]
crops_dir = meta["crops_dir"]
objects = meta["objects"]
dino_config = "/kaggle/working/groundingdino_ckpt/GroundingDINO_SwinT_OGC.py"
dino_checkpoint = "/kaggle/working/groundingdino_ckpt/groundingdino_swint_ogc.pth"
sam2_checkpoint = "/kaggle/working/sam2_ckpt/sam2_hiera_small.pt"

if not os.path.isfile(image_path):
    raise FileNotFoundError(image_path)
if not os.path.isfile(dino_config) or not os.path.isfile(dino_checkpoint):
    raise FileNotFoundError("GroundingDINO config/checkpoint is missing")

image_source, image_tensor = load_image(image_path)
height, width = image_source.shape[:2]
rgb = np.asarray(Image.open(image_path).convert("RGB"))
dino_model = load_model(dino_config, dino_checkpoint)
sam_predictor = SAM2ImagePredictor(
    build_sam2("sam2_hiera_s.yaml", sam2_checkpoint, device="cuda")
)
sam_predictor.set_image(rgb)

results = []
previews = []
for index, item in enumerate(objects, start=1):
    name = str(item.get("id") or f"{item.get('label', 'object')}_{index}")
    label = str(item.get("label", "furniture")).strip().lower() or "furniture"
    boxes, logits, _ = predict(
        model=dino_model,
        image=image_tensor,
        caption=label + ".",
        box_threshold=0.25,
        text_threshold=0.20,
    )

    detector_fallback = len(boxes) == 0
    if detector_fallback:
        x1, y1, x2, y2 = 0, 0, width, height
        confidence = 0.0
    else:
        best = int(torch.argmax(logits).item())
        center_x, center_y, box_width, box_height = boxes[best].tolist()
        x1 = max(0, int((center_x - box_width / 2) * width))
        y1 = max(0, int((center_y - box_height / 2) * height))
        x2 = min(width, int((center_x + box_width / 2) * width))
        y2 = min(height, int((center_y + box_height / 2) * height))
        confidence = float(logits[best].item())
        if x2 <= x1 or y2 <= y1:
            x1, y1, x2, y2 = 0, 0, width, height
            detector_fallback = True

    input_box = np.array([x1, y1, x2, y2], dtype=np.float32)
    masks, scores, _ = sam_predictor.predict(
        box=input_box,
        multimask_output=True,
    )
    best_mask = int(np.argmax(scores))
    mask = ndimage.binary_closing(
        masks[best_mask].astype(bool),
        structure=np.ones((3, 3)),
    )
    mask = ndimage.binary_fill_holes(mask)
    alpha = ndimage.gaussian_filter(mask.astype(np.float32) * 255.0, sigma=1.0)
    alpha = np.clip(alpha, 0, 255).astype(np.uint8)

    rgba = Image.open(image_path).convert("RGBA")
    rgba.putalpha(Image.fromarray(alpha))
    crop_path = os.path.join(crops_dir, f"{name}.png")
    rgba.save(crop_path)
    previews.append(
        Image.alpha_composite(
            Image.new("RGBA", rgba.size, "white"), rgba
        ).convert("RGB")
    )
    results.append({
        "name": name,
        "label": label,
        "confidence": confidence,
        "mask_score": float(scores[best_mask]),
        "box": [x1, y1, x2, y2],
        "final_box": [x1, y1, x2, y2],
        "crop_path": crop_path,
        "crop_url": f"/crops/{name}.png",
        "source_mode": "uploaded_grounded_sam2",
        "detector_fallback": detector_fallback,
    })

if previews:
    cards = [ImageOps.contain(image, (360, 360), Image.Resampling.LANCZOS) for image in previews]
    sheet = Image.new(
        "RGB",
        (360 * min(2, len(cards)), 360 * ((len(cards) + 1) // 2)),
        "white",
    )
    for index, card in enumerate(cards):
        sheet.paste(card, ((index % 2) * 360, (index // 2) * 360))
    sheet.save(os.path.join(crops_dir, "sam2_visual.png"))

with open(os.path.join(crops_dir, "sam2_results.json"), "w", encoding="utf-8") as file:
    json.dump(results, file, ensure_ascii=False, indent=2)
print(json.dumps(results))
'''.replace("__META__", UPLOAD_SEGMENTATION_META)
    Path(script_path).write_text(script, encoding="utf-8")
    result = subprocess.run([PY_PATH, script_path], capture_output=True, text=True, timeout=360)
    if result.returncode != 0:
        raise RuntimeError("Uploaded Grounded-SAM2 failed:\n" + result.stderr + "\n" + result.stdout)

    with open(os.path.join(crops_dir, "sam2_results.json"), encoding="utf-8") as file:
        return json.load(file)


def run_grounded_sam2(
    input_image_path: str,
    layout: dict,
    crops_dir: str,
    source_mode: str = "objectwise",
) -> list:
    """Segment either generated isolated images or an uploaded scene image."""
    if source_mode == "uploaded":
        return _run_uploaded_grounded_sam2(input_image_path, layout, crops_dir)

    if not os.path.isfile(OBJECT_MANIFEST):
        raise FileNotFoundError("Object images are missing. Run the SD3.5 object-wise stage first.")

    Path(crops_dir).mkdir(parents=True, exist_ok=True)
    script_path = "/kaggle/working/run_objectwise_sam2.py"
    script = r'''
import json
import os
import sys
sys.path.insert(0, "/opt/venv310/lib/python3.10/site-packages")
sys.modules["triton"] = None

import numpy as np
import scipy.ndimage as ndimage
from PIL import Image, ImageOps
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

MANIFEST = "__MANIFEST__"
CROPS_DIR = "__CROPS_DIR__"
SAM2_CKPT = "/kaggle/working/sam2_ckpt/sam2_hiera_small.pt"

with open(MANIFEST, encoding="utf-8") as file:
    objects = json.load(file)

predictor = SAM2ImagePredictor(build_sam2("sam2_hiera_s.yaml", SAM2_CKPT, device="cuda"))
results = []
previews = []
for item in objects:
    image = Image.open(item["image_path"]).convert("RGB")
    rgb = np.asarray(image)
    height, width = rgb.shape[:2]
    predictor.set_image(rgb)

    margin = max(18, int(min(width, height) * 0.04))
    box = np.array([margin, margin, width - margin, height - margin], dtype=np.float32)
    masks, scores, _ = predictor.predict(box=box, multimask_output=True)
    best = int(np.argmax(scores))
    mask = ndimage.binary_closing(masks[best].astype(bool), structure=np.ones((3, 3)))
    mask = ndimage.binary_fill_holes(mask)
    alpha = ndimage.gaussian_filter(mask.astype(np.float32) * 255.0, sigma=1.0)
    alpha = np.clip(alpha, 0, 255).astype(np.uint8)

    rgba = image.convert("RGBA")
    rgba.putalpha(Image.fromarray(alpha))
    name = item["name"]
    crop_path = os.path.join(CROPS_DIR, f"{name}.png")
    rgba.save(crop_path)
    previews.append(Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba).convert("RGB"))
    results.append({
        "name": name,
        "label": item["label"],
        "confidence": 1.0,
        "mask_score": float(scores[best]),
        "box": [margin, margin, width - margin, height - margin],
        "final_box": [0, 0, width, height],
        "crop_path": crop_path,
        "crop_url": f"/crops/{name}.png",
        "source_mode": "objectwise_sam2",
    })

if previews:
    cards = [ImageOps.contain(image, (360, 360), Image.Resampling.LANCZOS) for image in previews]
    sheet = Image.new("RGB", (360 * min(2, len(cards)), 360 * ((len(cards) + 1) // 2)), "white")
    for index, card in enumerate(cards):
        sheet.paste(card, ((index % 2) * 360, (index // 2) * 360))
    sheet.save(os.path.join(CROPS_DIR, "sam2_visual.png"))

with open(os.path.join(CROPS_DIR, "sam2_results.json"), "w", encoding="utf-8") as file:
    json.dump(results, file, ensure_ascii=False, indent=2)
print(json.dumps(results))
'''
    script = script.replace("__MANIFEST__", OBJECT_MANIFEST).replace("__CROPS_DIR__", crops_dir)
    Path(script_path).write_text(script, encoding="utf-8")
    result = subprocess.run([PY_PATH, script_path], capture_output=True, text=True, timeout=360)
    if result.returncode != 0:
        raise RuntimeError("Object-wise SAM2 failed:\n" + result.stderr + "\n" + result.stdout)

    with open(os.path.join(crops_dir, "sam2_results.json"), encoding="utf-8") as file:
        return json.load(file)
