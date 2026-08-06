"""Resident SAM2 + GroundingDINO segmentation worker."""

import argparse
import json
import os
import re
import sys
import threading
import traceback
from pathlib import Path


os.environ.setdefault("MPLBACKEND", "agg")
sys.path.insert(0, "/opt/venv310/lib/python3.10/site-packages")
sys.path.insert(0, "/kaggle/working")
sys.modules["triton"] = None

import numpy as np
import scipy.ndimage as ndimage
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image, ImageOps
from pydantic import BaseModel


DINO_CONFIG = "/kaggle/working/groundingdino_ckpt/GroundingDINO_SwinT_OGC.py"
DINO_CHECKPOINT = (
    "/kaggle/working/groundingdino_ckpt/groundingdino_swint_ogc.pth"
)
SAM2_CHECKPOINT = "/kaggle/working/sam2_ckpt/sam2_hiera_small.pt"

# Used only when the upload workflow is in automatic mode. GroundingDINO is
# open-vocabulary, but it still needs a text prompt; this broad furniture
# vocabulary lets it discover one or many objects without hard-coded labels
# such as "chair, table" in the UI.
AUTO_DETECTION_CAPTION = (
    "chair. table. sofa. couch. bed. desk. cabinet. wardrobe. shelf. "
    "bookcase. stool. bench. ottoman. lamp. dresser. nightstand. "
    "coffee table. dining table. furniture."
)

app = FastAPI(title="DATN SAM2 + GroundingDINO Worker", version="1.0.0")
dino_model = None
sam_predictor = None
dino_load_image = None
dino_predict = None
load_error = None
inference_lock = threading.Lock()


class SegmentRequest(BaseModel):
    input_image_path: str = ""
    objects: list[dict]
    crops_dir: str
    source_mode: str = "objectwise"
    auto_detect: bool = False


def _load_models() -> None:
    global dino_model, sam_predictor, dino_load_image, dino_predict, load_error

    try:
        required_files = [DINO_CONFIG, DINO_CHECKPOINT, SAM2_CHECKPOINT]
        missing = [path for path in required_files if not os.path.isfile(path)]
        if missing:
            raise FileNotFoundError("Missing checkpoints: " + ", ".join(missing))

        from groundingdino.util.inference import (
            load_image,
            load_model,
            predict,
        )
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        print("Loading GroundingDINO and SAM2...", flush=True)
        loaded_dino = load_model(DINO_CONFIG, DINO_CHECKPOINT, device="cuda")
        loaded_sam = SAM2ImagePredictor(
            build_sam2(
                "sam2_hiera_s.yaml",
                SAM2_CHECKPOINT,
                device="cuda",
            )
        )

        dino_model = loaded_dino
        sam_predictor = loaded_sam
        dino_load_image = load_image
        dino_predict = predict
        load_error = None
        print("SAM2 + GroundingDINO worker ready.", flush=True)
    except Exception as exc:
        load_error = f"{exc}\n{traceback.format_exc()}"
        print(f"SAM2 + GroundingDINO worker failed to load:\n{load_error}", flush=True)


@app.on_event("startup")
def startup_event() -> None:
    _load_models()


@app.get("/health")
def health() -> dict:
    return {
        "ready": dino_model is not None and sam_predictor is not None,
        "error": load_error,
    }


def _mask_alpha(masks, scores) -> tuple[np.ndarray, int]:
    best = int(np.argmax(scores))
    mask = ndimage.binary_closing(
        masks[best].astype(bool),
        structure=np.ones((3, 3)),
    )
    mask = ndimage.binary_fill_holes(mask)
    alpha = ndimage.gaussian_filter(mask.astype(np.float32) * 255.0, sigma=1.0)
    return np.clip(alpha, 0, 255).astype(np.uint8), best


def _save_preview(previews: list, crops_dir: str) -> None:
    if not previews:
        return
    cards = [
        ImageOps.contain(image, (360, 360), Image.Resampling.LANCZOS)
        for image in previews
    ]
    sheet = Image.new(
        "RGB",
        (360 * min(2, len(cards)), 360 * ((len(cards) + 1) // 2)),
        "white",
    )
    for index, card in enumerate(cards):
        sheet.paste(card, ((index % 2) * 360, (index // 2) * 360))
    sheet.save(os.path.join(crops_dir, "sam2_visual.png"))


def _segment_objectwise(objects: list[dict], crops_dir: str) -> list[dict]:
    results = []
    previews = []
    for item in objects:
        image_path = str(item.get("image_path", ""))
        if not os.path.isfile(image_path):
            raise FileNotFoundError(image_path or "Missing object image path")

        image = Image.open(image_path).convert("RGB")
        rgb = np.asarray(image)
        height, width = rgb.shape[:2]
        sam_predictor.set_image(rgb)

        margin = max(18, int(min(width, height) * 0.04))
        box = np.array(
            [margin, margin, width - margin, height - margin],
            dtype=np.float32,
        )
        masks, scores, _ = sam_predictor.predict(
            box=box,
            multimask_output=True,
        )
        alpha, best = _mask_alpha(masks, scores)

        rgba = image.convert("RGBA")
        rgba.putalpha(Image.fromarray(alpha))
        name = str(item.get("name", "object"))
        crop_path = os.path.join(crops_dir, f"{name}.png")
        rgba.save(crop_path)
        previews.append(
            Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba)
            .convert("RGB")
        )
        results.append(
            {
                "name": name,
                "label": item.get("label", name),
                "confidence": 1.0,
                "mask_score": float(scores[best]),
                "box": [margin, margin, width - margin, height - margin],
                "final_box": [0, 0, width, height],
                "crop_path": crop_path,
                "crop_url": f"/crops/{name}.png",
                "source_mode": "objectwise_sam2",
            }
        )

    _save_preview(previews, crops_dir)
    return results


def _segment_uploaded(
    image_path: str,
    objects: list[dict],
    crops_dir: str,
    auto_detect: bool = False,
) -> list[dict]:
    if not os.path.isfile(image_path):
        raise FileNotFoundError(image_path)

    image_source, image_tensor = dino_load_image(image_path)
    height, width = image_source.shape[:2]
    source_image = Image.open(image_path).convert("RGB")
    sam_predictor.set_image(np.asarray(source_image))

    if auto_detect:
        boxes, logits, phrases = dino_predict(
            model=dino_model,
            image=image_tensor,
            caption=AUTO_DETECTION_CAPTION,
            box_threshold=0.22,
            text_threshold=0.16,
        )

        box_array = boxes.detach().cpu().numpy()
        logit_array = logits.detach().cpu().numpy()
        if logit_array.ndim > 1:
            score_array = logit_array.max(axis=1)
        else:
            score_array = logit_array

        candidates = []
        for index, normalized_box in enumerate(box_array):
            center_x, center_y, box_width, box_height = normalized_box.tolist()
            x1 = max(0, int((center_x - box_width / 2) * width))
            y1 = max(0, int((center_y - box_height / 2) * height))
            x2 = min(width, int((center_x + box_width / 2) * width))
            y2 = min(height, int((center_y + box_height / 2) * height))
            if x2 <= x1 or y2 <= y1 or (x2 - x1) * (y2 - y1) < 256:
                continue

            label = str(phrases[index]).strip() if index < len(phrases) else "object"
            candidates.append(
                {
                    "box": [x1, y1, x2, y2],
                    "label": label or "object",
                    "confidence": float(score_array[index]),
                }
            )

        # GroundingDINO can return overlapping boxes for the same object when
        # the caption contains related labels (e.g. sofa/couch). Keep the
        # strongest box and remove near-duplicates before invoking SAM2.
        candidates.sort(key=lambda item: item["confidence"], reverse=True)
        kept = []
        for candidate in candidates:
            x1, y1, x2, y2 = candidate["box"]
            area = max(1, (x2 - x1) * (y2 - y1))
            duplicate = False
            for previous in kept:
                px1, py1, px2, py2 = previous["box"]
                intersection = max(0, min(x2, px2) - max(x1, px1)) * max(
                    0, min(y2, py2) - max(y1, py1)
                )
                previous_area = max(1, (px2 - px1) * (py2 - py1))
                union = area + previous_area - intersection
                if intersection / union >= 0.55:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(candidate)

        # If DINO cannot find a furniture phrase, preserve the one-object
        # behavior with a full-image fallback instead of returning zero crops.
        if not kept:
            kept = [
                {
                    "box": [0, 0, width, height],
                    "label": "object",
                    "confidence": 0.0,
                    "detector_fallback": True,
                }
            ]

        results = []
        previews = []
        for index, candidate in enumerate(kept, start=1):
            x1, y1, x2, y2 = candidate["box"]
            masks, scores, _ = sam_predictor.predict(
                box=np.array([x1, y1, x2, y2], dtype=np.float32),
                multimask_output=True,
            )
            alpha, best_mask = _mask_alpha(masks, scores)
            rgba = source_image.convert("RGBA")
            rgba.putalpha(Image.fromarray(alpha))
            safe_label = re.sub(r"[^a-z0-9]+", "_", candidate["label"].lower()).strip("_")
            name = f"{safe_label or 'object'}_{index}"
            crop_path = os.path.join(crops_dir, f"{name}.png")
            rgba.save(crop_path)
            previews.append(
                Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba)
                .convert("RGB")
            )
            results.append(
                {
                    "name": name,
                    "label": candidate["label"],
                    "confidence": candidate["confidence"],
                    "mask_score": float(scores[best_mask]),
                    "box": [x1, y1, x2, y2],
                    "final_box": [x1, y1, x2, y2],
                    "crop_path": crop_path,
                    "crop_url": f"/crops/{name}.png",
                    "source_mode": "uploaded_grounded_sam2_auto",
                    "detector_fallback": bool(candidate.get("detector_fallback", False)),
                }
            )

        _save_preview(previews, crops_dir)
        return results

    results = []
    previews = []
    for index, item in enumerate(objects, start=1):
        name = str(item.get("id") or f"{item.get('label', 'object')}_{index}")
        label = str(item.get("label", "furniture")).strip().lower() or "furniture"
        boxes, logits, _ = dino_predict(
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
            best_box = int(torch.argmax(logits).item())
            center_x, center_y, box_width, box_height = boxes[best_box].tolist()
            x1 = max(0, int((center_x - box_width / 2) * width))
            y1 = max(0, int((center_y - box_height / 2) * height))
            x2 = min(width, int((center_x + box_width / 2) * width))
            y2 = min(height, int((center_y + box_height / 2) * height))
            confidence = float(logits[best_box].item())
            if x2 <= x1 or y2 <= y1:
                x1, y1, x2, y2 = 0, 0, width, height
                detector_fallback = True

        masks, scores, _ = sam_predictor.predict(
            box=np.array([x1, y1, x2, y2], dtype=np.float32),
            multimask_output=True,
        )
        alpha, best_mask = _mask_alpha(masks, scores)
        rgba = source_image.convert("RGBA")
        rgba.putalpha(Image.fromarray(alpha))
        crop_path = os.path.join(crops_dir, f"{name}.png")
        rgba.save(crop_path)
        previews.append(
            Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba)
            .convert("RGB")
        )
        results.append(
            {
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
            }
        )

    _save_preview(previews, crops_dir)
    return results


@app.post("/segment")
def segment(payload: SegmentRequest) -> dict:
    if dino_model is None or sam_predictor is None:
        raise HTTPException(
            status_code=503,
            detail=load_error or "Segmentation models are still loading",
        )
    if not payload.objects:
        raise HTTPException(status_code=400, detail="No objects were provided")

    crops_dir = str(Path(payload.crops_dir))
    Path(crops_dir).mkdir(parents=True, exist_ok=True)
    source_mode = payload.source_mode.lower()

    with inference_lock:
        try:
            if source_mode == "uploaded":
                results = _segment_uploaded(
                    payload.input_image_path,
                    payload.objects,
                    crops_dir,
                    auto_detect=payload.auto_detect,
                )
            elif source_mode == "objectwise":
                results = _segment_objectwise(payload.objects, crops_dir)
            else:
                raise ValueError(f"Unsupported source_mode: {source_mode}")

            with open(
                os.path.join(crops_dir, "sam2_results.json"),
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(results, file, ensure_ascii=False, indent=2)
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Segmentation failed in {source_mode} mode: {exc}",
            ) from exc
        finally:
            torch.cuda.empty_cache()

    return {"results": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resident SAM2 + DINO worker")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8003)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
