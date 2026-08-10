"""Resident SAM2 + GroundingDINO segmentation worker."""

import argparse
import io
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
matting_session = None
matting_remove = None
matting_error = None
inference_lock = threading.Lock()

MATTING_ENABLED = os.environ.get("ENABLE_MATTING", "1").lower() not in {
    "0",
    "false",
    "no",
}
MATTING_MODEL = os.environ.get("MATTING_MODEL", "isnet-general-use")
MATTING_MAX_SIDE = int(os.environ.get("MATTING_MAX_SIDE", "1024"))
MASK_MIN_COMPONENT_RATIO = float(
    os.environ.get("MASK_MIN_COMPONENT_RATIO", "0.00002")
)


class SegmentRequest(BaseModel):
    input_image_path: str = ""
    objects: list[dict]
    crops_dir: str
    source_mode: str = "objectwise"
    auto_detect: bool = False


class SanitizeRequest(BaseModel):
    objects: list[dict]


def _load_matting() -> None:
    """Load an optional alpha-matting refiner without blocking SAM2 startup."""
    global matting_session, matting_remove, matting_error

    if not MATTING_ENABLED:
        matting_error = "Matting disabled by ENABLE_MATTING"
        print("Alpha matting disabled; using SAM2 only.", flush=True)
        return

    try:
        from rembg import new_session, remove

        print(f"Loading alpha matting model: {MATTING_MODEL}...", flush=True)
        matting_session = new_session(MATTING_MODEL)
        matting_remove = remove
        matting_error = None
        print("Alpha matting refiner ready.", flush=True)
    except Exception as exc:
        # DINO + SAM2 remain usable when the optional refiner cannot download
        # its ONNX weights or the environment has no rembg installation.
        matting_session = None
        matting_remove = None
        matting_error = f"{exc}\n{traceback.format_exc()}"
        print(f"Alpha matting unavailable; using SAM2 fallback:\n{matting_error}", flush=True)


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
        _load_matting()
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
        "matting_ready": matting_session is not None and matting_remove is not None,
        "matting_model": MATTING_MODEL if MATTING_ENABLED else None,
        "matting_error": matting_error,
    }


def _clean_alpha(alpha: np.ndarray, box: list[int]) -> np.ndarray:
    """Remove faint shadows/noise while preserving thin furniture parts."""
    height, width = alpha.shape[:2]
    x1, y1, x2, y2 = [int(value) for value in box]
    allowed = np.zeros_like(alpha, dtype=np.uint8)
    allowed[max(0, y1):min(height, y2), max(0, x1):min(width, x2)] = 255
    alpha = np.minimum(alpha, allowed)
    alpha[alpha < 24] = 0

    support = alpha >= 32
    # Close tiny pinholes in the contour, but do not fill large holes such as
    # the space between chair legs, rails, or bicycle spokes.
    support = ndimage.binary_closing(
        support,
        structure=np.ones((3, 3), dtype=bool),
    )
    labels, count = ndimage.label(support)
    if count:
        sizes = np.asarray(
            ndimage.sum(support, labels, index=np.arange(1, count + 1)),
            dtype=np.int64,
        )
        minimum = max(12, int(alpha.size * MASK_MIN_COMPONENT_RATIO))
        keep_labels = np.where(sizes >= minimum)[0] + 1
        if keep_labels.size == 0:
            keep_labels = np.asarray([int(np.argmax(sizes)) + 1])
        support = np.isin(labels, keep_labels)
        alpha = np.where(support, alpha, 0).astype(np.uint8)

    alpha = ndimage.gaussian_filter(alpha.astype(np.float32), sigma=0.65)
    alpha[alpha < 24] = 0
    return np.clip(alpha, 0, 255).astype(np.uint8)


def _mask_alpha(masks, scores) -> tuple[np.ndarray, int]:
    best = int(np.argmax(scores))
    mask = ndimage.binary_closing(
        masks[best].astype(bool),
        structure=np.ones((3, 3), dtype=bool),
    )
    # Fill only small internal pinholes. Filling all holes can destroy the
    # negative spaces of chairs, tables, bicycle frames, and similar objects.
    holes = ndimage.binary_fill_holes(mask)
    small_holes = np.logical_and(holes, np.logical_not(mask))
    hole_labels, hole_count = ndimage.label(small_holes)
    if hole_count:
        hole_sizes = np.asarray(
            ndimage.sum(small_holes, hole_labels, index=np.arange(1, hole_count + 1)),
            dtype=np.int64,
        )
        fill_labels = np.where(hole_sizes <= max(32, mask.size // 500))[0] + 1
        mask = np.logical_or(mask, np.isin(hole_labels, fill_labels))
    alpha = ndimage.gaussian_filter(mask.astype(np.float32) * 255.0, sigma=0.65)
    return np.clip(alpha, 0, 255).astype(np.uint8), best


def _matting_alpha(image: Image.Image, box: list[int]) -> np.ndarray | None:
    """Run general-object matting on a focused crop and place alpha on canvas."""
    global matting_session, matting_remove, matting_error

    if matting_session is None or matting_remove is None:
        return None

    try:
        width, height = image.size
        x1, y1, x2, y2 = [int(value) for value in box]
        box_width = max(1, x2 - x1)
        box_height = max(1, y2 - y1)
        padding = max(12, int(max(box_width, box_height) * 0.08))
        crop_box = (
            max(0, x1 - padding),
            max(0, y1 - padding),
            min(width, x2 + padding),
            min(height, y2 + padding),
        )
        crop = image.crop(crop_box).convert("RGB")
        original_size = crop.size
        scale = min(1.0, MATTING_MAX_SIDE / max(original_size))
        if scale < 1.0:
            crop = crop.resize(
                (
                    max(1, int(original_size[0] * scale)),
                    max(1, int(original_size[1] * scale)),
                ),
                Image.Resampling.LANCZOS,
            )

        buffer = io.BytesIO()
        crop.save(buffer, format="PNG")
        output = matting_remove(buffer.getvalue(), session=matting_session)
        if isinstance(output, Image.Image):
            cutout = output.convert("RGBA")
        else:
            cutout = Image.open(io.BytesIO(output)).convert("RGBA")
        alpha = np.asarray(cutout.getchannel("A"), dtype=np.uint8)
        if cutout.size != original_size:
            alpha = np.asarray(
                Image.fromarray(alpha).resize(
                    original_size,
                    Image.Resampling.LANCZOS,
                ),
                dtype=np.uint8,
            )

        # Keep only meaningful alpha. Very faint shadows should not become
        # mesh geometry, while the soft edge produced by matting is preserved.
        alpha[alpha < 12] = 0
        foreground = alpha >= 32
        area_ratio = float(foreground.mean()) if foreground.size else 0.0
        if area_ratio < 0.003 or area_ratio > 0.92:
            return None

        canvas = np.zeros((height, width), dtype=np.uint8)
        cx1, cy1, _, _ = crop_box
        crop_height, crop_width = alpha.shape
        canvas[cy1:cy1 + crop_height, cx1:cx1 + crop_width] = alpha
        return canvas
    except Exception as exc:
        # Matting is an enhancement, never a hard dependency for segmentation.
        # Disable it for the rest of this worker session and use SAM2 instead.
        matting_session = None
        matting_remove = None
        matting_error = f"Runtime matting disabled: {exc}\n{traceback.format_exc()}"
        print(matting_error, flush=True)
        return None


def _predict_refined_alpha(
    image: Image.Image,
    box: list[int],
) -> tuple[np.ndarray, int, float, str]:
    """Predict SAM2 structure and optionally refine the boundary with matting."""
    width, height = image.size
    x1, y1, x2, y2 = [int(value) for value in box]
    padding = max(12, int(min(width, height) * 0.035))
    sam_box = np.array(
        [
            max(0, x1 - padding),
            max(0, y1 - padding),
            min(width, x2 + padding),
            min(height, y2 + padding),
        ],
        dtype=np.float32,
    )

    sam_predictor.set_image(np.asarray(image.convert("RGB")))
    masks, scores, _ = sam_predictor.predict(
        box=sam_box,
        multimask_output=True,
    )
    sam_alpha, best_mask = _mask_alpha(masks, scores)
    alpha = sam_alpha
    method = "sam2"

    refined = _matting_alpha(image, [int(value) for value in sam_box])
    if refined is not None:
        sam_support = sam_alpha >= 32
        refined_support = refined >= 32
        intersection = np.logical_and(sam_support, refined_support).sum()
        union = np.logical_or(sam_support, refined_support).sum()
        overlap = float(intersection / max(1, union))
        if overlap >= 0.20:
            # Matting gives a cleaner edge, while a slightly dilated SAM mask
            # prevents it from absorbing a neighboring object or background.
            sam_gate = ndimage.binary_dilation(
                sam_support,
                structure=np.ones((5, 5), dtype=bool),
                iterations=1,
            )
            alpha = np.where(sam_gate, refined, 0).astype(np.uint8)
            method = f"sam2+{MATTING_MODEL}"

    # The detector box is a semantic guardrail: matting must not pull in a
    # neighboring chair/table even when the source image contains duplicates.
    allowed = np.zeros_like(alpha)
    sx1, sy1, sx2, sy2 = [int(value) for value in sam_box]
    allowed[sy1:sy2, sx1:sx2] = 255
    alpha = _clean_alpha(alpha, [sx1, sy1, sx2, sy2])
    return alpha, best_mask, float(scores[best_mask]), method


def _box_iou(first: list[int], second: list[int]) -> float:
    x1, y1, x2, y2 = first
    px1, py1, px2, py2 = second
    intersection = max(0, min(x2, px2) - max(x1, px1)) * max(
        0, min(y2, py2) - max(y1, py1)
    )
    first_area = max(1, (x2 - x1) * (y2 - y1))
    second_area = max(1, (px2 - px1) * (py2 - py1))
    return intersection / max(1, first_area + second_area - intersection)


def _detect_label_instances(
    image_path: str,
    label: str,
    box_threshold: float = 0.22,
) -> tuple[Image.Image, list[dict]]:
    image_source, image_tensor = dino_load_image(image_path)
    height, width = image_source.shape[:2]
    source_image = Image.open(image_path).convert("RGB")
    boxes, logits, _ = dino_predict(
        model=dino_model,
        image=image_tensor,
        caption=label.strip().lower() + ".",
        box_threshold=box_threshold,
        text_threshold=0.18,
    )

    candidates = []
    for index, normalized_box in enumerate(boxes.detach().cpu().numpy()):
        center_x, center_y, box_width, box_height = normalized_box.tolist()
        x1 = max(0, int((center_x - box_width / 2) * width))
        y1 = max(0, int((center_y - box_height / 2) * height))
        x2 = min(width, int((center_x + box_width / 2) * width))
        y2 = min(height, int((center_y + box_height / 2) * height))
        if x2 <= x1 or y2 <= y1 or (x2 - x1) * (y2 - y1) < 256:
            continue
        score_tensor = logits[index]
        confidence = float(
            score_tensor.max().item()
            if getattr(score_tensor, "ndim", 0)
            else score_tensor.item()
        )
        candidates.append(
            {"box": [x1, y1, x2, y2], "confidence": confidence}
        )

    # DINO sometimes returns near-identical boxes for one physical instance.
    # Suppress those boxes while retaining genuinely separate duplicate objects.
    candidates.sort(key=lambda item: item["confidence"], reverse=True)
    kept = []
    for candidate in candidates:
        if any(_box_iou(candidate["box"], item["box"]) >= 0.55 for item in kept):
            continue
        kept.append(candidate)

    # A detector may emit one large group box around two nearby chairs plus
    # one box for each chair. Remove group boxes that substantially contain
    # a smaller valid instance, then rank the remaining individual boxes.
    individual_boxes = []
    for candidate in kept:
        x1, y1, x2, y2 = candidate["box"]
        area = max(1, (x2 - x1) * (y2 - y1))
        contains_smaller = False
        for other in kept:
            if other is candidate:
                continue
            ox1, oy1, ox2, oy2 = other["box"]
            other_area = max(1, (ox2 - ox1) * (oy2 - oy1))
            intersection = max(0, min(x2, ox2) - max(x1, ox1)) * max(
                0, min(y2, oy2) - max(y1, oy1)
            )
            if area >= other_area * 1.35 and intersection / other_area >= 0.88:
                contains_smaller = True
                break
        if not contains_smaller:
            individual_boxes.append(candidate)
    if individual_boxes:
        kept = individual_boxes
    kept.sort(key=lambda item: item["confidence"], reverse=True)
    return source_image, kept


def _sanitize_object_images(objects: list[dict]) -> list[dict]:
    sanitized_jobs = []
    for item in objects:
        job = dict(item)
        image_path = str(job.get("image_path", ""))
        label = str(job.get("label", "object")).strip().lower() or "object"
        if not os.path.isfile(image_path):
            raise FileNotFoundError(image_path or "Missing generated object image")

        source_image, candidates = _detect_label_instances(image_path, label)
        job["detected_instances"] = len(candidates)
        job["count_validation"] = "detected" if candidates else "undetected"

        if not candidates:
            # Preserve uncommon open-vocabulary objects that DINO cannot name;
            # the later segmentation stage can still process the original image.
            job["sanitized"] = False
            job["segmentation_method"] = "not-detected"
            sanitized_jobs.append(job)
            continue

        selected = candidates[0]
        width, height = source_image.size
        x1, y1, x2, y2 = selected["box"]
        alpha, best_mask, mask_score, segmentation_method = _predict_refined_alpha(
            source_image,
            [x1, y1, x2, y2],
        )
        alpha_image = Image.fromarray(alpha)
        content_box = alpha_image.getbbox() or (x1, y1, x2, y2)

        rgba = source_image.convert("RGBA")
        rgba.putalpha(alpha_image)
        isolated = rgba.crop(content_box)
        max_object_size = (int(width * 0.82), int(height * 0.82))
        isolated.thumbnail(max_object_size, Image.Resampling.LANCZOS)

        clean_image = Image.new("RGB", (width, height), "white")
        paste_x = (width - isolated.width) // 2
        paste_y = (height - isolated.height) // 2
        clean_image.paste(isolated, (paste_x, paste_y), isolated)
        clean_image.save(image_path)

        job.update(
            {
                "sanitized": True,
                "kept_box": selected["box"],
                "kept_confidence": selected["confidence"],
                "mask_score": mask_score,
                "segmentation_method": segmentation_method,
            }
        )
        sanitized_jobs.append(job)
    return sanitized_jobs


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

        label = str(item.get("label", "object")).strip().lower() or "object"
        image, candidates = _detect_label_instances(image_path, label)
        width, height = image.size

        if candidates:
            box_values = candidates[0]["box"]
            confidence = candidates[0]["confidence"]
        else:
            margin = max(18, int(min(width, height) * 0.04))
            box_values = [margin, margin, width - margin, height - margin]
            confidence = 0.0
        alpha, best, mask_score, segmentation_method = _predict_refined_alpha(
            image,
            box_values,
        )

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
                "label": label,
                "confidence": confidence,
                "detected_instances": len(candidates),
                "mask_score": mask_score,
                "box": box_values,
                "final_box": [0, 0, width, height],
                "crop_path": crop_path,
                "crop_url": f"/crops/{name}.png",
                "source_mode": "objectwise_sam2",
                "segmentation_method": segmentation_method,
            }
        )

    _save_preview(previews, crops_dir)
    return results


@app.post("/sanitize")
def sanitize(payload: SanitizeRequest) -> dict:
    if dino_model is None or sam_predictor is None:
        raise HTTPException(
            status_code=503,
            detail=load_error or "Segmentation models are still loading",
        )
    if not payload.objects:
        raise HTTPException(status_code=400, detail="No objects were provided")

    with inference_lock:
        try:
            jobs = _sanitize_object_images(payload.objects)
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Generated-object sanitization failed: {exc}",
            ) from exc
    return {"jobs": jobs}


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
            alpha, best_mask, mask_score, segmentation_method = _predict_refined_alpha(
                source_image,
                [x1, y1, x2, y2],
            )
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
                    "mask_score": mask_score,
                    "box": [x1, y1, x2, y2],
                    "final_box": [x1, y1, x2, y2],
                    "crop_path": crop_path,
                    "crop_url": f"/crops/{name}.png",
                    "source_mode": "uploaded_grounded_sam2_auto",
                    "detector_fallback": bool(candidate.get("detector_fallback", False)),
                    "segmentation_method": segmentation_method,
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

        alpha, best_mask, mask_score, segmentation_method = _predict_refined_alpha(
            source_image,
            [x1, y1, x2, y2],
        )
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
                "mask_score": mask_score,
                "box": [x1, y1, x2, y2],
                "final_box": [x1, y1, x2, y2],
                "crop_path": crop_path,
                "crop_url": f"/crops/{name}.png",
                "source_mode": "uploaded_grounded_sam2",
                "detector_fallback": detector_fallback,
                "segmentation_method": segmentation_method,
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
