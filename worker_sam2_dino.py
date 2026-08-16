"""Resident SAM2 + GroundingDINO segmentation worker."""

import argparse
import gc
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
DINO_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# The legacy GroundingDINO BERT wrapper can keep an unregistered child on CPU.
# Keep the fallback device after the first mismatch instead of moving the model
# back to CUDA on every request.
DINO_RUNTIME_DEVICE = DINO_DEVICE
DINO_CPU_FALLBACK = False

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
SAM_BOX_PADDING_RATIO = float(
    os.environ.get("SAM_BOX_PADDING_RATIO", "0.05")
)
SAM_BOX_PADDING_MAX = int(os.environ.get("SAM_BOX_PADDING_MAX", "96"))
SAM_ALPHA_BLUR_SIGMA = float(
    os.environ.get("SAM_ALPHA_BLUR_SIGMA", "0.35")
)
SAM_POINT_REFINEMENT = os.environ.get("SAM_POINT_REFINEMENT", "1").lower() not in {
    "0",
    "false",
    "no",
}


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
    global dino_model, sam_predictor, dino_load_image, load_error

    try:
        required_files = [DINO_CONFIG, DINO_CHECKPOINT, SAM2_CHECKPOINT]
        missing = [path for path in required_files if not os.path.isfile(path)]
        if missing:
            raise FileNotFoundError("Missing checkpoints: " + ", ".join(missing))

        _patch_groundingdino_transformers_compat()
        from groundingdino.util.inference import (
            load_image,
            load_model,
        )
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        print("Loading GroundingDINO and SAM2...", flush=True)
        loaded_dino = load_model(DINO_CONFIG, DINO_CHECKPOINT, device=DINO_DEVICE)
        loaded_sam = SAM2ImagePredictor(
            build_sam2(
                "sam2_hiera_s.yaml",
                SAM2_CHECKPOINT,
                device=DINO_DEVICE,
            )
        )

        dino_model = loaded_dino
        sam_predictor = loaded_sam
        dino_load_image = load_image
        load_error = None
        print("SAM2 + GroundingDINO worker ready.", flush=True)
        _load_matting()
    except Exception as exc:
        load_error = f"{exc}\n{traceback.format_exc()}"
        print(f"SAM2 + GroundingDINO worker failed to load:\n{load_error}", flush=True)


def _patch_groundingdino_transformers_compat() -> None:
    """Patch the old GroundingDINO positional device argument once.

    GroundingDINO's BertModelWarper passes ``device`` positionally to
    Transformers' ``get_extended_attention_mask``.  Newer Transformers
    versions can interpret that position as ``dtype``, which causes the
    runtime error ``dtype=torch.device`` during every detection request.
    """
    import importlib.util

    spec = importlib.util.find_spec("groundingdino")
    if spec is None or not spec.submodule_search_locations:
        return

    package_root = Path(next(iter(spec.submodule_search_locations)))
    bertwarper_path = (
        package_root / "models" / "GroundingDINO" / "bertwarper.py"
    )
    if not bertwarper_path.is_file():
        return

    source = bertwarper_path.read_text(encoding="utf-8")
    old_calls = (
        """extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape, device
        )""",
        """extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device=device)""",
    )
    replacement = """try:
            extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
                attention_mask, input_shape, device=device
            )
        except TypeError as exc:
            if "unexpected keyword argument 'device'" not in str(exc):
                raise
            extended_attention_mask = self.get_extended_attention_mask(
                attention_mask,
                input_shape,
                dtype=self.embeddings.word_embeddings.weight.dtype,
            )"""
    patched = source
    for old_call in old_calls:
        patched = patched.replace(old_call, replacement, 1)
    if patched != source:
        bertwarper_path.write_text(patched, encoding="utf-8")
        print(
            "Patched GroundingDINO bertwarper for current Transformers.",
            flush=True,
        )


def _prepare_dino_model() -> None:
    """Keep every registered and wrapped DINO module on one device."""
    global DINO_RUNTIME_DEVICE

    if dino_model is None:
        raise RuntimeError("GroundingDINO model is not loaded")

    dino_model.to(DINO_RUNTIME_DEVICE)
    dino_model.eval()

    # Older GroundingDINO builds wrap BERT in a custom module.  Explicitly
    # moving common submodules handles builds whose wrapper is not traversed
    # correctly by the parent module's .to() call.
    for name in (
        "bert",
        "backbone",
        "transformer",
        "input_proj",
        "feat_map",
        "class_embed",
        "bbox_embed",
    ):
        child = getattr(dino_model, name, None)
        if child is not None and hasattr(child, "to"):
            child.to(DINO_RUNTIME_DEVICE)


def _cuda_memory() -> str:
    try:
        if not torch.cuda.is_available():
            return "CUDA unavailable"
        gib = 1024 ** 3
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        return (
            f"allocated={torch.cuda.memory_allocated() / gib:.2f} GiB, "
            f"reserved={torch.cuda.memory_reserved() / gib:.2f} GiB, "
            f"free={free_bytes / gib:.2f}/{total_bytes / gib:.2f} GiB"
        )
    except Exception as exc:
        return f"CUDA memory query failed: {type(exc).__name__}: {exc}"


def _safe_cuda_cleanup() -> None:
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception as exc:
        print(f"CUDA cleanup skipped: {exc}", flush=True)


def _preferred_inference_device() -> str:
    if DINO_CPU_FALLBACK or not torch.cuda.is_available():
        return "cpu"
    return "cuda"


def _move_models(device: str) -> None:
    """Move both segmentation models together so they do not pin VRAM."""
    global DINO_RUNTIME_DEVICE

    if dino_model is None or sam_predictor is None:
        raise RuntimeError("Segmentation models are not loaded")

    target = "cpu" if device == "cpu" else _preferred_inference_device()
    DINO_RUNTIME_DEVICE = target

    if target == "cpu":
        reset_predictor = getattr(sam_predictor, "reset_predictor", None)
        if callable(reset_predictor):
            reset_predictor()

    _prepare_dino_model()
    sam_model = getattr(sam_predictor, "model", None)
    if sam_model is not None and hasattr(sam_model, "to"):
        sam_model.to(target)
        sam_model.eval()

    if target == "cpu":
        _safe_cuda_cleanup()


def _dino_forward(image_tensor: torch.Tensor, caption: str):
    """Forward DINO and recover from partially-device-mapped legacy builds."""
    global DINO_RUNTIME_DEVICE, DINO_CPU_FALLBACK

    _prepare_dino_model()
    normalized_caption = str(caption).lower().strip()
    if not normalized_caption.endswith("."):
        normalized_caption += "."

    try:
        image = image_tensor.to(DINO_RUNTIME_DEVICE)
        with torch.no_grad():
            return dino_model(image[None], captions=[normalized_caption])
    except RuntimeError as exc:
        if "same device" not in str(exc).lower():
            raise

        # Some old GroundingDINO wrappers keep an unregistered BERT child on
        # CPU. Moving the complete model to CPU makes all tensors consistent
        # and keeps the API usable instead of returning HTTP 500.
        print(
            "GroundingDINO device mismatch; retrying the complete model on CPU.",
            flush=True,
        )
        DINO_RUNTIME_DEVICE = "cpu"
        DINO_CPU_FALLBACK = True
        dino_model.to("cpu")
        dino_model.eval()
        with torch.no_grad():
            return dino_model(image_tensor.to("cpu")[None], captions=[normalized_caption])


@app.on_event("startup")
def startup_event() -> None:
    _load_models()


def _predict_dino(
    image_tensor: torch.Tensor,
    caption: str,
    box_threshold: float,
    text_threshold: float,
):
    """Run GroundingDINO without the fragile utility wrapper.

    Some Kaggle GroundingDINO installs expose a ``predict`` helper that
    eventually passes a ``torch.device`` as a tensor dtype.  That produces
    ``to(dtype=torch.device)`` and breaks every uploaded-image request.  The
    model forward and official post-processing are small enough to keep here,
    so the worker uses an explicit device move and never relies on that helper.
    """
    if dino_model is None:
        raise RuntimeError("GroundingDINO model is not loaded")

    from groundingdino.util.utils import get_phrases_from_posmap

    normalized_caption = str(caption).lower().strip()
    if not normalized_caption.endswith("."):
        normalized_caption += "."

    outputs = _dino_forward(image_tensor, normalized_caption)

    prediction_logits = outputs["pred_logits"].detach().sigmoid()[0].cpu()
    prediction_boxes = outputs["pred_boxes"].detach()[0].cpu()
    scores = prediction_logits.max(dim=1).values
    keep = scores > float(box_threshold)
    boxes = prediction_boxes[keep]
    logits = prediction_logits[keep]

    tokenizer = dino_model.tokenizer
    tokenized = tokenizer(normalized_caption)
    phrases = [
        get_phrases_from_posmap(logit > float(text_threshold), tokenized, tokenizer)
        for logit in logits
    ]
    return boxes, logits, phrases


@app.get("/health")
def health() -> dict:
    return {
        "ready": dino_model is not None and sam_predictor is not None,
        "error": load_error,
        "device": DINO_RUNTIME_DEVICE,
        "cuda_memory": _cuda_memory(),
        "matting_ready": matting_session is not None and matting_remove is not None,
        "matting_model": MATTING_MODEL if MATTING_ENABLED else None,
        "matting_error": matting_error,
    }


@app.post("/offload")
def offload() -> dict:
    if dino_model is None or sam_predictor is None:
        raise HTTPException(
            status_code=503,
            detail=load_error or "Segmentation models are still loading",
        )

    with inference_lock:
        try:
            _move_models("cpu")
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to offload SAM2/DINO models: {exc}",
            ) from exc
    return {
        "ready": True,
        "device": DINO_RUNTIME_DEVICE,
        "cuda_memory": _cuda_memory(),
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

    alpha = ndimage.gaussian_filter(
        alpha.astype(np.float32),
        sigma=max(0.0, SAM_ALPHA_BLUR_SIGMA),
    )
    alpha[alpha < 24] = 0
    return np.clip(alpha, 0, 255).astype(np.uint8)


def _mask_extent(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    rows, columns = np.where(mask)
    if not rows.size or not columns.size:
        return None
    return (
        int(columns.min()),
        int(rows.min()),
        int(columns.max()) + 1,
        int(rows.max()) + 1,
    )


def _mask_completeness(mask, sam_score: float, box: list[int]) -> float:
    """Rank SAM candidates by score and object extent, not score alone."""
    x1, y1, x2, y2 = [int(value) for value in box]
    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)
    box_area = box_width * box_height
    support = mask.astype(bool)
    extent = _mask_extent(support)
    if extent is None:
        return -1.0

    mx1, my1, mx2, my2 = extent
    width_coverage = min(1.0, (mx2 - mx1) / box_width)
    height_coverage = min(1.0, (my2 - my1) / box_height)
    area_ratio = float(support.sum() / max(1, box_area))
    plausible_area = 1.0 if 0.035 <= area_ratio <= 1.05 else 0.0

    # Thin furniture often has a lower raw SAM score than a seat-only or
    # tabletop-only mask. Extent coverage therefore carries more weight.
    return (
        float(sam_score)
        + 0.42 * height_coverage
        + 0.28 * width_coverage
        + 0.08 * plausible_area
    )


def _select_complete_mask(masks, scores, box: list[int]) -> int:
    """Prefer a complete object mask over a high-score partial component."""
    ranked = [
        (_mask_completeness(candidate, float(scores[index]), box), index)
        for index, candidate in enumerate(masks)
    ]

    return max(ranked)[1]


def _mask_alpha(masks, scores, box: list[int]) -> tuple[np.ndarray, int]:
    best = _select_complete_mask(masks, scores, box)
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
    alpha = ndimage.gaussian_filter(
        mask.astype(np.float32) * 255.0,
        sigma=max(0.0, SAM_ALPHA_BLUR_SIGMA),
    )
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
    label: str = "",
) -> tuple[np.ndarray, int, float, str]:
    """Predict SAM2 structure and optionally refine the boundary with matting."""
    width, height = image.size
    x1, y1, x2, y2 = [int(value) for value in box]
    normalized_label = str(label).strip().lower()
    # Tables need a tighter prompt box because the COCO scene often contains
    # floor/wall regions close to the tabletop. Chairs and sofas keep a wider
    # margin so SAM2 can recover thin legs, arms, and rails.
    label_padding = {
        "table": (0.035, 64),
        "dining table": (0.035, 64),
        "chair": (0.045, 88),
        "dining chair": (0.045, 88),
        "sofa": (0.05, 96),
        "couch": (0.05, 96),
    }.get(normalized_label, (SAM_BOX_PADDING_RATIO, SAM_BOX_PADDING_MAX))
    padding_ratio, padding_max = label_padding
    padding = max(
        16,
        min(
            padding_max,
            int(max(width, height) * padding_ratio),
        ),
    )
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
    sam_box_values = [int(value) for value in sam_box]
    best_mask = _select_complete_mask(masks, scores, sam_box_values)
    initial_quality = _mask_completeness(
        masks[best_mask],
        float(scores[best_mask]),
        sam_box_values,
    )

    # A positive point at the deepest interior location helps SAM2 choose the
    # whole object when the box-only candidates prefer a tabletop or seat.
    # Keep the box-only result as a fallback when point refinement loses extent.
    selected_masks = masks
    selected_scores = scores
    if SAM_POINT_REFINEMENT:
        initial_support = masks[best_mask].astype(bool)
        distance = ndimage.distance_transform_edt(initial_support)
        if distance.size and float(distance.max()) > 0:
            point_y, point_x = np.unravel_index(int(distance.argmax()), distance.shape)
        else:
            point_x = int((sam_box[0] + sam_box[2]) / 2)
            point_y = int((sam_box[1] + sam_box[3]) / 2)
        try:
            refined_masks, refined_scores, _ = sam_predictor.predict(
                point_coords=np.asarray([[point_x, point_y]], dtype=np.float32),
                point_labels=np.asarray([1], dtype=np.int32),
                box=sam_box,
                multimask_output=True,
            )
            refined_best = _select_complete_mask(
                refined_masks,
                refined_scores,
                sam_box_values,
            )
            refined_quality = _mask_completeness(
                refined_masks[refined_best],
                float(refined_scores[refined_best]),
                sam_box_values,
            )
            if refined_quality >= initial_quality - 0.03:
                selected_masks = refined_masks
                selected_scores = refined_scores
                best_mask = refined_best
        except Exception as exc:
            print(f"SAM2 point refinement skipped: {exc}", flush=True)

    sam_alpha, best_mask = _mask_alpha(
        selected_masks,
        selected_scores,
        sam_box_values,
    )
    alpha = sam_alpha
    method = "sam2"

    refined = _matting_alpha(image, sam_box_values)
    if refined is not None:
        sam_support = sam_alpha >= 32
        refined_support = refined >= 32
        intersection = np.logical_and(sam_support, refined_support).sum()
        union = np.logical_or(sam_support, refined_support).sum()
        overlap = float(intersection / max(1, union))
        sam_extent = _mask_extent(sam_support)
        refined_extent = _mask_extent(refined_support)
        preserves_extent = False
        if sam_extent is not None and refined_extent is not None:
            smx1, smy1, smx2, smy2 = sam_extent
            rmx1, rmy1, rmx2, rmy2 = refined_extent
            preserves_extent = (
                (rmx2 - rmx1) >= 0.85 * max(1, smx2 - smx1)
                and (rmy2 - rmy1) >= 0.85 * max(1, smy2 - smy1)
            )
        if overlap >= 0.20 and preserves_extent:
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
    sx1, sy1, sx2, sy2 = sam_box_values
    allowed[sy1:sy2, sx1:sx2] = 255
    alpha = _clean_alpha(alpha, [sx1, sy1, sx2, sy2])
    final_extent = _mask_extent(alpha >= 32)
    if final_extent is None:
        raise ValueError("SAM2 returned an empty object mask")
    fmx1, fmy1, fmx2, fmy2 = final_extent
    width_coverage = (fmx2 - fmx1) / max(1, sx2 - sx1)
    height_coverage = (fmy2 - fmy1) / max(1, sy2 - sy1)
    if width_coverage < 0.38 or height_coverage < 0.48:
        raise ValueError(
            "SAM2 returned an incomplete object mask "
            f"(width coverage={width_coverage:.2f}, height coverage={height_coverage:.2f})"
        )
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
    normalized_label = label.strip().lower()
    # COCO's couch and dining-table wording is not always the strongest query
    # for GroundingDINO. Use short, object-specific aliases while keeping the
    # reported label unchanged.
    query_label = {
        "couch": "sofa",
        "dining table": "table",
    }.get(normalized_label, normalized_label)
    label_thresholds = {
        "chair": (0.18, 0.14),
        "couch": (0.18, 0.14),
        "dining table": (0.20, 0.15),
    }
    box_threshold, text_threshold = label_thresholds.get(
        normalized_label,
        (box_threshold, 0.18),
    )
    boxes, logits, _ = _predict_dino(
        image_tensor=image_tensor,
        caption=query_label + ".",
        box_threshold=box_threshold,
        text_threshold=text_threshold,
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
            if label in {"chair", "dining chair", "table", "dining table", "sofa", "bed"}:
                raise ValueError(
                    f"GroundingDINO could not verify the generated '{label}'. "
                    "Regenerate the 2D object instead of forwarding an unchecked image."
                )
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
            label=label,
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
            label=label,
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
            _move_models(_preferred_inference_device())
            jobs = _sanitize_object_images(payload.objects)
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Generated-object sanitization failed: {exc}",
            ) from exc
        finally:
            try:
                _move_models("cpu")
            except Exception as cleanup_exc:
                print(
                    f"Failed to offload SAM2/DINO after sanitization: {cleanup_exc}",
                    flush=True,
                )
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
        boxes, logits, phrases = _predict_dino(
            image_tensor=image_tensor,
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
                label=str(candidate.get("label", "object")),
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
    detection_cache = {}
    label_occurrences = {}
    for index, item in enumerate(objects, start=1):
        label = str(item.get("label", "furniture")).strip().lower() or "furniture"
        safe_label = re.sub(r"[^a-z0-9]+", "_", label).strip("_") or "object"
        requested_name = str(item.get("id", "")).strip()
        name = (
            requested_name
            if re.fullmatch(r"[A-Za-z0-9_-]{1,80}", requested_name)
            else f"{safe_label}_{index}"
        )

        if label not in detection_cache:
            label_thresholds = {
                "chair": (0.20, 0.16),
                "dining chair": (0.20, 0.16),
                "sofa": (0.20, 0.16),
                "couch": (0.20, 0.16),
                "table": (0.22, 0.17),
                "dining table": (0.22, 0.17),
            }
            query_label = {
                "couch": "sofa",
                "dining chair": "chair",
                "dining table": "table",
            }.get(label, label)
            box_threshold, text_threshold = label_thresholds.get(
                label,
                (0.25, 0.20),
            )
            boxes, logits, _ = _predict_dino(
                image_tensor=image_tensor,
                caption=query_label + ".",
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )
            scores = logits.detach().cpu()
            if scores.ndim > 1:
                scores = scores.max(dim=1).values
            candidates = []
            for box_index, normalized_box in enumerate(boxes.detach().cpu()):
                center_x, center_y, box_width, box_height = normalized_box.tolist()
                candidate_box = [
                    max(0, int((center_x - box_width / 2) * width)),
                    max(0, int((center_y - box_height / 2) * height)),
                    min(width, int((center_x + box_width / 2) * width)),
                    min(height, int((center_y + box_height / 2) * height)),
                ]
                x1, y1, x2, y2 = candidate_box
                if x2 <= x1 or y2 <= y1:
                    continue
                candidates.append({
                    "box": candidate_box,
                    "confidence": float(scores[box_index].item()),
                })
            candidates.sort(key=lambda candidate: candidate["confidence"], reverse=True)
            kept = []
            for candidate in candidates:
                if any(
                    _box_iou(candidate["box"], previous["box"]) >= 0.55
                    for previous in kept
                ):
                    continue
                kept.append(candidate)
            detection_cache[label] = kept

        occurrence = label_occurrences.get(label, 0)
        label_occurrences[label] = occurrence + 1
        candidates = detection_cache[label]
        if occurrence >= len(candidates):
            raise ValueError(
                f"GroundingDINO found {len(candidates)} instance(s) of '{label}', "
                f"but {occurrence + 1} were requested"
            )
        selected = candidates[occurrence]
        x1, y1, x2, y2 = selected["box"]
        confidence = selected["confidence"]
        detector_fallback = False

        alpha, best_mask, mask_score, segmentation_method = _predict_refined_alpha(
            source_image,
            [x1, y1, x2, y2],
            label=label,
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
            _move_models(_preferred_inference_device())
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
                detail=(
                    f"Segmentation failed in {source_mode} mode: {exc}\n"
                    f"{traceback.format_exc()}"
                ),
            ) from exc
        finally:
            try:
                _move_models("cpu")
            except Exception as cleanup_exc:
                print(
                    f"Failed to offload SAM2/DINO after segmentation: {cleanup_exc}",
                    flush=True,
                )

    return {"results": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resident SAM2 + DINO worker")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8003)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
