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

app = FastAPI(title="DATN SAM2 + GroundingDINO Worker", version="1.0.2")
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
# A true product mask can be dense (for example a wardrobe), but it should not
# be an almost-solid rectangle that occupies the complete detector prompt box.
# This deliberately conservative threshold catches background cards/panels
# without rejecting normal box-shaped furniture.
BACKDROP_MASK_MIN_DENSITY = float(
    os.environ.get("BACKDROP_MASK_MIN_DENSITY", "0.88")
)
BACKDROP_BOX_COVERAGE = float(
    os.environ.get("BACKDROP_BOX_COVERAGE", "0.92")
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


def _looks_like_floor_attachment(alpha: np.ndarray) -> bool:
    """Detect a broad opaque sheet attached below an otherwise isolated asset.

    This intentionally uses only silhouette geometry, so it applies to every
    object class and every background colour.  Valid furniture normally ends
    in sparse feet/legs.  A rug, floor patch or support card remains broad and
    dense all the way to the lowest rows of the alpha extent.
    """
    support = np.asarray(alpha) >= 32
    extent = _mask_extent(support)
    if extent is None:
        return False
    x1, y1, x2, y2 = extent
    crop = support[y1:y2, x1:x2]
    height, width = crop.shape
    if height < 32 or width < 32:
        return False

    row_fill = crop.mean(axis=1)
    band_size = max(5, int(round(height * 0.18)))
    bottom = row_fill[-band_size:]
    tail = row_fill[-max(3, int(round(height * 0.07))):]
    middle = row_fill[
        int(round(height * 0.20)):max(int(round(height * 0.20)) + 1, int(round(height * 0.62)))
    ]
    broad_rows = float((bottom >= 0.55).mean())
    middle_fill = float(np.median(middle)) if middle.size else 0.0
    bottom_fill = float(np.median(bottom))
    return (
        bottom_fill >= 0.42
        and float(np.median(tail)) >= 0.28
        and broad_rows >= 0.45
        and bottom_fill >= middle_fill * 1.22
    )


def _generated_asset_semantic_error(
    image_path: str,
    label: str,
    target_confidence: float,
) -> str | None:
    """Reject clearly named support props and sofa/bed identity swaps.

    A target-only DINO query can still call a bed-like wooden frame a sofa.
    This second, narrow query is used only as a veto for high-confidence
    confusers; it does not replace the target detector or accept new objects.
    """
    normalized = " ".join(label.lower().replace("_", " ").split())
    confusers = ["rug", "carpet", "floor mat"]
    if "sofa" in normalized or "couch" in normalized:
        confusers.extend(["bed", "bed frame"])

    _source, image_tensor = dino_load_image(image_path)
    _boxes, logits, phrases = _predict_dino(
        image_tensor=image_tensor,
        caption=". ".join(confusers) + ".",
        box_threshold=0.22,
        text_threshold=0.16,
    )
    for index, phrase in enumerate(phrases):
        detected = str(phrase).strip().lower()
        score_tensor = logits[index]
        score = float(
            score_tensor.max().item()
            if getattr(score_tensor, "ndim", 0)
            else score_tensor.item()
        )
        if any(term in detected for term in ("rug", "carpet", "floor mat")):
            if score >= 0.34:
                return f"detected an unwanted support surface '{detected}' ({score:.2f})"
        elif ("bed" in detected) and score >= max(0.34, target_confidence * 1.05):
            return f"the requested sofa is visually a '{detected}' ({score:.2f})"
    return None


def _category_shape_warning(alpha: np.ndarray, label: str) -> str | None:
    """Catch category-specific silhouette failures before image-to-3D."""
    normalized = " ".join(label.lower().replace("_", " ").split())
    support = np.asarray(alpha) >= 32
    extent = _mask_extent(support)
    if extent is None:
        return "empty object silhouette"
    x1, y1, x2, y2 = extent
    crop = support[y1:y2, x1:x2]
    height, width = crop.shape
    if height < 8 or width < 8:
        return "object silhouette is too small"

    if "lamp" in normalized or "light" in normalized:
        aspect = height / max(1, width)
        middle = crop[int(height * 0.28):max(int(height * 0.28) + 1, int(height * 0.76))]
        middle_fill = float(np.median(middle.mean(axis=1))) if middle.size else 1.0
        if aspect < 1.25:
            return "floor lamp silhouette is not tall enough"
        if middle_fill > 0.42:
            return "floor lamp is a solid block instead of shade, thin stem and base"
    return None


def _mask_density(mask: np.ndarray, box: list[int]) -> float:
    """Measure foreground density inside a detector box."""
    height, width = mask.shape[:2]
    x1, y1, x2, y2 = [int(value) for value in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        return 1.0
    return float(mask[y1:y2, x1:x2].astype(bool).mean())


def _looks_like_rectangular_backdrop(alpha: np.ndarray, box: list[int]) -> bool:
    """Return whether a mask is an opaque rectangular background/card.

    This check is label-agnostic on purpose.  SAM2 can latch onto a panel,
    studio floor, or wall patch behind *any* object, not only a floor lamp.
    The thresholds are kept high so ordinary dense furniture is accepted.
    """
    support = alpha >= 32
    extent = _mask_extent(support)
    if extent is None:
        return False
    x1, y1, x2, y2 = [int(value) for value in box]
    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)
    ex1, ey1, ex2, ey2 = extent
    extent_width = ex2 - ex1
    extent_height = ey2 - ey1
    density = _mask_density(support, box)
    fills_box = (
        extent_width >= BACKDROP_BOX_COVERAGE * box_width
        and extent_height >= BACKDROP_BOX_COVERAGE * box_height
    )
    return fills_box and density >= BACKDROP_MASK_MIN_DENSITY


def _border_background_alpha(image: Image.Image, box: list[int]) -> np.ndarray | None:
    """Extract a product against the solid background estimated from image edges.

    Object-wise SD3.5 images are intentionally generated on a plain contrasting
    backdrop.  Estimating that backdrop from the border gives every object the
    same second segmentation path when SAM2 or matting keeps a wall/panel.
    """
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    height, width = rgb.shape[:2]
    if min(width, height) < 32:
        return None

    border = max(3, min(width, height) // 48)
    samples = np.concatenate(
        (
            rgb[:border, :, :].reshape(-1, 3),
            rgb[-border:, :, :].reshape(-1, 3),
            rgb[:, :border, :].reshape(-1, 3),
            rgb[:, -border:, :].reshape(-1, 3),
        ),
        axis=0,
    )
    background = np.median(samples, axis=0)
    border_distance = np.linalg.norm(samples - background, axis=1)
    threshold = max(22.0, float(np.percentile(border_distance, 96)) * 1.8 + 8.0)
    distance = np.linalg.norm(rgb - background, axis=2)
    alpha = np.where(distance >= threshold, 255, 0).astype(np.uint8)
    alpha = _clean_alpha(alpha, box)
    support = alpha >= 32
    if _mask_extent(support) is None or _looks_like_rectangular_backdrop(alpha, box):
        return None
    return alpha


def _uniform_border(image: Image.Image) -> bool:
    """Return whether the image border looks like a plain studio backdrop."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    height, width = rgb.shape[:2]
    if min(width, height) < 32:
        return False
    border = max(3, min(width, height) // 48)
    samples = np.concatenate(
        (
            rgb[:border, :, :].reshape(-1, 3),
            rgb[-border:, :, :].reshape(-1, 3),
            rgb[:, :border, :].reshape(-1, 3),
            rgb[:, -border:, :].reshape(-1, 3),
        ),
        axis=0,
    )
    # White/grey studio backgrounds have very little colour variation on the
    # border. This is deliberately only used as a signal that a mask may be
    # inverted; it is not itself a segmentation decision.
    return float(np.mean(np.std(samples, axis=0))) <= 26.0


def _mask_boundary_ratio(mask: np.ndarray, box: list[int]) -> float:
    """Measure how much of a detector-box boundary is marked foreground."""
    height, width = mask.shape[:2]
    x1, y1, x2, y2 = [int(value) for value in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    region = mask[y1:y2, x1:x2].astype(bool)
    if region.size == 0:
        return 0.0
    thickness = max(1, min(region.shape) // 30)
    boundary = np.concatenate(
        (
            region[:thickness, :].reshape(-1),
            region[-thickness:, :].reshape(-1),
            region[:, :thickness].reshape(-1),
            region[:, -thickness:].reshape(-1),
        )
    )
    return float(boundary.mean()) if boundary.size else 0.0


def _looks_like_inverted_uploaded_mask(
    image: Image.Image,
    alpha: np.ndarray,
    box: list[int],
) -> bool:
    """Detect the common SAM2 failure where the white background is foreground.

    On a plain background the wrong mask usually fills the detector box and
    touches its boundary, while the real object is the transparent hole in the
    middle. Detecting this before TRELLIS is important: otherwise the transparent
    object is discarded and the white rectangular backdrop becomes geometry.
    """
    support = alpha >= 32
    extent = _mask_extent(support)
    if extent is None or not _uniform_border(image):
        return False

    x1, y1, x2, y2 = [int(value) for value in box]
    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)
    ex1, ey1, ex2, ey2 = extent
    extent_width = (ex2 - ex1) / box_width
    extent_height = (ey2 - ey1) / box_height
    density = _mask_density(support, box)
    boundary_ratio = _mask_boundary_ratio(support, box)
    area_ratio = float(support.mean())

    return (
        extent_width >= 0.82
        and extent_height >= 0.82
        and density >= 0.34
        and boundary_ratio >= 0.52
        and area_ratio >= 0.18
    )


def _looks_like_connected_background_leak(
    image: Image.Image,
    alpha: np.ndarray,
    box: list[int],
) -> bool:
    """Detect a plausible SAM mask that still contains a large studio panel.

    This is label-agnostic and is used for generated as well as uploaded
    assets. The inverted-mask check catches the obvious case where SAM marks
    nearly the whole detector box as foreground. A subtler failure is a mask
    with a connected floor/card/background shape attached to its lower edge.
    That mask passes a density check but TRELLIS turns it into a flat mesh.
    """
    if not _uniform_border(image):
        return False

    height, width = alpha.shape[:2]
    reference = _border_background_alpha(image, [0, 0, width, height])
    if reference is None:
        return False

    candidate_support = alpha >= 32
    reference_support = reference >= 32
    candidate_area = int(candidate_support.sum())
    reference_area = int(reference_support.sum())
    if candidate_area == 0 or reference_area < max(256, int(alpha.size * 0.025)):
        return False

    candidate_extent = _mask_extent(candidate_support)
    reference_extent = _mask_extent(reference_support)
    if candidate_extent is None or reference_extent is None:
        return False

    x1, y1, x2, y2 = [int(value) for value in box]
    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)
    cx1, cy1, cx2, cy2 = candidate_extent
    rx1, ry1, rx2, ry2 = reference_extent
    candidate_width = max(1, cx2 - cx1)
    candidate_height = max(1, cy2 - cy1)
    reference_width = rx2 - rx1
    reference_height = ry2 - ry1

    # The independent mask must cover most of the detected object's extent,
    # while the SAM mask must contain substantially more area than it does.
    # This is the signature of an object plus a connected background panel.
    reference_covers_object = (
        reference_width >= 0.50 * candidate_width
        and reference_height >= 0.50 * candidate_height
    )
    area_excess = candidate_area / max(1, reference_area)
    if not reference_covers_object or area_excess < 1.35:
        return False

    dilated_reference = ndimage.binary_dilation(
        reference_support,
        structure=np.ones((9, 9), dtype=bool),
    )
    excess = np.logical_and(candidate_support, np.logical_not(dilated_reference))
    excess_ratio = float(excess.sum() / max(1, candidate_area))
    candidate_density = _mask_density(candidate_support, box)
    boundary_ratio = _mask_boundary_ratio(candidate_support, box)
    fills_box = (
        candidate_width >= 0.78 * box_width
        and candidate_height >= 0.78 * box_height
    )
    return (
        fills_box
        and candidate_density >= 0.18
        and boundary_ratio >= 0.30
        and excess_ratio >= 0.30
    )


def _uploaded_alpha(
    image: Image.Image,
    box: list[int],
    label: str,
) -> tuple[np.ndarray, float, str]:
    """Get a foreground alpha for an uploaded image without forwarding a bad mask."""
    width, height = image.size

    # If the user supplied a transparent PNG, it is already a stronger
    # foreground signal than a second DINO/SAM prediction.  Reuse it directly
    # after the same cleanup used for SAM masks; this also avoids turning a
    # valid transparent upload into a white-background detection problem.
    if "A" in image.getbands():
        supplied_alpha = np.asarray(image.getchannel("A"), dtype=np.uint8)
        if _is_reusable_alpha(supplied_alpha):
            cleaned = _clean_alpha(supplied_alpha, [0, 0, width, height])
            if _is_reusable_alpha(cleaned) and not _looks_like_rectangular_backdrop(
                cleaned,
                [0, 0, width, height],
            ):
                return cleaned, 1.0, "uploaded-alpha"

    prediction_error = None
    try:
        alpha, _best_mask, mask_score, method = _predict_refined_alpha(
            image,
            box,
            label=label,
        )
        inverted_mask = _looks_like_inverted_uploaded_mask(image, alpha, box)
        leaked_background = _looks_like_connected_background_leak(image, alpha, box)
        if not inverted_mask and not leaked_background:
            return alpha, float(mask_score), method
        prediction_error = (
            "SAM2 selected the plain background as foreground"
            if inverted_mask
            else "SAM2 mask still contains a large connected background panel"
        )
    except Exception as exc:
        prediction_error = str(exc)

    # Uploaded product photos commonly use a white/grey background. Estimating
    # that colour from the border is safer than trusting an inverted SAM mask.
    full_box = [0, 0, width, height]
    fallback_alpha = _border_background_alpha(image, full_box)
    if fallback_alpha is not None:
        return fallback_alpha, 0.0, "border-background-fallback"

    # For non-uniform backgrounds, use the optional matting model as the next
    # recovery path. If neither method can produce a reliable alpha, fail here
    # with a clear segmentation error instead of generating a black backdrop.
    fallback_alpha = _full_image_matting_alpha(image)
    if fallback_alpha is not None:
        return fallback_alpha, 0.0, f"{MATTING_MODEL}-fallback"

    detail = prediction_error or "no usable foreground mask"
    raise ValueError(
        f"Could not isolate uploaded '{label}': {detail}. "
        "Use a clear physical-object photo with visible contrast from its background."
    )


def _save_trellis_rgba_crop(
    image: Image.Image,
    alpha: np.ndarray,
    crops_dir: str,
    name: str,
) -> tuple[str, list[int], Image.Image]:
    """Save a tight transparent crop with a safety margin for TRELLIS.

    This is shared by text-generated and uploaded inputs.  TRELLIS respects an
    existing alpha mask but does not re-crop such an image, so forwarding a
    full 1024px transparent canvas makes a small object look like background
    geometry.  Every valid object must therefore be tightened here.
    """
    support = alpha >= 32
    extent = _mask_extent(support)
    if extent is None:
        raise ValueError("Uploaded segmentation returned an empty foreground mask")

    x1, y1, x2, y2 = extent
    content = image.convert("RGBA")
    content.putalpha(Image.fromarray(alpha))
    content = content.crop((x1, y1, x2, y2))
    # Keep transparent breathing room so a dense object cannot look like an
    # opaque full-frame panel to TRELLIS' input validator.
    padding = max(12, int(max(content.size) * 0.08))
    cropped = Image.new(
        "RGBA",
        (content.width + padding * 2, content.height + padding * 2),
        (255, 255, 255, 0),
    )
    cropped.alpha_composite(content, (padding, padding))
    crop_path = os.path.join(crops_dir, f"{name}.png")
    cropped.save(crop_path)
    return crop_path, [x1, y1, x2, y2], cropped


def _is_reusable_alpha(alpha: np.ndarray) -> bool:
    """Return whether an existing alpha channel contains a usable object.

    Object-wise generation already produces a SAM2 alpha mask during the
    sanitization step.  Re-running GroundingDINO on the flattened white preview
    is unsafe for white furniture, so only reuse an alpha channel when it is
    clearly neither empty nor an almost-opaque full canvas.
    """
    if alpha.ndim != 2 or not alpha.size:
        return False
    support = alpha >= 32
    extent = _mask_extent(support)
    if extent is None:
        return False
    area_ratio = float(support.mean())
    return 0.001 <= area_ratio <= 0.92


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


def _full_image_matting_alpha(image: Image.Image) -> np.ndarray | None:
    """Use matting on the whole generated object image as a DINO fallback."""
    width, height = image.size
    return _matting_alpha(image, [0, 0, width, height])


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
        "lamp": (0.02, 48),
        "floor lamp": (0.02, 48),
        "arc floor lamp": (0.02, 48),
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

    # If SAM2 chose a dense rectangle, it likely selected the studio backdrop
    # or a generated product card instead of the requested object.  Use the
    # background colour estimated at the image border for every object type.
    if _looks_like_rectangular_backdrop(alpha, sam_box_values):
        contrast_alpha = _border_background_alpha(image, sam_box_values)
        if contrast_alpha is not None:
            alpha = contrast_alpha
            method = f"{method}+border-background-refine"
        else:
            raise ValueError(
                "SAM2 isolated an opaque rectangular backdrop instead of the requested object"
            )

    # A backdrop does not have to be a perfect rectangle. Generated images can
    # contain a connected blue/white card or floor shape that DINO labels as
    # the requested furniture. Compare it with an independent border-colour
    # mask before it is sent to TRELLIS.
    if _looks_like_connected_background_leak(image, alpha, sam_box_values):
        contrast_alpha = _border_background_alpha(image, sam_box_values)
        if contrast_alpha is None or _looks_like_connected_background_leak(
            image,
            contrast_alpha,
            sam_box_values,
        ):
            raise ValueError(
                f"SAM2 isolated a connected background/card instead of '{label or 'the object'}'"
            )
        alpha = contrast_alpha
        method = f"{method}+border-background-leak-refine"

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
    if _looks_like_rectangular_backdrop(alpha, sam_box_values):
        raise ValueError(
            "SAM2 isolated an opaque rectangular backdrop instead of the requested object"
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
        "arc floor lamp": "floor lamp",
        "standing lamp": "floor lamp",
    }.get(normalized_label, normalized_label)
    label_thresholds = {
        "chair": (0.18, 0.14),
        "couch": (0.18, 0.14),
        "dining table": (0.20, 0.15),
        "lamp": (0.16, 0.13),
        "floor lamp": (0.16, 0.13),
        "arc floor lamp": (0.16, 0.13),
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
        width, height = source_image.size
        detector_fallback = False
        quality_warnings = []
        if candidates:
            selected = candidates[0]
            selected_box = selected["box"]
            confidence = selected["confidence"]
            try:
                semantic_error = _generated_asset_semantic_error(
                    image_path,
                    label,
                    confidence,
                )
                if semantic_error:
                    quality_warnings.append(semantic_error)
            except Exception as semantic_exc:
                # This query is a quality hint, not part of the core DINO+SAM
                # isolation path. A weak/conflicting auxiliary prediction must
                # never turn an otherwise valid pipeline run into HTTP 500.
                job["semantic_check_error"] = type(semantic_exc).__name__
            try:
                alpha, best_mask, mask_score, segmentation_method = _predict_refined_alpha(
                    source_image,
                    selected_box,
                    label=label,
                )
            except Exception as sam_exc:
                # A white object can produce a weak/partial SAM2 mask even
                # when DINO found its box.  Since generated object images are
                # expected to contain one object, matting the whole image is a
                # safer fallback than forwarding a broken mask to TRELLIS.
                fallback_alpha = _border_background_alpha(source_image, selected_box)
                if fallback_alpha is None:
                    fallback_alpha = _full_image_matting_alpha(source_image)
                if fallback_alpha is None:
                    raise sam_exc
                if _looks_like_rectangular_backdrop(
                    fallback_alpha,
                    selected_box,
                ):
                    raise ValueError(
                        "Could not isolate the requested object from its rectangular backdrop"
                    ) from sam_exc
                alpha = fallback_alpha
                selected_box = list(_mask_extent(alpha >= 32) or selected_box)
                confidence = selected["confidence"]
                mask_score = 0.0
                segmentation_method = f"{MATTING_MODEL}-fallback"
                detector_fallback = True
            job["detected_instances"] = len(candidates)
            job["count_validation"] = "detected"
        else:
            # GroundingDINO can miss a low-contrast white object on a white
            # background.  The generated image is object-wise and is expected
            # to contain exactly one object, so use the semantic matting model
            # on the full image before rejecting the job.
            alpha = _border_background_alpha(source_image, [0, 0, width, height])
            if alpha is None:
                alpha = _full_image_matting_alpha(source_image)
            if alpha is None:
                raise ValueError(
                    f"GroundingDINO and {MATTING_MODEL} could not isolate the generated '{label}'. "
                    "Use a contrasting background or regenerate the 2D object."
                )
            if _looks_like_rectangular_backdrop(
                alpha,
                [0, 0, width, height],
            ):
                raise ValueError(
                    "Could not isolate the requested object from its rectangular backdrop"
                )
            selected_box = list(_mask_extent(alpha >= 32) or [0, 0, width, height])
            confidence = 0.0
            mask_score = 0.0
            segmentation_method = f"{MATTING_MODEL}-fallback"
            detector_fallback = True
            job["detected_instances"] = 0
            job["count_validation"] = "matting_fallback"

        if _looks_like_floor_attachment(alpha):
            quality_warnings.append(
                "silhouette may contain a rug, floor patch, support sheet, "
                "or connected background"
            )
        shape_warning = _category_shape_warning(alpha, label)
        if shape_warning:
            quality_warnings.append(shape_warning)

        alpha_image = Image.fromarray(alpha)
        content_box = alpha_image.getbbox() or tuple(selected_box)

        rgba = source_image.convert("RGBA")
        rgba.putalpha(alpha_image)
        isolated = rgba.crop(content_box)
        max_object_size = (int(width * 0.82), int(height * 0.82))
        isolated.thumbnail(max_object_size, Image.Resampling.LANCZOS)

        # Keep the object on a transparent canvas.  Flattening this image to
        # RGB white here loses the SAM2 result and makes a white object
        # indistinguishable from its background when objectwise segmentation
        # runs again later in the pipeline.
        clean_image = Image.new("RGBA", (width, height), (255, 255, 255, 0))
        paste_x = (width - isolated.width) // 2
        paste_y = (height - isolated.height) // 2
        clean_image.alpha_composite(isolated, (paste_x, paste_y))
        clean_image.save(image_path)

        job.update(
            {
                "sanitized": True,
                "kept_box": selected_box,
                "kept_confidence": confidence,
                "mask_score": mask_score,
                "segmentation_method": segmentation_method,
                "alpha_preserved": True,
                "backdrop_checked": True,
                "detector_fallback": detector_fallback,
                "quality_warnings": quality_warnings,
                "retry_recommended": bool(quality_warnings),
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
        with Image.open(image_path) as opened_image:
            has_alpha = "A" in opened_image.getbands() or "transparency" in opened_image.info
            image = opened_image.convert("RGBA" if has_alpha else "RGB")
        width, height = image.size

        # Sanitization has already run GroundingDINO + SAM2.  Reuse that alpha
        # mask instead of asking GroundingDINO to find a white object on the
        # white canvas produced by the old flow.
        preserved_alpha = None
        if has_alpha:
            candidate_alpha = np.asarray(image.getchannel("A"), dtype=np.uint8)
            if _is_reusable_alpha(candidate_alpha):
                preserved_alpha = _clean_alpha(candidate_alpha, [0, 0, width, height])
                if not _is_reusable_alpha(preserved_alpha):
                    preserved_alpha = None

        if preserved_alpha is not None:
            alpha = preserved_alpha
            box_values = list(_mask_extent(alpha >= 32) or [0, 0, width, height])
            confidence = float(item.get("kept_confidence", item.get("confidence", 1.0)) or 0.0)
            mask_score = float(item.get("mask_score", 1.0) or 1.0)
            best = 0
            segmentation_method = f"{item.get('segmentation_method') or 'sam2'}+reused_alpha"
        else:
            # Backward-compatible fallback for old manifests that contain an
            # RGB image without a preserved alpha channel.
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

        name = str(item.get("name", "object"))
        crop_path, final_box, rgba = _save_trellis_rgba_crop(
            image,
            alpha,
            crops_dir,
            name,
        )
        previews.append(
            Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba)
            .convert("RGB")
        )
        results.append(
            {
                "name": name,
                "label": label,
                "confidence": confidence,
                "detected_instances": int(
                    item.get("detected_instances", 1 if preserved_alpha is not None else len(candidates))
                ),
                "mask_score": mask_score,
                "box": box_values,
                "final_box": final_box,
                "crop_path": crop_path,
                "crop_url": f"/crops/{name}.png",
                "source_mode": (
                    "objectwise_reused_alpha"
                    if preserved_alpha is not None
                    else "objectwise_sam2"
                ),
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
            jobs = []
            # Process jobs independently so the caller knows exactly which
            # object needs a clean-background retry.  The models stay resident
            # under the same inference lock; this is not a reload per object.
            for item in payload.objects:
                try:
                    jobs.extend(_sanitize_object_images([item]))
                except Exception as item_exc:
                    name = str(item.get("name", "object"))
                    label = str(item.get("label", "object"))
                    raise ValueError(
                        f"Object '{name}' ({label}) could not be isolated: {item_exc}"
                    ) from item_exc
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
    with Image.open(image_path) as opened_image:
        has_alpha = "A" in opened_image.getbands() or "transparency" in opened_image.info
        source_image = opened_image.convert("RGBA" if has_alpha else "RGB")
    # SAM2 expects an RGB array, while _uploaded_alpha may reuse a preserved
    # alpha channel from a transparent upload.
    sam_predictor.set_image(np.asarray(source_image.convert("RGB")))

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
            alpha, mask_score, segmentation_method = _uploaded_alpha(
                source_image,
                [x1, y1, x2, y2],
                label=str(candidate.get("label", "object")),
            )
            safe_label = re.sub(r"[^a-z0-9]+", "_", candidate["label"].lower()).strip("_")
            name = f"{safe_label or 'object'}_{index}"
            crop_path, final_box, rgba = _save_trellis_rgba_crop(
                source_image,
                alpha,
                crops_dir,
                name,
            )
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
                    "final_box": final_box,
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

        alpha, mask_score, segmentation_method = _uploaded_alpha(
            source_image,
            [x1, y1, x2, y2],
            label=label,
        )
        crop_path, final_box, rgba = _save_trellis_rgba_crop(
            source_image,
            alpha,
            crops_dir,
            name,
        )
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
            "final_box": final_box,
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
