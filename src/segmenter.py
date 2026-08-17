import json
import os
import re
from pathlib import Path

import requests
from PIL import Image, ImageOps


OBJECT_MANIFEST = "/kaggle/working/object_images_manifest.json"
SAM2_DINO_WORKER_URL = os.environ.get(
    "SAM2_DINO_WORKER_URL",
    "http://127.0.0.1:8003",
).rstrip("/")
SAM2_DINO_WORKER_TIMEOUT = float(
    os.environ.get("SAM2_DINO_WORKER_TIMEOUT", "360")
)


def _load_preserved_rgba(path: str):
    """Load a sanitized object image only when it has a usable alpha mask."""
    source = Path(path)
    if not source.is_file():
        return None
    try:
        with Image.open(source) as opened:
            has_alpha = "A" in opened.getbands() or "transparency" in opened.info
            if not has_alpha:
                return None
            image = opened.convert("RGBA")
    except Exception:
        return None

    alpha = image.getchannel("A")
    support = alpha.point(lambda value: 255 if value >= 32 else 0)
    if support.getbbox() is None:
        return None
    alpha_min, alpha_max = alpha.getextrema()
    if alpha_min >= 255 and support.getbbox() == (0, 0, image.width, image.height):
        return None
    return image


def _save_reused_preview(images: list[Image.Image], crops_dir: str) -> None:
    if not images:
        return
    cards = [
        ImageOps.contain(image, (360, 360), Image.Resampling.LANCZOS)
        for image in images
    ]
    sheet = Image.new(
        "RGB",
        (360 * min(2, len(cards)), 360 * ((len(cards) + 1) // 2)),
        "white",
    )
    for index, card in enumerate(cards):
        sheet.paste(card, ((index % 2) * 360, (index // 2) * 360))
    sheet.save(Path(crops_dir) / "sam2_visual.png")


def _reuse_sanitized_object_crops(
    objects: list[dict],
    crops_dir: str,
) -> list[dict] | None:
    """Reuse the alpha crops produced by the earlier sanitize step.

    The text pipeline already validates every generated object with
    GroundingDINO + SAM2 before writing the manifest.  Calling the detector a
    second time on a flattened/white preview makes white furniture ambiguous
    and can turn the background into a failed mask.  Return ``None`` when an
    old manifest has no alpha channel so the legacy worker fallback remains
    available.
    """
    reused = []
    previews = []
    target_dir = Path(crops_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    for item in objects:
        image = _load_preserved_rgba(str(item.get("image_path", "")))
        if image is None:
            return None

        name = str(item.get("name", "object")).strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", name):
            return None
        crop_path = target_dir / f"{name}.png"
        image.save(crop_path)
        previews.append(
            Image.alpha_composite(
                Image.new("RGBA", image.size, "white"),
                image,
            ).convert("RGB")
        )

        width, height = image.size
        result = dict(item)
        result.update(
            {
                "name": name,
                "label": item.get("label", name),
                "confidence": float(
                    item.get("kept_confidence", item.get("confidence", 0.0)) or 0.0
                ),
                "mask_score": float(item.get("mask_score", 1.0) or 1.0),
                "box": item.get("kept_box", [0, 0, width, height]),
                "final_box": [0, 0, width, height],
                "crop_path": str(crop_path),
                "crop_url": f"/crops/{name}.png",
                "source_mode": "objectwise_reused_alpha",
                "segmentation_method": (
                    f"{item.get('segmentation_method') or 'sam2'}+reused_alpha"
                ),
                "alpha_preserved": True,
                "detector_fallback": False,
            }
        )
        reused.append(result)

    _save_reused_preview(previews, crops_dir)
    return reused


def _check_worker_ready() -> None:
    try:
        response = requests.get(f"{SAM2_DINO_WORKER_URL}/health", timeout=5)
        response.raise_for_status()
        health = response.json()
    except requests.RequestException as exc:
        raise RuntimeError(
            "SAM2 + GroundingDINO worker is unavailable at "
            f"{SAM2_DINO_WORKER_URL}. Start worker_sam2_dino.py first."
        ) from exc

    if not health.get("ready"):
        detail = health.get("error") or "models are still loading"
        raise RuntimeError(f"SAM2 + GroundingDINO worker is not ready: {detail}")


def _worker_segment(
    input_image_path: str,
    objects: list[dict],
    crops_dir: str,
    source_mode: str,
    auto_detect: bool = False,
) -> list:
    _check_worker_ready()
    try:
        response = requests.post(
            f"{SAM2_DINO_WORKER_URL}/segment",
            json={
                "input_image_path": input_image_path,
                "objects": objects,
                "crops_dir": crops_dir,
                "source_mode": source_mode,
                "auto_detect": auto_detect,
            },
            timeout=SAM2_DINO_WORKER_TIMEOUT,
        )
        response.raise_for_status()
        results = response.json().get("results", [])
    except requests.RequestException as exc:
        detail = getattr(exc.response, "text", "") if exc.response else ""
        raise RuntimeError(
            f"SAM2 + GroundingDINO worker failed: {detail or exc}"
        ) from exc

    if not auto_detect and len(results) != len(objects):
        raise RuntimeError(
            f"Segmentation worker returned {len(results)} results "
            f"for {len(objects)} objects"
        )
    for result in results:
        crop_path = result.get("crop_path")
        if not crop_path or not os.path.isfile(crop_path):
            raise FileNotFoundError(crop_path or "Missing segmentation crop path")
    return results


def _run_uploaded_grounded_sam2(
    input_image_path: str,
    layout: dict,
    crops_dir: str,
    auto_detect: bool = False,
) -> list:
    """Detect uploaded-image objects with resident DINO, then mask with SAM2."""
    objects = list(layout.get("objects", []))
    if not objects:
        raise ValueError("Upload segmentation needs at least one labeled object")
    return _worker_segment(
        input_image_path,
        objects,
        crops_dir,
        source_mode="uploaded",
        auto_detect=auto_detect,
    )


def run_grounded_sam2(
    input_image_path: str,
    layout: dict,
    crops_dir: str,
    source_mode: str = "objectwise",
    auto_detect: bool = False,
    manifest_path: str = OBJECT_MANIFEST,
) -> list:
    """Segment generated object images or an uploaded scene via the worker."""
    if source_mode == "uploaded":
        return _run_uploaded_grounded_sam2(
            input_image_path,
            layout,
            crops_dir,
            auto_detect=auto_detect,
        )

    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(
            "Object images are missing. Run the SD3.5 object-wise stage first."
        )

    Path(crops_dir).mkdir(parents=True, exist_ok=True)
    with open(manifest_path, encoding="utf-8") as file:
        objects = json.load(file)
    if not objects:
        raise ValueError("The object image manifest is empty")

    # The objectwise SD3.5 stage has already run GroundingDINO + SAM2 during
    # sanitization.  Reuse those transparent crops and avoid a second detector
    # pass that is especially fragile for white objects on white backgrounds.
    reused_crops = _reuse_sanitized_object_crops(objects, crops_dir)
    if reused_crops is not None:
        return reused_crops

    return _worker_segment(
        input_image_path,
        objects,
        crops_dir,
        source_mode="objectwise",
    )
