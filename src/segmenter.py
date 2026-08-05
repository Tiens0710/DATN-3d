import json
import os
from pathlib import Path

import requests


OBJECT_MANIFEST = "/kaggle/working/object_images_manifest.json"
SAM2_DINO_WORKER_URL = os.environ.get(
    "SAM2_DINO_WORKER_URL",
    "http://127.0.0.1:8003",
).rstrip("/")
SAM2_DINO_WORKER_TIMEOUT = float(
    os.environ.get("SAM2_DINO_WORKER_TIMEOUT", "360")
)


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

    if len(results) != len(objects):
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
    )


def run_grounded_sam2(
    input_image_path: str,
    layout: dict,
    crops_dir: str,
    source_mode: str = "objectwise",
) -> list:
    """Segment generated object images or an uploaded scene via the worker."""
    if source_mode == "uploaded":
        return _run_uploaded_grounded_sam2(
            input_image_path,
            layout,
            crops_dir,
        )

    if not os.path.isfile(OBJECT_MANIFEST):
        raise FileNotFoundError(
            "Object images are missing. Run the SD3.5 object-wise stage first."
        )

    Path(crops_dir).mkdir(parents=True, exist_ok=True)
    with open(OBJECT_MANIFEST, encoding="utf-8") as file:
        objects = json.load(file)
    if not objects:
        raise ValueError("The object image manifest is empty")

    return _worker_segment(
        input_image_path,
        objects,
        crops_dir,
        source_mode="objectwise",
    )
