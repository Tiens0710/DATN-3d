import os
from pathlib import Path

import requests


TRELLIS_WORKER_URL = os.environ.get(
    "TRELLIS_WORKER_URL",
    "http://127.0.0.1:8002",
).rstrip("/")
TRELLIS_WORKER_TIMEOUT = float(
    os.environ.get("TRELLIS_WORKER_TIMEOUT", "1200")
)


def _check_worker_ready() -> None:
    try:
        response = requests.get(f"{TRELLIS_WORKER_URL}/health", timeout=5)
        response.raise_for_status()
        health = response.json()
    except requests.RequestException as exc:
        raise RuntimeError(
            "TRELLIS worker is unavailable at "
            f"{TRELLIS_WORKER_URL}. Start worker_trellis.py first."
        ) from exc

    if not health.get("ready"):
        detail = health.get("error") or "model is still loading"
        raise RuntimeError(f"TRELLIS worker is not ready: {detail}")


def generate_3d_models(crops: list, multi_glb_dir: str) -> list:
    """Generate one GLB per transparent SAM2 crop through the resident worker."""
    if not crops:
        raise ValueError("No SAM2 crops were provided to TRELLIS")

    os.makedirs(multi_glb_dir, exist_ok=True)
    _check_worker_ready()

    models = []
    for crop in crops:
        name = str(crop.get("name", "")).strip()
        crop_path = str(crop.get("crop_path", "")).strip()
        if not name or not crop_path:
            raise ValueError("Each crop must include name and crop_path")

        try:
            response = requests.post(
                f"{TRELLIS_WORKER_URL}/generate",
                json={
                    "crop_path": crop_path,
                    "name": name,
                    "output_dir": multi_glb_dir,
                },
                timeout=TRELLIS_WORKER_TIMEOUT,
            )
            response.raise_for_status()
            result = response.json()
        except requests.RequestException as exc:
            detail = getattr(exc.response, "text", "") if exc.response else ""
            raise RuntimeError(
                f"TRELLIS worker failed for {name}: {detail or exc}"
            ) from exc

        model_path = result.get("model_path")
        if not model_path or not Path(model_path).is_file():
            raise FileNotFoundError(
                f"TRELLIS worker returned a missing model for {name}: {model_path}"
            )

        models.append(
            {
                "name": name,
                "label": crop["label"],
                "model_url": f"/multi_object_glb/{name}.glb",
                "model_path": model_path,
                "final_box": crop["final_box"],
            }
        )

    return models
