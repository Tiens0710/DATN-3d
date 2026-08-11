import os
import re
from pathlib import Path

import requests


TRELLIS_WORKER_URL = os.environ.get(
    "TRELLIS_WORKER_URL",
    "http://127.0.0.1:8002",
).rstrip("/")
TRELLIS_WORKER_TIMEOUT = float(
    os.environ.get("TRELLIS_WORKER_TIMEOUT", "1200")
)


def _response_error_detail(response) -> str:
    if response is None:
        return ""
    try:
        payload = response.json()
        detail = payload.get("detail") if isinstance(payload, dict) else None
        if detail:
            return str(detail)
    except ValueError:
        pass
    return response.text.strip()


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


def _path_within(path: str, root: str) -> Path:
    resolved = Path(path).resolve()
    resolved_root = Path(root).resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ValueError(f"Path is outside the active run: {path}")
    return resolved


def _mesh_thickness_ratio(model_path: str) -> float | None:
    """Return the thinnest-to-longest extent ratio for a generated GLB."""
    try:
        import numpy as np
        import trimesh

        loaded = trimesh.load(model_path, force="scene", process=False)
        extents = np.asarray(loaded.bounds[1] - loaded.bounds[0], dtype=float)
        longest = float(extents.max())
        return float(extents.min() / longest) if longest > 0 else 0.0
    except Exception:
        return None


def generate_3d_models(
    crops: list,
    multi_glb_dir: str,
    allowed_root: str | None = None,
    public_prefix: str = "/multi_object_glb",
) -> list:
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
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", name):
            raise ValueError(f"Invalid crop name: {name}")
        if allowed_root:
            crop_path = str(_path_within(crop_path, allowed_root))

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
            detail = _response_error_detail(exc.response)
            raise RuntimeError(
                f"TRELLIS worker failed for {name}: {detail or exc}"
            ) from exc

        model_path = result.get("model_path")
        if not model_path or not Path(model_path).is_file():
            raise FileNotFoundError(
                f"TRELLIS worker returned a missing model for {name}: {model_path}"
            )
        thickness_ratio = _mesh_thickness_ratio(model_path)
        if thickness_ratio is not None and thickness_ratio < 0.015:
            raise RuntimeError(
                f"TRELLIS produced an almost-flat mesh for {name} "
                f"(thickness ratio {thickness_ratio:.4f}). Use a complete physical-object photo "
                "with visible depth instead of flat artwork or a severe crop."
            )

        models.append(
            {
                "name": name,
                "label": crop.get("label", name),
                "model_url": f"{public_prefix.rstrip('/')}/{name}.glb",
                "model_path": model_path,
                "final_box": crop.get("final_box", [0, 0, 0, 0]),
                "thickness_ratio": thickness_ratio,
            }
        )

    return models
