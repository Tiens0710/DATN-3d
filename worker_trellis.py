"""Resident TRELLIS image-to-3D worker for the Kaggle GPU session."""

import argparse
import os
import sys
import threading
import time
import traceback
from pathlib import Path


# Keep the original environment workarounds before importing torch/TRELLIS.
os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("SPARSE_ATTN", "xformers")
os.environ.setdefault("MPLBACKEND", "agg")
sys.modules["triton"] = None

TRELLIS_ROOT = "/kaggle/working/TRELLIS"
sys.path.insert(0, "/opt/venv310/lib/python3.10/site-packages")
sys.path.insert(0, TRELLIS_ROOT)

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel


app = FastAPI(title="DATN TRELLIS Worker", version="1.1.0")
pipeline = None
postprocessing_utils = None
load_error = None
inference_lock = threading.Lock()
TRELLIS_SPARSE_STEPS = int(os.environ.get("TRELLIS_SPARSE_STEPS", "16"))
TRELLIS_SLAT_STEPS = int(os.environ.get("TRELLIS_SLAT_STEPS", "16"))
TRELLIS_TEXTURE_SIZE = int(os.environ.get("TRELLIS_TEXTURE_SIZE", "1024"))
TRELLIS_SIMPLIFY = float(os.environ.get("TRELLIS_SIMPLIFY", "0.98"))
TRELLIS_TEXTURE_MODE = os.environ.get("TRELLIS_TEXTURE_MODE", "opt").lower()
TRELLIS_TEXTURE_VIEWS = int(os.environ.get("TRELLIS_TEXTURE_VIEWS", "64"))
TRELLIS_BAKE_RESOLUTION = int(
    os.environ.get("TRELLIS_BAKE_RESOLUTION", "768")
)
TRELLIS_FILL_HOLES = os.environ.get("TRELLIS_FILL_HOLES", "0").lower() in {
    "1",
    "true",
    "yes",
    "on",
}


class GenerateRequest(BaseModel):
    crop_path: str
    name: str
    output_dir: str


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


def _log(message: str) -> None:
    print(f"[TRELLIS] {message}", flush=True)


def _safe_empty_cache(name: str) -> None:
    try:
        torch.cuda.empty_cache()
    except Exception as exc:
        _log(
            f"{name}: CUDA cleanup failed but original result/error is preserved: "
            f"{type(exc).__name__}: {exc}"
        )


def _load_pipeline() -> None:
    """Load once and keep failures observable through /health."""
    global pipeline, postprocessing_utils, load_error

    try:
        if not os.path.isdir(os.path.join(TRELLIS_ROOT, "trellis")):
            raise RuntimeError(
                "TRELLIS source is missing. Run the notebook setup cells first."
            )

        import nvdiffrast.torch  # noqa: F401
        import spconv  # noqa: F401
        import utils3d  # noqa: F401
        import xformers  # noqa: F401
        from diff_gaussian_rasterization import _C  # noqa: F401
        from trellis.pipelines import TrellisImageTo3DPipeline
        from trellis.utils import postprocessing_utils as loaded_postprocessing

        if TRELLIS_TEXTURE_MODE not in {"fast", "opt"}:
            raise ValueError(
                "TRELLIS_TEXTURE_MODE must be either 'fast' or 'opt'"
            )

        # TRELLIS' stock GLB exporter always requests the expensive texture
        # path (2,500 optimization iterations, 100 views at 1024px).  Keep the
        # official exporter but configure its module-level helpers so the
        # resident worker can use the much cheaper built-in fast baker.
        original_bake_texture = loaded_postprocessing.bake_texture
        original_render_multiview = loaded_postprocessing.render_multiview

        def configured_bake_texture(*args, **kwargs):
            kwargs["mode"] = TRELLIS_TEXTURE_MODE
            return original_bake_texture(*args, **kwargs)

        def configured_render_multiview(*args, **kwargs):
            kwargs["resolution"] = min(
                int(kwargs.get("resolution", TRELLIS_BAKE_RESOLUTION)),
                TRELLIS_BAKE_RESOLUTION,
            )
            kwargs["nviews"] = min(
                int(kwargs.get("nviews", TRELLIS_TEXTURE_VIEWS)),
                TRELLIS_TEXTURE_VIEWS,
            )
            return original_render_multiview(*args, **kwargs)

        loaded_postprocessing.bake_texture = configured_bake_texture
        loaded_postprocessing.render_multiview = configured_render_multiview

        _log("Loading TRELLIS-image-large...")
        loaded_pipeline = TrellisImageTo3DPipeline.from_pretrained(
            "JeffreyXiang/TRELLIS-image-large"
        )
        loaded_pipeline.to("cuda")
        pipeline = loaded_pipeline
        postprocessing_utils = loaded_postprocessing
        load_error = None
        _log(
            "Worker ready; "
            f"steps={TRELLIS_SPARSE_STEPS}+{TRELLIS_SLAT_STEPS}, "
            f"texture={TRELLIS_TEXTURE_SIZE}px/{TRELLIS_TEXTURE_MODE}, "
            f"views={TRELLIS_TEXTURE_VIEWS}@{TRELLIS_BAKE_RESOLUTION}px, "
            f"fill_holes={TRELLIS_FILL_HOLES}; {_cuda_memory()}"
        )
    except Exception as exc:
        load_error = f"{exc}\n{traceback.format_exc()}"
        _log(f"Worker failed to load:\n{load_error}")


@app.on_event("startup")
def startup_event() -> None:
    _load_pipeline()


@app.get("/health")
def health() -> dict:
    return {
        "ready": pipeline is not None,
        "error": load_error,
        "profile": {
            "sparse_steps": TRELLIS_SPARSE_STEPS,
            "slat_steps": TRELLIS_SLAT_STEPS,
            "texture_size": TRELLIS_TEXTURE_SIZE,
            "texture_mode": TRELLIS_TEXTURE_MODE,
            "texture_views": TRELLIS_TEXTURE_VIEWS,
            "bake_resolution": TRELLIS_BAKE_RESOLUTION,
            "fill_holes": TRELLIS_FILL_HOLES,
        },
    }


@app.post("/generate")
def generate(payload: GenerateRequest) -> dict:
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail=load_error or "TRELLIS model is still loading",
        )

    crop_path = Path(payload.crop_path)
    if not crop_path.is_file():
        raise HTTPException(status_code=404, detail=f"Crop not found: {crop_path}")

    name = Path(payload.name).name
    if not name or name in {".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid model name")
    output_dir = Path(payload.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{name}.glb"

    with inference_lock:
        started_at = time.perf_counter()
        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            _log(f"Request {name}: crop={crop_path}; {_cuda_memory()}")

            image = Image.open(crop_path).convert("RGB")
            _log(f"{name}: preprocessing image ({image.width}x{image.height})")
            image = pipeline.preprocess_image(image)

            _log(f"{name}: running sparse structure + SLAT diffusion")
            with torch.inference_mode():
                outputs = pipeline.run(
                    image,
                    seed=42,
                    formats=["gaussian", "mesh"],
                    preprocess_image=False,
                    sparse_structure_sampler_params={
                        "steps": TRELLIS_SPARSE_STEPS,
                        "cfg_strength": 7.5,
                    },
                    slat_sampler_params={
                        "steps": TRELLIS_SLAT_STEPS,
                        "cfg_strength": 3.0,
                    },
                )
            torch.cuda.synchronize()
            _log(
                f"{name}: diffusion completed in "
                f"{time.perf_counter() - started_at:.1f}s; {_cuda_memory()}"
            )

            _log(f"{name}: converting Gaussian + mesh output to textured GLB")
            with torch.inference_mode():
                glb = postprocessing_utils.to_glb(
                    outputs["gaussian"][0],
                    outputs["mesh"][0],
                    simplify=TRELLIS_SIMPLIFY,
                    fill_holes=TRELLIS_FILL_HOLES,
                    texture_size=TRELLIS_TEXTURE_SIZE,
                    verbose=False,
                )
            glb.export(str(output_path))
            if not output_path.is_file():
                raise RuntimeError("GLB export finished but no file was written")
            torch.cuda.synchronize()
            peak_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)
            _log(
                f"{name}: saved {output_path} in "
                f"{time.perf_counter() - started_at:.1f}s; "
                f"peak_allocated={peak_gib:.2f} GiB"
            )
        except Exception as exc:
            error_trace = traceback.format_exc()
            _log(
                f"{name}: FAILED after {time.perf_counter() - started_at:.1f}s; "
                f"{_cuda_memory()}\n{error_trace}"
            )
            detail = f"{type(exc).__name__}: {exc}"
            if "out of memory" in str(exc).lower():
                detail += (
                    ". GPU out of memory while all resident workers are loaded; "
                    "inspect trellis_worker.log and nvidia-smi in section 11."
                )
            raise HTTPException(
                status_code=500,
                detail=f"TRELLIS generation failed for {name}: {detail}",
            ) from exc
        finally:
            _safe_empty_cache(name)
            _log(f"{name}: request cleanup; {_cuda_memory()}")

    return {"model_path": str(output_path)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resident TRELLIS worker")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8002)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
