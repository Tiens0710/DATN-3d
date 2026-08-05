"""Resident TRELLIS image-to-3D worker for the Kaggle GPU session."""

import argparse
import os
import sys
import threading
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


class GenerateRequest(BaseModel):
    crop_path: str
    name: str
    output_dir: str


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

        print("Loading TRELLIS-image-large...", flush=True)
        loaded_pipeline = TrellisImageTo3DPipeline.from_pretrained(
            "JeffreyXiang/TRELLIS-image-large"
        )
        loaded_pipeline.to("cuda")
        pipeline = loaded_pipeline
        postprocessing_utils = loaded_postprocessing
        load_error = None
        print("TRELLIS worker ready.", flush=True)
    except Exception as exc:
        load_error = f"{exc}\n{traceback.format_exc()}"
        print(f"TRELLIS worker failed to load:\n{load_error}", flush=True)


@app.on_event("startup")
def startup_event() -> None:
    _load_pipeline()


@app.get("/health")
def health() -> dict:
    return {"ready": pipeline is not None, "error": load_error}


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
        try:
            print(f"Generating TRELLIS model for {name}...", flush=True)
            image = Image.open(crop_path).convert("RGB")
            image = pipeline.preprocess_image(image)
            outputs = pipeline.run(
                image,
                seed=42,
                formats=["gaussian", "mesh"],
                preprocess_image=False,
                sparse_structure_sampler_params={
                    "steps": 12,
                    "cfg_strength": 7.5,
                },
                slat_sampler_params={
                    "steps": 12,
                    "cfg_strength": 3.0,
                },
            )

            glb = postprocessing_utils.to_glb(
                outputs["gaussian"][0],
                outputs["mesh"][0],
                simplify=0.95,
                texture_size=1024,
                verbose=False,
            )
            glb.export(str(output_path))
            if not output_path.is_file():
                raise RuntimeError("GLB export finished but no file was written")
            print(f"Saved: {output_path}", flush=True)
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"TRELLIS generation failed for {name}: {exc}",
            ) from exc
        finally:
            torch.cuda.empty_cache()

    return {"model_path": str(output_path)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resident TRELLIS worker")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8002)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
