"""Resident TRELLIS image-to-3D worker.

The worker is started once per Kaggle session and keeps the TRELLIS pipeline
on the GPU.  The main FastAPI process talks to it over localhost so a request
does not have to create a new Python process or reload the model.
"""

import argparse
import os
import sys
import threading
from pathlib import Path


TRELLIS_ROOT = "/kaggle/working/TRELLIS"

# These must be set before importing the CUDA-heavy libraries.
os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("SPARSE_ATTN", "xformers")
os.environ.setdefault("MPLBACKEND", "agg")

sys.path.insert(0, "/opt/venv310/lib/python3.10/site-packages")
sys.path.insert(0, TRELLIS_ROOT)
sys.modules["triton"] = None

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import postprocessing_utils
from PIL import Image


app = FastAPI(title="DATN TRELLIS Worker", version="1.0.0")
pipeline = None
load_error = None
inference_lock = threading.Lock()


class GenerateRequest(BaseModel):
    crop_path: str
    name: str
    output_dir: str


@app.on_event("startup")
def load_model() -> None:
    """Load TRELLIS exactly once when this worker process starts."""
    global pipeline, load_error

    try:
        pipeline = TrellisImageTo3DPipeline.from_pretrained(
            "JeffreyXiang/TRELLIS-image-large"
        )
        pipeline.to("cuda")
    except Exception as exc:
        load_error = str(exc)
        raise


@app.get("/health")
def health() -> dict:
    return {
        "ready": pipeline is not None,
        "error": load_error,
        "cuda": torch.cuda.is_available(),
    }


@app.post("/generate")
def generate(request: GenerateRequest) -> dict:
    if pipeline is None:
        raise HTTPException(status_code=503, detail=load_error or "Worker is not ready")

    crop_path = Path(request.crop_path)
    if not crop_path.is_file():
        raise HTTPException(status_code=404, detail=f"Crop not found: {crop_path}")

    output_dir = Path(request.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    name = Path(request.name).name
    if not name or name in {".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid model name")

    output_path = output_dir / f"{name}.glb"

    # The main API already queues jobs, but keep the worker safe if it is
    # called directly by more than one client.
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
            torch.cuda.empty_cache()
            print(f"Saved: {output_path}", flush=True)
        except Exception as exc:
            torch.cuda.empty_cache()
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"model_path": str(output_path)}


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="Resident TRELLIS worker")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8002)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
