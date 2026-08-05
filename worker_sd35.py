"""Resident Stable Diffusion 3.5 worker for per-object image generation."""

import argparse
import gc
import json
import os
import shutil
import sys
import threading
import traceback


os.environ.setdefault("MPLBACKEND", "agg")
sys.path.insert(0, "/opt/venv310/lib/python3.10/site-packages")
sys.path.insert(0, "/kaggle/working")
sys.modules["triton"] = None

import diffusers
import torch
import uvicorn
from diffusers import StableDiffusion3Pipeline
from fastapi import FastAPI, HTTPException
from packaging.version import Version
from pydantic import BaseModel

from src.generator_2d import OBJECT_MANIFEST, build_object_jobs


SD35_CACHE_DIR = "/kaggle/working/sd35_medium_cache_v1"
SD35_NUM_INFERENCE_STEPS = int(os.environ.get("SD35_NUM_INFERENCE_STEPS", "24"))
SD35_IMAGE_SIZE = int(os.environ.get("SD35_IMAGE_SIZE", "704"))
SD35_GUIDANCE_SCALE = float(os.environ.get("SD35_GUIDANCE_SCALE", "4.5"))

app = FastAPI(title="DATN SD3.5 Worker", version="1.0.0")
pipeline = None
pipeline_device = None
load_error = None
inference_lock = threading.Lock()


class GenerateRequest(BaseModel):
    scene_prompt: str
    objects: list[dict]
    lora_scale: float = 0.0


def _create_pipeline(*, force_download: bool = False):
    if Version(diffusers.__version__) < Version("0.32.0"):
        raise RuntimeError(
            "SD3.5 Medium requires Diffusers >= 0.32.0. Found "
            f"{diffusers.__version__}. Run the notebook dependency cell first."
        )

    hf_token = os.environ.get("HF_TOKEN", "").strip()
    if not hf_token:
        raise RuntimeError("HF_TOKEN is not configured")

    return StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-medium",
        text_encoder_3=None,
        tokenizer_3=None,
        torch_dtype=torch.float16,
        token=hf_token,
        cache_dir=SD35_CACHE_DIR,
        force_download=force_download,
    ).to("cuda")


def _safe_cuda_cleanup() -> None:
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception as exc:  # CUDA context may already be in an error state.
        print(f"CUDA cache cleanup skipped: {exc}", flush=True)


def _move_pipeline(device: str) -> None:
    global pipeline_device

    if pipeline is None:
        raise RuntimeError("SD3.5 pipeline is not loaded")
    if pipeline_device == device:
        return

    print(f"Moving SD3.5 pipeline to {device}...", flush=True)
    pipeline.to(device)
    pipeline_device = device
    if device == "cpu":
        _safe_cuda_cleanup()
    print(f"SD3.5 pipeline is now on {device}.", flush=True)


def _load_pipeline() -> None:
    global pipeline, pipeline_device, load_error

    try:
        print(f"Diffusers version: {diffusers.__version__}", flush=True)
        try:
            loaded = _create_pipeline()
        except Exception as exc:
            message = str(exc)
            cache_mismatch = (
                "expected shape tensor" in message or "size mismatch" in message
            )
            if not cache_mismatch:
                raise
            print("Repairing the dedicated SD3.5 cache and retrying...", flush=True)
            shutil.rmtree(SD35_CACHE_DIR, ignore_errors=True)
            loaded = _create_pipeline(force_download=True)

        pipeline = loaded
        pipeline_device = "cuda"
        load_error = None
        print("SD3.5 worker ready.", flush=True)
    except Exception as exc:
        load_error = f"{exc}\n{traceback.format_exc()}"
        print(f"SD3.5 worker failed to load:\n{load_error}", flush=True)


@app.on_event("startup")
def startup_event() -> None:
    _load_pipeline()


@app.get("/health")
def health() -> dict:
    return {
        "ready": pipeline is not None,
        "error": load_error,
        "device": pipeline_device,
    }


@app.post("/offload")
def offload() -> dict:
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail=load_error or "SD3.5 model is still loading",
        )

    with inference_lock:
        try:
            _move_pipeline("cpu")
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to offload SD3.5 pipeline: {exc}",
            ) from exc
    return {"ready": True, "device": pipeline_device}


@app.post("/generate")
def generate(payload: GenerateRequest) -> dict:
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail=load_error or "SD3.5 model is still loading",
        )
    if not payload.objects:
        raise HTTPException(status_code=400, detail="No objects were provided")

    # The current pipeline accepts lora_scale for API compatibility, but the
    # SD3.5 implementation does not load a LoRA adapter yet.
    _ = payload.lora_scale
    jobs = build_object_jobs(payload.scene_prompt, payload.objects)

    with inference_lock:
        try:
            _move_pipeline("cuda")
            with torch.inference_mode():
                for job in jobs:
                    print(
                        f"Generating isolated object: {job['name']} {job['label']}",
                        flush=True,
                    )
                    image = pipeline(
                        prompt=job["prompt"],
                        negative_prompt=job["negative_prompt"],
                        num_inference_steps=SD35_NUM_INFERENCE_STEPS,
                        guidance_scale=SD35_GUIDANCE_SCALE,
                        generator=torch.Generator("cuda").manual_seed(job["seed"]),
                        width=SD35_IMAGE_SIZE,
                        height=SD35_IMAGE_SIZE,
                    ).images[0]
                    image.save(job["image_path"])

            with open(OBJECT_MANIFEST, "w", encoding="utf-8") as file:
                json.dump(jobs, file, ensure_ascii=False, indent=2)
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"SD3.5 object generation failed: {exc}",
            ) from exc
        finally:
            try:
                _move_pipeline("cpu")
            except Exception as cleanup_exc:
                print(
                    f"Failed to offload SD3.5 pipeline after generation: "
                    f"{cleanup_exc}",
                    flush=True,
                )
                _safe_cuda_cleanup()

    return {"jobs": jobs}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resident SD3.5 worker")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
