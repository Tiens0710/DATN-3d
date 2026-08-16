"""Resident Stable Diffusion 3.5 worker for per-object image generation."""

import argparse
import gc
import inspect
import json
import os
import shutil
import sys
import threading
import traceback
from pathlib import Path


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

from src.generator_2d import build_object_jobs


SD35_CACHE_DIR = "/kaggle/working/sd35_medium_cache_v1"
SD35_NUM_INFERENCE_STEPS = int(os.environ.get("SD35_NUM_INFERENCE_STEPS", "28"))
SD35_IMAGE_SIZE = int(os.environ.get("SD35_IMAGE_SIZE", "768"))
SD35_GUIDANCE_SCALE = float(os.environ.get("SD35_GUIDANCE_SCALE", "4.5"))
SD35_LORA_PATH = os.environ.get(
    "SD35_LORA_PATH",
    "/kaggle/working/lora_sd35_fast_safe/best",
).strip()
SD35_LORA_SCALE = float(os.environ.get("SD35_LORA_SCALE", "0.2"))

app = FastAPI(title="DATN SD3.5 Worker", version="1.0.0")
pipeline = None
pipeline_device = None
load_error = None
lora_loaded = False
lora_error = None
inference_lock = threading.Lock()


class GenerateRequest(BaseModel):
    scene_prompt: str
    objects: list[dict]
    lora_scale: float = 0.2
    output_dir: str


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
    )


def _compatible_lora_directory(source: str) -> str:
    """Copy only the adapter metadata/weights and remove unknown config keys."""
    source_path = Path(source)
    compat_path = Path("/kaggle/working/sd35_lora_compat")
    compat_path.mkdir(parents=True, exist_ok=True)

    with (source_path / "adapter_config.json").open(
        "r", encoding="utf-8"
    ) as file:
        config = json.load(file)

    from peft import LoraConfig

    allowed = set(inspect.signature(LoraConfig.__init__).parameters)
    config = {
        key: value
        for key, value in config.items()
        if key in allowed or key in {"peft_type", "task_type"}
    }

    shutil.copy2(
        source_path / "adapter_model.safetensors",
        compat_path / "adapter_model.safetensors",
    )
    with (compat_path / "adapter_config.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(config, file, indent=2)
    return str(compat_path)


def _load_lora_adapter(loaded_pipeline):
    """Load and merge the optional adapter at the configured multiplier."""
    global lora_loaded, lora_error

    lora_loaded = False
    lora_error = None
    if SD35_LORA_SCALE <= 0:
        print("SD35 LoRA disabled; using the base model.", flush=True)
        return loaded_pipeline

    source_path = Path(SD35_LORA_PATH)
    required = (
        source_path / "adapter_config.json",
        source_path / "adapter_model.safetensors",
    )
    if not SD35_LORA_PATH or not all(path.is_file() for path in required):
        print(
            "SD35 LoRA checkpoint not found; using base model. "
            f"Expected: {SD35_LORA_PATH}",
            flush=True,
        )
        return loaded_pipeline

    try:
        import peft.import_utils as peft_import_utils
        import peft.tuners.lora.torchao as peft_torchao

        peft_import_utils.is_torchao_available = lambda: False
        peft_torchao.is_torchao_available = lambda: False
    except Exception as exc:
        print(f"TorchAO patch skipped: {exc}", flush=True)

    from peft import PeftModel

    print(
        f"Loading SD35 LoRA: {SD35_LORA_PATH} "
        f"(scale={SD35_LORA_SCALE})",
        flush=True,
    )
    try:
        try:
            peft_transformer = PeftModel.from_pretrained(
                loaded_pipeline.transformer,
                SD35_LORA_PATH,
                is_trainable=False,
            )
        except TypeError as exc:
            if "unexpected keyword argument" not in str(exc):
                raise
            compat_path = _compatible_lora_directory(SD35_LORA_PATH)
            print(
                f"Retrying with compatible LoRA config: {compat_path}",
                flush=True,
            )
            peft_transformer = PeftModel.from_pretrained(
                loaded_pipeline.transformer,
                compat_path,
                is_trainable=False,
            )

        scaled_branches = 0
        for module in peft_transformer.modules():
            scaling = getattr(module, "scaling", None)
            if isinstance(scaling, dict):
                for adapter_name in list(scaling):
                    scaling[adapter_name] = SD35_LORA_SCALE
                    scaled_branches += 1
        if scaled_branches == 0:
            raise RuntimeError("No LoRA scaling branches were found")

        loaded_pipeline.transformer = peft_transformer.merge_and_unload(
            safe_merge=True
        )
        del peft_transformer
        gc.collect()
        lora_loaded = True
        print(
            f"SD35 LoRA merged successfully: scale={SD35_LORA_SCALE}, "
            f"branches={scaled_branches}",
            flush=True,
        )
        return loaded_pipeline
    except Exception as exc:
        lora_error = f"{type(exc).__name__}: {exc}"
        raise RuntimeError(
            f"SD35 LoRA checkpoint could not be loaded from {SD35_LORA_PATH}: "
            f"{lora_error}"
        ) from exc


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
    try:
        pipeline.to(device)
    except Exception:
        # A failed CUDA transfer can leave some modules moved and makes the
        # cached device flag unreliable.  Mark it unknown so recovery always
        # attempts a real CPU transfer instead of skipping cleanup.
        pipeline_device = None
        raise
    pipeline_device = device
    if device == "cpu":
        _safe_cuda_cleanup()
    print(f"SD3.5 pipeline is now on {device}.", flush=True)


def _force_offload_pipeline() -> None:
    """Move the pipeline to CPU even after a partially failed CUDA transfer."""
    global pipeline_device

    if pipeline is None:
        return
    try:
        if pipeline_device != "cpu":
            pipeline.to("cpu")
        pipeline_device = "cpu"
    finally:
        _safe_cuda_cleanup()


def _is_cuda_memory_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "cuda out of memory" in message


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
        pipeline = _load_lora_adapter(pipeline)
        pipeline = pipeline.to("cuda")
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
        "lora_loaded": lora_loaded,
        "lora_path": SD35_LORA_PATH,
        "lora_scale": SD35_LORA_SCALE,
        "lora_error": lora_error,
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

    if abs(float(payload.lora_scale) - SD35_LORA_SCALE) > 1e-6:
        print(
            "Request lora_scale differs from resident worker scale; "
            f"using resident scale={SD35_LORA_SCALE}",
            flush=True,
        )
    output_dir = Path(payload.output_dir).resolve()
    runs_root = Path("/kaggle/working/runs").resolve()
    run_id = output_dir.parent.name
    if (
        runs_root not in output_dir.parents
        or output_dir.name != "object_images"
        or len(run_id) != 32
        or any(character not in "0123456789abcdef" for character in run_id)
    ):
        raise HTTPException(status_code=400, detail="Invalid object image output directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = build_object_jobs(
        payload.scene_prompt,
        payload.objects,
        object_image_dir=str(output_dir),
    )

    with inference_lock:
        try:
            for attempt in range(2):
                try:
                    _safe_cuda_cleanup()
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
                    break
                except RuntimeError as exc:
                    if attempt or not _is_cuda_memory_error(exc):
                        raise
                    print(
                        "SD3.5 CUDA out-of-memory; resetting the pipeline to CPU "
                        "and retrying the complete object batch once.",
                        flush=True,
                    )
                    _force_offload_pipeline()

        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"SD3.5 object generation failed: {exc}",
            ) from exc
        finally:
            try:
                _force_offload_pipeline()
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
