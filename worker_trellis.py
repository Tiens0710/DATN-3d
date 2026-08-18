"""Resident TRELLIS image-to-3D worker for the Kaggle GPU session."""

import argparse
import gc
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


app = FastAPI(title="DATN TRELLIS Worker", version="1.2.3")
pipeline = None
pipeline_device = None
variant_pipeline = None
variant_pipeline_device = None
postprocessing_utils = None
load_error = None
variant_load_error = None
inference_lock = threading.Lock()
TRELLIS_SPARSE_STEPS = int(os.environ.get("TRELLIS_SPARSE_STEPS", "16"))
TRELLIS_SLAT_STEPS = int(os.environ.get("TRELLIS_SLAT_STEPS", "16"))
TRELLIS_VARIANT_MODEL = os.environ.get(
    "TRELLIS_VARIANT_MODEL", "microsoft/TRELLIS-text-base"
)
TRELLIS_VARIANT_SLAT_STEPS = int(
    os.environ.get("TRELLIS_VARIANT_SLAT_STEPS", str(TRELLIS_SLAT_STEPS))
)
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
TRELLIS_REMOVE_FLOOR_BACKDROP = os.environ.get(
    "TRELLIS_REMOVE_FLOOR_BACKDROP", "1"
).lower() in {"1", "true", "yes", "on"}


class GenerateRequest(BaseModel):
    crop_path: str
    name: str
    output_dir: str


class VariantRequest(BaseModel):
    base_mesh_path: str
    prompt: str
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


def _load_trellis_input(path: Path) -> Image.Image:
    """Load a crop without flattening a segmentation alpha mask.

    White furniture is valid input, but a white RGB background is ambiguous to
    TRELLIS' own background remover.  The SAM2 worker now preserves RGBA crops;
    pass those through unchanged and reject an unusable alpha mask early.
    """
    with Image.open(path) as opened_image:
        has_alpha = "A" in opened_image.getbands() or "transparency" in opened_image.info
        image = opened_image.convert("RGBA" if has_alpha else "RGB")

    if not has_alpha:
        return image

    import numpy as np

    alpha = np.asarray(image.getchannel("A"), dtype=np.uint8)
    support = alpha >= 32
    area_ratio = float(support.mean()) if support.size else 0.0
    if area_ratio < 0.001:
        raise ValueError("TRELLIS input has an empty alpha mask; segmentation produced no object")
    if area_ratio > 0.92:
        raise ValueError(
            "TRELLIS input alpha mask covers almost the entire image; refusing to model the background"
        )
    # Keep the cleaned RGBA crop intact. TRELLIS 1's official
    # ``preprocess_image`` function uses this alpha channel, crops/resizes the
    # object, then premultiplies RGB exactly once before its DINOv2 encoder.
    # Premultiplying here as well darkens anti-aliased object edges twice and
    # can make a black floor hallucination more likely.
    return image


def _remove_hallucinated_floor_backdrop(glb) -> tuple[int, str]:
    """Remove an obvious large horizontal floor hallucinated by image-to-3D.

    TRELLIS is trained to infer a complete asset from a single image.  With
    product photos it occasionally emits a large, black, horizontal sheet
    beneath the object even though the input alpha has no such background.
    The exported mesh uses Y as its up axis, so a true floor is both near the
    lowest Y coordinate and spans almost the whole X/Z footprint.  Those two
    constraints make this conservative: ordinary small horizontal parts such
    as shelves, seats and table tops are not removed.
    """
    if not TRELLIS_REMOVE_FLOOR_BACKDROP:
        return 0, "disabled"

    try:
        import numpy as np

        vertices = np.asarray(glb.vertices, dtype=np.float64)
        faces = np.asarray(glb.faces, dtype=np.int64)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or len(faces) < 12:
            return 0, "mesh has no usable faces"

        extents = np.ptp(vertices, axis=0)
        xz_extent = extents[[0, 2]]
        y_extent = float(extents[1])
        if y_extent <= 1e-6 or np.any(xz_extent <= 1e-6):
            return 0, "degenerate bounds"

        face_vertices = vertices[faces]
        face_y_max = face_vertices[:, :, 1].max(axis=1)
        floor_band = max(0.006, y_extent * 0.085)
        near_floor = face_y_max <= vertices[:, 1].min() + floor_band
        horizontal = np.abs(np.asarray(glb.face_normals)[:, 1]) >= 0.94
        horizontal_candidates = np.flatnonzero(near_floor & horizontal)
        candidates = np.flatnonzero(near_floor)
        if len(horizontal_candidates) < 4 or len(candidates) < 8:
            return 0, "no broad lower horizontal faces"

        candidate_vertices = vertices[
            np.unique(faces[horizontal_candidates].reshape(-1))
        ]
        candidate_xz = np.ptp(candidate_vertices[:, [0, 2]], axis=0)
        footprint_coverage = candidate_xz / np.maximum(xz_extent, 1e-8)
        candidate_area = float(
            np.asarray(glb.area_faces)[horizontal_candidates].sum()
        )
        footprint_area = float(xz_extent[0] * xz_extent[1])
        total_area = max(float(np.asarray(glb.area_faces).sum()), 1e-8)

        # The plane must cover nearly the full scene footprint and represent a
        # material part of the mesh.  This avoids touching the undersides of
        # valid furniture, which are much smaller than a generated backdrop.
        if (
            footprint_coverage[0] < 0.76
            or footprint_coverage[1] < 0.76
            or candidate_area < max(footprint_area * 0.22, total_area * 0.10)
        ):
            return 0, "lower faces do not match a large backdrop"

        keep = np.ones(len(faces), dtype=bool)
        keep[candidates] = False
        glb.update_faces(keep)
        glb.remove_unreferenced_vertices()
        return int(len(candidates)), (
            "removed large lower backdrop "
            f"(coverage={footprint_coverage[0]:.2f}x{footprint_coverage[1]:.2f})"
        )
    except Exception as exc:
        # A failed cleanup must never discard a successfully generated asset.
        return 0, f"cleanup skipped: {type(exc).__name__}: {exc}"


def _safe_empty_cache(name: str) -> None:
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception as exc:
        _log(
            f"{name}: CUDA cleanup failed but original result/error is preserved: "
            f"{type(exc).__name__}: {exc}"
        )


def _move_pipeline(device: str) -> None:
    global pipeline_device

    if pipeline is None:
        raise RuntimeError("TRELLIS pipeline is not loaded")
    if pipeline_device == device:
        return

    _log(f"Moving TRELLIS pipeline to {device}...")
    try:
        pipeline.to(device)
    except Exception:
        pipeline_device = None
        raise
    pipeline_device = device
    if device == "cpu":
        _safe_empty_cache("pipeline offload")
    _log(f"TRELLIS pipeline is now on {device}; {_cuda_memory()}")


def _move_variant_pipeline(device: str) -> None:
    global variant_pipeline_device

    if variant_pipeline is None:
        raise RuntimeError("TRELLIS variant pipeline is not loaded")
    if variant_pipeline_device == device:
        return

    _log(f"Moving TRELLIS variant pipeline to {device}...")
    try:
        variant_pipeline.to(device)
    except Exception:
        variant_pipeline_device = None
        raise
    variant_pipeline_device = device
    if device == "cpu":
        _safe_empty_cache("variant pipeline offload")
    _log(f"TRELLIS variant pipeline is now on {device}; {_cuda_memory()}")


def _load_pipeline() -> None:
    """Load once and keep failures observable through /health."""
    global pipeline, pipeline_device, postprocessing_utils, load_error

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
        pipeline_device = "cuda"
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


def _load_variant_pipeline() -> None:
    """Load the official TRELLIS 1 text pipeline only when a variant is requested."""
    global variant_pipeline, variant_pipeline_device, variant_load_error

    if variant_pipeline is not None:
        return

    try:
        from trellis.pipelines import TrellisTextTo3DPipeline

        _log(f"Loading TRELLIS variant pipeline {TRELLIS_VARIANT_MODEL}...")
        loaded_pipeline = TrellisTextTo3DPipeline.from_pretrained(
            TRELLIS_VARIANT_MODEL
        )
        loaded_pipeline.to("cuda")
        variant_pipeline = loaded_pipeline
        variant_pipeline_device = "cuda"
        variant_load_error = None
        _log(
            f"TRELLIS variant pipeline ready ({TRELLIS_VARIANT_MODEL}); "
            f"{_cuda_memory()}"
        )
    except Exception as exc:
        variant_load_error = f"{exc}\n{traceback.format_exc()}"
        _log(f"TRELLIS variant pipeline failed to load:\n{variant_load_error}")
        raise


@app.on_event("startup")
def startup_event() -> None:
    _load_pipeline()


@app.get("/health")
def health() -> dict:
    return {
        "ready": pipeline is not None,
        "error": load_error,
        "device": pipeline_device,
        "variant_ready": variant_pipeline is not None,
        "variant_device": variant_pipeline_device,
        "variant_error": variant_load_error,
        "variant_model": TRELLIS_VARIANT_MODEL,
        "cuda_memory": _cuda_memory(),
        "profile": {
            "sparse_steps": TRELLIS_SPARSE_STEPS,
            "slat_steps": TRELLIS_SLAT_STEPS,
            "variant_slat_steps": TRELLIS_VARIANT_SLAT_STEPS,
            "variant_model": TRELLIS_VARIANT_MODEL,
            "texture_size": TRELLIS_TEXTURE_SIZE,
            "texture_mode": TRELLIS_TEXTURE_MODE,
            "texture_views": TRELLIS_TEXTURE_VIEWS,
            "bake_resolution": TRELLIS_BAKE_RESOLUTION,
            "fill_holes": TRELLIS_FILL_HOLES,
        },
    }


@app.post("/offload")
def offload() -> dict:
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail=load_error or "TRELLIS model is still loading",
        )

    with inference_lock:
        try:
            _move_pipeline("cpu")
            if variant_pipeline is not None:
                _move_variant_pipeline("cpu")
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to offload TRELLIS pipeline: {exc}",
            ) from exc
    return {
        "ready": True,
        "device": pipeline_device,
        "cuda_memory": _cuda_memory(),
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
            if variant_pipeline is not None:
                _move_variant_pipeline("cpu")
            _move_pipeline("cuda")
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            _log(f"Request {name}: crop={crop_path}; {_cuda_memory()}")

            image = _load_trellis_input(crop_path)
            input_mode = "RGBA/SAM2 alpha" if "A" in image.getbands() else "RGB fallback"
            _log(
                f"{name}: preprocessing image ({image.width}x{image.height}, {input_mode})"
            )
            image = pipeline.preprocess_image(image)

            _log(f"{name}: running sparse structure + SLAT diffusion")
            run_context = (
                torch.no_grad()
                if TRELLIS_TEXTURE_MODE == "opt"
                else torch.inference_mode()
            )
            with run_context:
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
            # The optimized texture baker performs gradient-based updates.
            # Running it inside inference_mode() produces tensors without a
            # grad_fn and fails with "does not require grad".
            if TRELLIS_TEXTURE_MODE == "opt":
                with torch.enable_grad():
                    glb = postprocessing_utils.to_glb(
                        outputs["gaussian"][0],
                        outputs["mesh"][0],
                        simplify=TRELLIS_SIMPLIFY,
                        fill_holes=TRELLIS_FILL_HOLES,
                        texture_size=TRELLIS_TEXTURE_SIZE,
                        verbose=False,
                    )
            else:
                with torch.inference_mode():
                    glb = postprocessing_utils.to_glb(
                        outputs["gaussian"][0],
                        outputs["mesh"][0],
                        simplify=TRELLIS_SIMPLIFY,
                        fill_holes=TRELLIS_FILL_HOLES,
                        texture_size=TRELLIS_TEXTURE_SIZE,
                        verbose=False,
                    )
            removed_faces, cleanup_detail = _remove_hallucinated_floor_backdrop(glb)
            if removed_faces:
                _log(f"{name}: {cleanup_detail}; removed {removed_faces} floor faces")
            else:
                _log(f"{name}: floor cleanup: {cleanup_detail}")
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
            # These objects can retain large CUDA tensors until the function
            # returns.  Release them before empty_cache so the next worker can
            # safely claim the GPU.
            if "outputs" in locals():
                del outputs
            if "glb" in locals():
                del glb
            if "image" in locals():
                del image
            gc.collect()
            _safe_empty_cache(name)
            _log(f"{name}: request cleanup; {_cuda_memory()}")

    return {"model_path": str(output_path)}


def _open3d_mesh_from_glb(path: Path):
    """Read a GLB as one triangle mesh for TRELLIS' official run_variant API."""
    import numpy as np
    import open3d as o3d
    import trimesh

    loaded = trimesh.load(str(path), force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = loaded.dump(concatenate=True)
    vertices = np.asarray(getattr(loaded, "vertices", []), dtype=np.float64)
    faces = np.asarray(getattr(loaded, "faces", []), dtype=np.int32)
    if vertices.size == 0 or faces.size == 0:
        raise RuntimeError(f"Base GLB has no triangle mesh: {path}")

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        raise RuntimeError(f"Base GLB has no usable triangles: {path}")
    return mesh


@app.post("/variant")
def generate_variant(payload: VariantRequest) -> dict:
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail=load_error or "TRELLIS image pipeline is not loaded",
        )

    base_mesh_path = Path(payload.base_mesh_path)
    if not base_mesh_path.is_file() or base_mesh_path.suffix.lower() != ".glb":
        raise HTTPException(status_code=404, detail="Base GLB not found")
    prompt = " ".join(str(payload.prompt or "").split()).strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Variant prompt must not be empty")
    if len(prompt) > 1000:
        raise HTTPException(status_code=400, detail="Variant prompt is too long")

    name = Path(payload.name).name
    if not name or name in {".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid variant model name")
    output_dir = Path(payload.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{name}.glb"

    with inference_lock:
        started_at = time.perf_counter()
        try:
            # The T4 cannot keep both large pipelines on CUDA.  The image
            # pipeline is moved to CPU while the official text variant model
            # performs run_variant, then it is released again in finally.
            _move_pipeline("cpu")
            _load_variant_pipeline()
            _move_variant_pipeline("cuda")
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            _log(
                f"{name}: run_variant base={base_mesh_path}; "
                f"prompt={prompt}; {_cuda_memory()}"
            )
            base_mesh = _open3d_mesh_from_glb(base_mesh_path)
            with torch.no_grad():
                outputs = variant_pipeline.run_variant(
                    base_mesh,
                    prompt,
                    seed=42,
                    formats=["gaussian", "mesh"],
                    slat_sampler_params={
                        "steps": TRELLIS_VARIANT_SLAT_STEPS,
                        "cfg_strength": 3.0,
                    },
                )

            _log(f"{name}: converting variant output to textured GLB")
            if TRELLIS_TEXTURE_MODE == "opt":
                with torch.enable_grad():
                    glb = postprocessing_utils.to_glb(
                        outputs["gaussian"][0],
                        outputs["mesh"][0],
                        simplify=TRELLIS_SIMPLIFY,
                        fill_holes=TRELLIS_FILL_HOLES,
                        texture_size=TRELLIS_TEXTURE_SIZE,
                        verbose=False,
                    )
            else:
                with torch.inference_mode():
                    glb = postprocessing_utils.to_glb(
                        outputs["gaussian"][0],
                        outputs["mesh"][0],
                        simplify=TRELLIS_SIMPLIFY,
                        fill_holes=TRELLIS_FILL_HOLES,
                        texture_size=TRELLIS_TEXTURE_SIZE,
                        verbose=False,
                    )
            removed_faces, cleanup_detail = _remove_hallucinated_floor_backdrop(glb)
            if removed_faces:
                _log(
                    f"{name}: variant {cleanup_detail}; "
                    f"removed {removed_faces} floor faces"
                )
            else:
                _log(f"{name}: variant floor cleanup: {cleanup_detail}")
            glb.export(str(output_path))
            if not output_path.is_file():
                raise RuntimeError("TRELLIS variant export finished but no file was written")
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            _log(
                f"{name}: variant saved in "
                f"{time.perf_counter() - started_at:.1f}s; {_cuda_memory()}"
            )
        except HTTPException:
            raise
        except Exception as exc:
            error_trace = traceback.format_exc()
            _log(
                f"{name}: VARIANT FAILED after "
                f"{time.perf_counter() - started_at:.1f}s; {_cuda_memory()}\n"
                f"{error_trace}"
            )
            detail = f"{type(exc).__name__}: {exc}"
            if "out of memory" in str(exc).lower():
                detail += ". GPU out of memory while running TRELLIS variant."
            raise HTTPException(
                status_code=500,
                detail=f"TRELLIS variant generation failed for {name}: {detail}",
            ) from exc
        finally:
            if "outputs" in locals():
                del outputs
            if "glb" in locals():
                del glb
            if "base_mesh" in locals():
                del base_mesh
            if variant_pipeline is not None and variant_pipeline_device == "cuda":
                try:
                    _move_variant_pipeline("cpu")
                except Exception as exc:
                    _log(f"{name}: variant pipeline offload failed: {exc}")
            gc.collect()
            _safe_empty_cache(name)
            _log(f"{name}: variant request cleanup; {_cuda_memory()}")

    return {
        "model_path": str(output_path),
        "mode": "variant",
        "base_mesh_path": str(base_mesh_path),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resident TRELLIS worker")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8002)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
