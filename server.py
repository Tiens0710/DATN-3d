import os
import asyncio
import io
import importlib.util
import json
import shutil
import threading
import uuid
import zipfile
from pathlib import Path
from urllib import request as urllib_request
from urllib.parse import urlparse
from typing import Dict, List, Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from src.parser import parse_scene_graph, parse_scene_graph_from_objects
from src.layout import compute_layout
from src.generator_2d import create_object_contact_sheet, generate_object_images
from src.segmenter import run_grounded_sam2
from src.generator_3d import generate_3d_models
from src.combiner import combine_scene_meshes


KAGGLE_WORKING = "/kaggle/working"
CROPS_DIR = os.path.join(KAGGLE_WORKING, "crops")
MULTI_GLB_DIR = os.path.join(KAGGLE_WORKING, "multi_object_glb")
OUT_DIR = os.path.join(KAGGLE_WORKING, "outputs", "trellis")
OBJECT_IMAGE_DIR = os.path.join(KAGGLE_WORKING, "object_images")
UPLOADS_DIR = os.path.join(KAGGLE_WORKING, "uploads")

os.makedirs(CROPS_DIR, exist_ok=True)
os.makedirs(MULTI_GLB_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(OBJECT_IMAGE_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

app = FastAPI(
    title="DATN 3D Scene Reconstruction API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/crops", StaticFiles(directory=CROPS_DIR), name="crops")
app.mount(
    "/multi_object_glb",
    StaticFiles(directory=MULTI_GLB_DIR),
    name="multi_object_glb",
)
app.mount("/outputs", StaticFiles(directory=OUT_DIR), name="outputs")
app.mount(
    "/object_images",
    StaticFiles(directory=OBJECT_IMAGE_DIR),
    name="object_images",
)
app.mount("/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")


class TextPrompt(BaseModel):
    text: str


class PromptOptimizeRequest(BaseModel):
    prompt: str


class ImageGenRequest(BaseModel):
    prompt: str
    layout: Dict[str, Any]
    lora_scale: float = 0.6


class Sam2Request(BaseModel):
    image_url: str
    layout: Dict[str, Any]
    prompt: str = ""
    mode: str = "objectwise"
    auto_detect: bool = False


class TrellisRequest(BaseModel):
    crops: List[Dict[str, Any]]


class CombineRequest(BaseModel):
    models: List[Dict[str, Any]]
    layout: Dict[str, Any]
    scale_factor: float = 0.01


trellis_jobs: Dict[str, Dict[str, Any]] = {}
trellis_jobs_lock = threading.Lock()
# One worker processes GPU-heavy TRELLIS jobs in FIFO order.
# Blocking inference runs outside the event loop via asyncio.to_thread().
trellis_queue = asyncio.Queue()
trellis_worker_task = None

GEMINI_MODEL = os.environ.get(
    "GEMINI_MODEL",
    "gemini-3.1-flash-lite",
)

PROMPT_OPTIMIZER_INSTRUCTION = """
You optimize user prompts for Stable Diffusion 3.5 Medium in a furniture
text-to-2D-to-3D pipeline.
Return exactly one English prompt and nothing else.
Preserve every requested object, exact object count, material, color, style,
spatial relationship, decorative motif, carving, and engraving.
When a furniture style is not specified, enrich the requested furniture with
subtle, physically plausible craftsmanship details such as visible wood grain,
beveled edges, a framed panel, a restrained carved border, or a small inlay.
These details belong to the object; they must never become extra objects.
Do not force ornate decoration when the user asks for a plain or minimalist
design.
When the user requests one object, explicitly write "exactly one".
For example, one table and one chair must become exactly one table and exactly
one chair.
Never invent extra furniture, duplicate objects, or decorative props.
If the input is Vietnamese, translate it naturally to English.
Make each object fully visible with accurate geometry, realistic construction,
and minimal overlap.
Use a three-quarter product-photography view, centered composition, clean
neutral studio background, soft even lighting, sharp focus, and realistic
materials. The result must be suitable for GroundingDINO, SAM2 segmentation,
and single-image 3D reconstruction. Keep the prompt under 90 words.
""".strip()

SCENE_GRAPH_INSTRUCTION = """
Extract the exact object plan from the user's prompt for an image-to-3D
pipeline. Return JSON only, with this schema:
{"objects":[{"label":"singular concrete object name in English", "count":1,
"description":"short description of this object"}],
"relation":"single|next_to|on_top_of|under|in_front_of|behind|left_of|right_of",
"relations":[{"subject":0,"relation":"next_to","object":1}]}

Rules:
- Recognize ANY concrete object, not only furniture. Keep labels such as
  bicycle, house, toy, car, lamp, plant, cup, or any other object requested.
- Preserve the exact requested quantity. "two chairs" means count 2; do not
  count a reference such as "the table" after "beside the table" again.
- Do not turn parts, materials, colors, or decorative motifs into objects.
  For example, bicycle wheels and table legs are parts, not separate objects.
- Use one object entry per object category and put repeated quantity in count.
- Keep the user's material, color, style, and distinguishing details in
  description. Do not invent props or objects.
- Relations use zero-based indexes into objects. Use an empty relations list
  for a single object.
""".strip()


def optimize_prompt_with_gemini(prompt: str) -> Dict[str, Any]:
    clean_prompt = " ".join(prompt.split()).strip()
    if not clean_prompt:
        raise ValueError("Prompt must not be empty")

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return {
            "optimized_prompt": (
                f"{clean_prompt}, photorealistic furniture product photography, "
                "full objects visible, accurate proportions and construction, "
                "three-quarter view, centered composition, clean neutral studio "
                "background, soft even lighting, sharp focus, realistic materials, "
                "minimal overlap, no people, no text, no watermark, no extra objects, "
                "suitable for object segmentation and single-image 3D reconstruction"
            ),
            "used_gemini": False,
            "warning": "GEMINI_API_KEY is not configured",
            "model": GEMINI_MODEL,
        }
    endpoint = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent"
    )
    payload = {
        "systemInstruction": {
            "parts": [{"text": PROMPT_OPTIMIZER_INSTRUCTION}]
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": clean_prompt}],
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 220,
        },
    }
    http_request = urllib_request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        },
        method="POST",
    )

    try:
        with urllib_request.urlopen(http_request, timeout=45) as response:
            response_data = json.loads(response.read().decode("utf-8"))

        parts = (
            response_data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [])
        )
        optimized_prompt = " ".join(
            part.get("text", "").strip()
            for part in parts
            if part.get("text")
        ).strip()
        if not optimized_prompt:
            raise RuntimeError("Gemini returned an empty prompt")

        return {
            "optimized_prompt": optimized_prompt,
            "used_gemini": True,
            "model": GEMINI_MODEL,
        }
    except Exception as exc:
        return {
            "optimized_prompt": (
                f"{clean_prompt}, photorealistic furniture product photography, "
                "full objects visible, accurate proportions and construction, "
                "visible material texture and tasteful woodworking details, "
                "three-quarter view, centered composition, clean neutral studio "
                "background, soft even lighting, sharp focus, realistic materials, "
                "minimal overlap, no people, no text, no watermark, no extra objects, "
                "suitable for object segmentation and single-image 3D reconstruction"
            ),
            "used_gemini": False,
            "warning": f"Gemini optimization failed: {exc}",
            "model": GEMINI_MODEL,
        }


def _gemini_scene_graph(prompt: str) -> dict | None:
    """Ask Gemini for arbitrary object extraction; return None on fallback."""
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None

    endpoint = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent"
    )
    payload = {
        "systemInstruction": {"parts": [{"text": SCENE_GRAPH_INSTRUCTION}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 400},
    }
    http_request = urllib_request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        },
        method="POST",
    )
    with urllib_request.urlopen(http_request, timeout=45) as response:
        response_data = json.loads(response.read().decode("utf-8"))
    parts = (
        response_data.get("candidates", [{}])[0]
        .get("content", {})
        .get("parts", [])
    )
    text = " ".join(
        part.get("text", "").strip()
        for part in parts
        if part.get("text")
    ).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
    data = json.loads(text)
    objects = data.get("objects")
    if not isinstance(objects, list) or not objects or len(objects) > 12:
        raise ValueError("Gemini returned an invalid object list")
    clean_objects = []
    for item in objects:
        if not isinstance(item, dict):
            continue
        label = " ".join(str(item.get("label", "")).split()).strip(" ,.;:")
        if not label:
            continue
        clean_objects.append({
            "label": label.lower(),
            "count": item.get("count", 1),
            "description": " ".join(
                str(item.get("description", label)).split()
            ).strip(),
        })
    if not clean_objects:
        raise ValueError("Gemini returned no usable objects")
    return parse_scene_graph_from_objects(
        prompt,
        clean_objects,
        relation=data.get("relation"),
        relations=data.get("relations", []),
        parser_source="gemini_structured",
    )


def parse_scene_graph_for_request(prompt: str) -> dict:
    """Use Gemini for open-vocabulary parsing, then deterministic fallback."""
    clean_prompt = " ".join(str(prompt).split()).strip()
    try:
        graph = _gemini_scene_graph(clean_prompt)
        if graph:
            return graph
    except Exception as exc:
        fallback = parse_scene_graph(clean_prompt)
        fallback["parser_warning"] = f"Gemini scene parsing failed: {exc}"
        return fallback
    return parse_scene_graph(clean_prompt)


def backend_readiness() -> Dict[str, Any]:
    """Report whether the current Kaggle session can run every pipeline stage."""
    required_files = {
        "groundingdino_weights": Path(
            "/kaggle/working/groundingdino_ckpt/groundingdino_swint_ogc.pth"
        ),
        "groundingdino_config": Path(
            "/kaggle/working/groundingdino_ckpt/GroundingDINO_SwinT_OGC.py"
        ),
        "sam2_weights": Path(
            "/kaggle/working/sam2_ckpt/sam2_hiera_small.pt"
        ),
        "trellis_source": Path("/kaggle/working/TRELLIS/trellis"),
    }
    required_modules = (
        "torch",
        "diffusers",
        "transformers",
        "peft",
        "groundingdino",
        "sam2",
        "spconv",
        "xformers",
        "trimesh",
        "multipart",
    )

    files = {name: path.exists() for name, path in required_files.items()}
    modules = {
        name: importlib.util.find_spec(name) is not None
        for name in required_modules
    }

    cuda_available = False
    cuda_name = None
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        if cuda_available:
            cuda_name = torch.cuda.get_device_name(0)
    except Exception:
        pass

    checks = {
        "hf_token": bool(os.environ.get("HF_TOKEN", "").strip()),
        "gemini_api_key": bool(
            os.environ.get("GEMINI_API_KEY", "").strip()
        ),
        "cuda": cuda_available,
        "files": files,
        "modules": modules,
    }
    ready = (
        checks["hf_token"]
        and checks["cuda"]
        and all(files.values())
        and all(modules.values())
    )
    return {
        "ready": ready,
        "checks": checks,
        "cuda_name": cuda_name,
    }


def run_trellis_job(job_id: str, crops: List[Dict[str, Any]]) -> None:
    with trellis_jobs_lock:
        trellis_jobs[job_id]["status"] = "running"

    try:
        models = generate_3d_models(crops, MULTI_GLB_DIR)
        with trellis_jobs_lock:
            trellis_jobs[job_id] = {
                "status": "completed",
                "models": models,
            }
    except Exception as exc:
        with trellis_jobs_lock:
            trellis_jobs[job_id] = {
                "status": "failed",
                "error": str(exc),
            }


def get_trellis_queue_position(job_id: str) -> int:
    with trellis_jobs_lock:
        queued_ids = [
            current_id
            for current_id, job in trellis_jobs.items()
            if job.get("status") == "queued"
        ]
        try:
            return queued_ids.index(job_id) + 1
        except ValueError:
            return 0


async def trellis_queue_worker() -> None:
    """Process queued TRELLIS jobs without blocking the FastAPI event loop."""
    while True:
        job_id, crops = await trellis_queue.get()
        try:
            await asyncio.to_thread(run_trellis_job, job_id, crops)
        finally:
            trellis_queue.task_done()


@app.on_event("startup")
async def start_background_workers() -> None:
    global trellis_worker_task
    if trellis_worker_task is None or trellis_worker_task.done():
        trellis_worker_task = asyncio.create_task(
            trellis_queue_worker(),
            name="trellis-queue-worker",
        )


@app.on_event("shutdown")
async def stop_background_workers() -> None:
    global trellis_worker_task
    if trellis_worker_task is None:
        return

    trellis_worker_task.cancel()
    try:
        await trellis_worker_task
    except asyncio.CancelledError:
        pass
    trellis_worker_task = None


@app.get("/")
def root():
    return {"status": "ok", "message": "DATN API is running"}


@app.get("/api/health")
def health_check():
    readiness = backend_readiness()
    return {
        "status": "ok",
        "version": "1.1.0",
        **readiness,
        "gemini_model": GEMINI_MODEL,
        "gemini_configured": readiness["checks"]["gemini_api_key"],
    }


@app.post("/api/optimize_prompt")
def api_optimize_prompt(request: PromptOptimizeRequest):
    try:
        return optimize_prompt_with_gemini(request.prompt)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/upload_image")
async def api_upload_image(file: UploadFile = File(...)):
    """Store a user image for the upload-to-3D branch of the pipeline."""
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Only image files are supported")

    raw = await file.read(20 * 1024 * 1024 + 1)
    if len(raw) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image must be 20 MB or smaller")

    try:
        with Image.open(io.BytesIO(raw)) as image:
            image.verify()
        with Image.open(io.BytesIO(raw)) as image:
            rgb = image.convert("RGB")
            width, height = rgb.size
            filename = f"upload_{uuid.uuid4().hex}.png"
            output_path = os.path.join(UPLOADS_DIR, filename)
            rgb.save(output_path, format="PNG")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}")

    # Keep the legacy path available for archive/download tooling.
    shutil.copy2(output_path, os.path.join(KAGGLE_WORKING, "input.png"))
    return {
        "status": "success",
        "mode": "uploaded",
        "image_url": f"/uploads/{filename}",
        "filename": filename,
        "width": width,
        "height": height,
    }


@app.post("/api/parse_scene_graph")
def api_parse_scene_graph(request: TextPrompt):
    try:
        return parse_scene_graph_for_request(request.text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/generate_layout")
def api_generate_layout(scene_graph: Dict[str, Any]):
    try:
        return compute_layout(scene_graph)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/generate_image")
def api_generate_image(request: ImageGenRequest):
    try:
        layout = dict(request.layout or {})
        objects = list(layout.get("objects", []))
        if not objects:
            scene_graph = parse_scene_graph_for_request(request.prompt)
            layout = compute_layout(scene_graph)
            objects = layout["objects"]

        generated = generate_object_images(
            request.prompt,
            objects,
            request.lora_scale,
        )
        contact_sheet = os.path.join(OUT_DIR, "objectwise_2d.png")
        create_object_contact_sheet(generated, contact_sheet)
        shutil.copy2(contact_sheet, os.path.join(KAGGLE_WORKING, "input.png"))
        images = [
            {
                "name": item["name"],
                "label": item["label"],
                "image_url": f"/object_images/{item['name']}.png",
                "detected_instances": item.get("detected_instances"),
                "sanitized": bool(item.get("sanitized", False)),
                "count_validation": item.get("count_validation", "unknown"),
                "segmentation_method": item.get("segmentation_method", "unknown"),
            }
            for item in generated
        ]
        return {
            "status": "success",
            "mode": "objectwise",
            "image_url": "/outputs/objectwise_2d.png",
            "images": images,
            "layout": layout,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/run_sam2")
def api_run_sam2(request: Sam2Request):
    try:
        source_mode = str(request.mode or "objectwise").lower()
        input_path = os.path.join(KAGGLE_WORKING, "input.png")
        if source_mode == "uploaded":
            upload_name = Path(urlparse(request.image_url).path).name
            input_path = os.path.join(UPLOADS_DIR, upload_name)
            if not upload_name or not os.path.isfile(input_path):
                raise FileNotFoundError("Uploaded image was not found on the backend")
        effective_layout = request.layout
        scene_graph = None

        if (
            not effective_layout.get("layout")
            and request.prompt.strip()
        ):
            scene_graph = parse_scene_graph_for_request(request.prompt.strip())
            effective_layout = compute_layout(scene_graph)

        crops = run_grounded_sam2(
            input_path,
            effective_layout,
            CROPS_DIR,
            source_mode=source_mode,
            auto_detect=request.auto_detect,
        )
        return {
            "status": "success",
            "mode": source_mode,
            "crops": crops,
            "scene_graph": scene_graph,
            "layout": effective_layout,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/generate_3d")
async def api_generate_3d(request: TrellisRequest):
    job_id = uuid.uuid4().hex
    with trellis_jobs_lock:
        trellis_jobs[job_id] = {"status": "queued"}

    await trellis_queue.put((job_id, request.crops))
    queue_position = get_trellis_queue_position(job_id)

    return {
        "status": "queued",
        "job_id": job_id,
        "queue_position": queue_position,
    }


@app.get("/api/generate_3d/status/{job_id}")
def api_generate_3d_status(job_id: str):
    with trellis_jobs_lock:
        job = trellis_jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="TRELLIS job not found")
        result = dict(job)

    if result.get("status") == "queued":
        result["queue_position"] = get_trellis_queue_position(job_id)
    return result


@app.post("/api/combine_scene")
def api_combine_scene(request: CombineRequest):
    try:
        scene_path = os.path.join(OUT_DIR, "scene_combined.glb")
        if not combine_scene_meshes(
            request.models,
            scene_path,
            request.layout,
            request.scale_factor,
        ):
            raise RuntimeError("Scene combining failed")

        zip_path = os.path.join(OUT_DIR, "scene_assets.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
            input_path = os.path.join(KAGGLE_WORKING, "input.png")
            if os.path.exists(input_path):
                archive.write(input_path, "input_image.png")

            for directory, archive_dir in (
                (CROPS_DIR, "crops"),
                (MULTI_GLB_DIR, "models"),
            ):
                for filename in os.listdir(directory):
                    file_path = os.path.join(directory, filename)
                    if os.path.isfile(file_path):
                        archive.write(
                            file_path,
                            os.path.join(archive_dir, filename),
                        )

            archive.write(scene_path, "scene_combined.glb")

        return {
            "status": "success",
            "scene_url": "/outputs/scene_combined.glb",
            "zip_url": "/outputs/scene_assets.zip",
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
