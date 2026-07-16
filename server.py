import os
import asyncio
import json
import shutil
import threading
import uuid
import zipfile
from urllib import request as urllib_request
from typing import Dict, List, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.parser import parse_scene_graph
from src.layout import compute_layout
from src.generator_2d import generate_2d_image
from src.segmenter import run_grounded_sam2
from src.generator_3d import generate_3d_models
from src.combiner import combine_scene_meshes


KAGGLE_WORKING = "/kaggle/working"
CROPS_DIR = os.path.join(KAGGLE_WORKING, "crops")
MULTI_GLB_DIR = os.path.join(KAGGLE_WORKING, "multi_object_glb")
OUT_DIR = os.path.join(KAGGLE_WORKING, "outputs", "trellis")

os.makedirs(CROPS_DIR, exist_ok=True)
os.makedirs(MULTI_GLB_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

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
text-to-2D-to-3D pipeline. Return exactly one English prompt and nothing else.
Preserve every requested object, object count, material, color, style, and
spatial relationship. Never invent extra furniture or decorative props. If the
input is Vietnamese, translate it naturally to English. Make each object fully
visible with accurate geometry, realistic construction, and minimal overlap.
Use a three-quarter product-photography view, centered composition, clean
neutral studio background, soft even lighting, sharp focus, and realistic
materials. The result must be suitable for GroundingDINO, SAM2 segmentation,
and single-image 3D reconstruction. Keep the prompt under 90 words.
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
                "three-quarter view, centered composition, clean neutral studio "
                "background, soft even lighting, sharp focus, realistic materials, "
                "minimal overlap, no people, no text, no watermark, no extra objects, "
                "suitable for object segmentation and single-image 3D reconstruction"
            ),
            "used_gemini": False,
            "warning": f"Gemini optimization failed: {exc}",
            "model": GEMINI_MODEL,
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
    return {
        "status": "ok",
        "version": "1.1.0",
        "gemini_model": GEMINI_MODEL,
        "gemini_configured": bool(os.environ.get("GEMINI_API_KEY", "").strip()),
    }


@app.post("/api/optimize_prompt")
def api_optimize_prompt(request: PromptOptimizeRequest):
    try:
        return optimize_prompt_with_gemini(request.prompt)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/parse_scene_graph")
def api_parse_scene_graph(request: TextPrompt):
    try:
        return parse_scene_graph(request.text)
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
        input_path = os.path.join(KAGGLE_WORKING, "input.png")
        if not generate_2d_image(request.prompt, request.lora_scale, input_path):
            raise RuntimeError("Image generation failed")

        output_path = os.path.join(OUT_DIR, "input_2d.png")
        shutil.copy2(input_path, output_path)
        return {"status": "success", "image_url": "/outputs/input_2d.png"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/run_sam2")
def api_run_sam2(request: Sam2Request):
    try:
        input_path = os.path.join(KAGGLE_WORKING, "input.png")
        effective_layout = request.layout
        scene_graph = None

        if (
            not effective_layout.get("layout")
            and request.prompt.strip()
        ):
            scene_graph = parse_scene_graph(request.prompt.strip())
            effective_layout = compute_layout(scene_graph)

        crops = run_grounded_sam2(
            input_path,
            effective_layout,
            CROPS_DIR,
        )
        return {
            "status": "success",
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
