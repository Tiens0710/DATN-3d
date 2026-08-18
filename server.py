import os
import asyncio
import base64
import io
import importlib.util
import json
import hmac
import re
import shutil
import threading
import time
import uuid
import zipfile
from pathlib import Path
from urllib import request as urllib_request
from urllib.parse import urlparse
from typing import Dict, List, Any

import requests
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from src.parser import parse_scene_graph, parse_scene_graph_from_objects
from src.layout import compute_layout
from src.generator_2d import create_object_contact_sheet, generate_object_images
from src.segmenter import run_grounded_sam2
from src.generator_3d import generate_3d_models, generate_3d_variant
from src.combiner import combine_scene_meshes
from src.materials import patch_glb_materials


KAGGLE_WORKING = "/kaggle/working"
RUNS_DIR = os.path.join(KAGGLE_WORKING, "runs")
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
os.makedirs(RUNS_DIR, exist_ok=True)

app = FastAPI(
    title="DATN 3D Scene Reconstruction API",
    version="1.4.0",
)

BACKEND_BUILD = "gemini-metric-layout-v22-preserve-multi-object-inventory"

allowed_origins = [
    origin.strip()
    for origin in os.environ.get("ALLOWED_ORIGINS", "*").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=allowed_origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

APP_API_KEY = os.environ.get("APP_API_KEY", "").strip()


@app.middleware("http")
async def protect_expensive_api(request: Request, call_next):
    if (
        APP_API_KEY
        and request.method != "OPTIONS"
        and request.url.path.startswith("/api/")
        and request.url.path != "/api/health"
    ):
        supplied = request.headers.get("x-api-key", "")
        if not hmac.compare_digest(supplied, APP_API_KEY):
            return JSONResponse(status_code=401, content={"detail": "Invalid API key"})
    return await call_next(request)

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
app.mount("/runs", StaticFiles(directory=RUNS_DIR), name="runs")


class TextPrompt(BaseModel):
    text: str


class PromptOptimizeRequest(BaseModel):
    prompt: str


class ImageGenRequest(BaseModel):
    run_id: str
    prompt: str
    layout: Dict[str, Any]
    lora_scale: float = 0.2


class Sam2Request(BaseModel):
    run_id: str
    image_url: str
    layout: Dict[str, Any]
    prompt: str = ""
    mode: str = "objectwise"
    auto_detect: bool = False


class TrellisRequest(BaseModel):
    run_id: str
    crops: List[Dict[str, Any]]


class CombineRequest(BaseModel):
    run_id: str
    models: List[Dict[str, Any]]
    layout: Dict[str, Any]
    scale_factor: float = 0.01


class MaterialEditRequest(BaseModel):
    run_id: str
    source_file: str = "outputs/scene_combined.glb"
    materials: List[Dict[str, Any]]


class TrellisVariantRequest(BaseModel):
    run_id: str
    source_file: str
    prompt: str
    name: str = ""


trellis_jobs: Dict[str, Dict[str, Any]] = {}
trellis_jobs_lock = threading.Lock()
cancelled_runs: set[str] = set()
# One worker processes GPU-heavy TRELLIS jobs in FIFO order.
# Blocking inference runs outside the event loop via asyncio.to_thread().
trellis_queue = asyncio.Queue()
trellis_worker_task = None
MAX_TRELLIS_QUEUE = int(os.environ.get("MAX_TRELLIS_QUEUE", "4"))
JOB_TTL_SECONDS = int(os.environ.get("JOB_TTL_SECONDS", "21600"))
RUN_TTL_SECONDS = int(os.environ.get("RUN_TTL_SECONDS", "21600"))
MAX_STORED_RUNS = int(os.environ.get("MAX_STORED_RUNS", "20"))

RUN_ID_PATTERN = re.compile(r"^[a-f0-9]{32}$")


def _validate_run_id(run_id: str) -> str:
    clean = str(run_id or "").strip().lower()
    if not RUN_ID_PATTERN.fullmatch(clean):
        raise HTTPException(status_code=400, detail="Invalid run_id")
    return clean


def _is_run_cancelled(run_id: str) -> bool:
    clean = _validate_run_id(run_id)
    with trellis_jobs_lock:
        return clean in cancelled_runs


def _raise_if_run_cancelled(run_id: str) -> None:
    if _is_run_cancelled(run_id):
        raise HTTPException(status_code=409, detail="Pipeline run was cancelled")


def _run_paths(run_id: str, create: bool = False) -> Dict[str, Path]:
    clean = _validate_run_id(run_id)
    root = (Path(RUNS_DIR) / clean).resolve()
    runs_root = Path(RUNS_DIR).resolve()
    if runs_root not in root.parents:
        raise HTTPException(status_code=400, detail="Invalid run path")
    paths = {
        "root": root,
        "input": root / "input.png",
        "objects": root / "object_images",
        "manifest": root / "object_images_manifest.json",
        "crops": root / "crops",
        "models": root / "models",
        "outputs": root / "outputs",
    }
    if create:
        for key in ("root", "objects", "crops", "models", "outputs"):
            paths[key].mkdir(parents=True, exist_ok=True)
    elif not root.is_dir():
        raise HTTPException(status_code=404, detail="Pipeline run not found")
    return paths


def _write_run_metadata(paths: Dict[str, Path], filename: str, payload: Dict[str, Any]) -> None:
    metadata_path = paths["root"] / filename
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, default=str)


def _create_full_run_archive(paths: Dict[str, Path]) -> Path:
    """Package every useful artifact from one isolated pipeline run."""
    archive_path = paths["outputs"] / "pipeline_full_results.zip"
    archive_resolved = archive_path.resolve()
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for file_path in sorted(paths["root"].rglob("*")):
            if not file_path.is_file() or file_path.resolve() == archive_resolved:
                continue
            archive.write(
                file_path,
                file_path.relative_to(paths["root"]).as_posix(),
            )
    return archive_path


def _active_run_ids() -> set[str]:
    with trellis_jobs_lock:
        return {
            str(job.get("run_id"))
            for job in trellis_jobs.values()
            if job.get("status") in {"queued", "running"}
        }


def _cleanup_state() -> None:
    now = time.time()
    with trellis_jobs_lock:
        expired = [
            job_id
            for job_id, job in trellis_jobs.items()
            if now - float(job.get("updated_at", job.get("created_at", now))) > JOB_TTL_SECONDS
        ]
        for job_id in expired:
            trellis_jobs.pop(job_id, None)

    active = _active_run_ids()
    run_dirs = sorted(
        (path for path in Path(RUNS_DIR).iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for index, path in enumerate(run_dirs):
        if path.name in active:
            continue
        expired_by_age = now - path.stat().st_mtime > RUN_TTL_SECONDS
        expired_by_count = index >= MAX_STORED_RUNS
        if expired_by_age or expired_by_count:
            shutil.rmtree(path, ignore_errors=True)

GEMINI_MODEL = os.environ.get(
    "GEMINI_MODEL",
    "gemini-3.1-flash-lite",
)

PROMPT_OPTIMIZER_INSTRUCTION = """
Rewrite the user's request as one concise English prompt for Stable Diffusion
3.5 Medium. Return only the prompt, no explanation, and keep it under 55 words.
Start with the exact inventory: "Exactly one ..." for every requested object.
Preserve count, identity, material, color, shape, style, relation, and any
explicitly requested carving or pattern. Never add objects, props, ornaments,
or styles. Use conventional, physically plausible construction.
Every object must be separate, fully visible from top to floor, and minimally
overlapping. A generic dining table has one rectangular horizontal top, an
attached apron, and four straight connected legs. A generic chair has one seat,
one backrest, connected supports, and four legs. Use an eye-level front
three-quarter product view, white studio background, soft even lighting, and
realistic proportions. Never request top-down, close-up, cropped, surreal,
sculptural, floating, fused, or duplicated furniture. Translate Vietnamese to
natural English.
""".strip()

SCENE_GRAPH_INSTRUCTION = """
Extract the exact object plan from the user's prompt for an image-to-3D
pipeline. Return JSON only, with this schema:
{"objects":[{"label":"singular concrete object name in English", "count":1,
"description":"short description of this object"}],
"relation":"single|next_to|on_top_of|under|in_front_of|behind|left_of|right_of",
"relations":[{"subject":0,"relation":"next_to","object":1}],
"placements":[{"object":0,"position_xyz":[0.0,0.0,0.0],
"rotation_y_degrees":0.0}]}

Rules:
- Act as a practical interior designer and 3D scene planner. You decide the
  most realistic relative placement from the user's wording and normal object
  function; do not merely preserve the order in which nouns appear.
- Recognize ANY concrete object, not only furniture. Keep labels such as
  bicycle, house, toy, car, lamp, plant, cup, or any other object requested.
- Preserve the exact requested quantity. "two chairs" means count 2; do not
  count a reference such as "the table" after "beside the table" again.
- Do not turn parts, materials, colors, or decorative motifs into objects.
  For example, bicycle wheels and table legs are parts, not separate objects.
- Use one object entry per object category and put repeated quantity in count.
- Keep the user's material, color, style, and distinguishing details in
  description. Do not invent props or objects.
- Relations use zero-based indexes into objects. For multiple objects, return
  enough relations to connect every object into one practical scene. Preserve
  explicit spatial instructions from the user. When no position is stated,
  infer a normal functional layout instead of putting everything in a row:
  a coffee table is in_front_of a sofa, a floor lamp or plant is next_to the
  sofa, a nightstand is next_to a bed, a chair is next_to a dining table, and
  a small table lamp can be on_top_of a table or nightstand. Use an empty
  relations list only for a single object.
- Express the relation from the movable/accessory object to its anchor. Example:
  for sofa + coffee table + floor lamp, return table in_front_of sofa and lamp
  right_of sofa.
- Every object in a multi-object request must be connected to the scene by at
  least one relation. Prefer one large stable object such as sofa, bed, or table
  as the anchor. Do not use next_to for every edge unless the user explicitly
  asks for a row. For a living room, a valid answer is:
  {"objects":[{"label":"sofa","count":1,"description":"sofa"},
  {"label":"coffee table","count":1,"description":"coffee table"},
  {"label":"floor lamp","count":1,"description":"floor lamp"}],
  "relation":"in_front_of","relations":[
  {"subject":1,"relation":"in_front_of","object":0},
  {"subject":2,"relation":"right_of","object":0}]}
- You are the authority for 3D placement. Return one placement for every object
  entry. Coordinates are metres in a Y-up scene: X is left/right, Y is height,
  and Z is front/back. Put the main anchor at [0,0,0]. Negative Z is in front
  of the anchor and positive Z is behind it. Floor-standing objects use Y=0.
  Keep a normal room compact, generally within X/Z = -3 to 3 metres. Use
  rotation_y_degrees to make interacting objects face naturally. Do not overlap
  solid objects. For a sofa, coffee table and floor lamp, a practical example is
  sofa [0,0,0], table [0,0,-1.1], and lamp [1.3,0,0].
""".strip()

IMAGE_ANALYSIS_INSTRUCTION = """
Inspect the uploaded image for an image-to-3D pipeline. Return JSON only:
{"reconstructable":true,"reason":"short reason","objects":[
{"label":"singular concrete English object label","count":1,
"description":"short visual description"}]}

Rules:
- Mark reconstructable true only when the image clearly shows one or more
  complete physical objects with enough visible shape for single-image 3D.
- Mark false for drawings, logos, text, faces or people, flat artwork,
  screenshots, textures, severe crops, heavy occlusion, or unclear subjects.
- Include only prominent foreground objects intended for reconstruction.
- Do not count object parts, shadows, reflections, or background surfaces.
- Preserve repeated object quantity in count. Use concise GroundingDINO labels.
""".strip()


def _contains_prompt_term(text: str, terms: tuple[str, ...]) -> bool:
    """Match complete words/phrases, never substrings inside another word.

    The old ``"bàn" in source`` test interpreted ``ghế bành`` as containing a
    table because the letters ``bàn`` are the prefix of ``bành``.
    """
    return any(
        re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text, re.IGNORECASE)
        for term in terms
    )


def _is_strict_single_object_prompt(prompt: str) -> bool:
    """Recognize a true one-object inventory, not a no-extra-objects clause."""
    normalized = " ".join(str(prompt).lower().split())
    exactly_one_mentions = len(
        re.findall(r"(?<!\w)exactly\s+one(?!\w)", normalized)
    )
    return (
        exactly_one_mentions == 1
        and any(
            marker in normalized
            for marker in (
                "no other object",
                "no extra object",
                "without other object",
                "standalone",
            )
        )
    )


def _ensure_furniture_details(source_prompt: str, optimized_prompt: str) -> str:
    """Append short geometry guards without bloating the CLIP prompt."""
    source = source_prompt.lower()
    guards = []

    if _contains_prompt_term(source, ("table", "tables", "bàn", "ban")):
        guards.append("complete rectangular table with four connected legs visible")
    # In Vietnamese, "ghế sofa" contains the generic word "ghế" but it is not
    # a separate chair. Remove sofa phrases before checking chair markers.
    chair_source = re.sub(r"\b(?:ghế|ghe)\s+sofa\b", "sofa", source)
    if _contains_prompt_term(chair_source, ("chair", "chairs", "ghế", "ghe")):
        guards.append("complete chair with seat, backrest, and all legs visible")
    if _contains_prompt_term(source, ("sofa", "sofas", "couch", "couches")):
        guards.append("complete sofa with arms, base, and feet visible")
    if _contains_prompt_term(
        source,
        ("floor lamp", "floor lamps", "standing lamp", "đèn sàn", "den san"),
    ):
        guards.append("complete floor lamp with shade, thin stem, and base visible")
    bed_source = re.sub(
        r"(?<!\w)(?:bedside table|nightstand|tủ đầu giường|tu dau giuong)(?!\w)",
        "nightstand",
        source,
        flags=re.IGNORECASE,
    )
    if _contains_prompt_term(bed_source, ("bed", "beds", "giường", "giuong")):
        guards.append("complete bed with a thick mattress visibly covering the frame")
    if _contains_prompt_term(
        source,
        ("nightstand", "nightstands", "bedside table", "tủ đầu giường", "tu dau giuong"),
    ):
        guards.append("short compact bedside nightstand, not a tall dresser")

    detail_markers = (
        "carv", "ornate", "decor", "inlay", "engrave", "pattern", "motif",
        "floral", "cham", "chạm", "hoa van", "hoa văn", "họa tiết",
    )
    if any(marker in source for marker in detail_markers):
        guards.append("requested decoration clearly visible but structurally restrained")

    if not guards:
        return optimized_prompt

    return optimized_prompt.rstrip(" .") + ", " + ", ".join(guards) + "."


def optimize_prompt_with_gemini(prompt: str) -> Dict[str, Any]:
    clean_prompt = " ".join(prompt.split()).strip()
    if not clean_prompt:
        raise ValueError("Prompt must not be empty")

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return {
            "optimized_prompt": (
                f"{clean_prompt}, photorealistic product photography, "
                "full objects visible from top to floor, conventional accurate proportions and construction, "
                "eye-level three-quarter view, never overhead or top-down, centered composition, clean neutral studio "
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
        optimized_prompt = _ensure_furniture_details(clean_prompt, optimized_prompt)

        return {
            "optimized_prompt": optimized_prompt,
            "used_gemini": True,
            "model": GEMINI_MODEL,
        }
    except Exception as exc:
        return {
            "optimized_prompt": (
                f"{clean_prompt}, photorealistic product photography, "
                "full objects visible from top to floor, conventional accurate proportions and construction, "
                "eye-level three-quarter view, never overhead or top-down, centered composition, clean neutral studio "
                "background, soft even lighting, sharp focus, realistic materials, "
                "minimal overlap, no people, no text, no watermark, no extra objects, "
                "suitable for object segmentation and single-image 3D reconstruction"
            ),
            "used_gemini": False,
            "warning": f"Gemini optimization failed: {exc}",
            "model": GEMINI_MODEL,
        }


def analyze_uploaded_image_with_gemini(image_path: str) -> Dict[str, Any]:
    """Identify reconstructable foreground objects before segmentation."""
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is required for uploaded-image analysis")

    with Image.open(image_path) as source:
        image = source.convert("RGB")
        image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=88, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")

    endpoint = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent"
    )
    payload = {
        "systemInstruction": {
            "parts": [{"text": IMAGE_ANALYSIS_INSTRUCTION}]
        },
        "contents": [{
            "role": "user",
            "parts": [
                {"text": "Analyze this exact uploaded image."},
                {"inlineData": {"mimeType": "image/jpeg", "data": encoded}},
            ],
        }],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 500,
            "responseMimeType": "application/json",
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
    with urllib_request.urlopen(http_request, timeout=60) as response:
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
    reconstructable = bool(data.get("reconstructable"))
    objects = data.get("objects") if isinstance(data.get("objects"), list) else []
    clean_objects = []
    for item in objects[:8]:
        if not isinstance(item, dict):
            continue
        label = " ".join(str(item.get("label", "")).split()).strip(" ,.;:").lower()
        if not label:
            continue
        try:
            count = int(item.get("count", 1))
        except (TypeError, ValueError):
            count = 1
        clean_objects.append({
            "label": label,
            "count": max(1, min(count, 6)),
            "description": " ".join(
                str(item.get("description", label)).split()
            ).strip(),
        })
    if reconstructable and not clean_objects:
        raise ValueError("Gemini found no usable foreground object labels")
    return {
        "reconstructable": reconstructable,
        "reason": " ".join(str(data.get("reason", "")).split()).strip(),
        "objects": clean_objects,
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
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 700},
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

    # Respect an explicit single-object request even if the structured model
    # hallucinates a prop. Prefer the label mentioned in the first inventory
    # sentence, then fall back to Gemini's first object.
    prompt_lower = prompt.lower()
    strict_single = _is_strict_single_object_prompt(prompt_lower)
    if strict_single and len(clean_objects) > 1:
        inventory_sentence = re.split(r"[.;]", prompt_lower, maxsplit=1)[0]
        selected_index = next(
            (
                index
                for index, item in enumerate(clean_objects)
                if str(item.get("label", "")).lower() in inventory_sentence
            ),
            0,
        )
        clean_objects = [clean_objects[selected_index]]
        data["relations"] = []
        data["relation"] = "single"
        data["placements"] = [
            {"object": 0, "position_xyz": [0.0, 0.0, 0.0], "rotation_y_degrees": 0.0}
        ]
    return parse_scene_graph_from_objects(
        prompt,
        clean_objects,
        relation=data.get("relation"),
        relations=data.get("relations", []),
        placements=data.get("placements", []),
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

    worker_urls = {
        "sd35": os.environ.get("SD35_WORKER_URL", "http://127.0.0.1:8001").rstrip("/"),
        "trellis": os.environ.get("TRELLIS_WORKER_URL", "http://127.0.0.1:8002").rstrip("/"),
        "sam2_dino": os.environ.get("SAM2_DINO_WORKER_URL", "http://127.0.0.1:8003").rstrip("/"),
    }
    workers = {}
    for name, url in worker_urls.items():
        try:
            response = requests.get(f"{url}/health", timeout=2)
            response.raise_for_status()
            payload = response.json()
            workers[name] = {
                "ready": bool(payload.get("ready")),
                "error": payload.get("error"),
                "device": payload.get("device"),
                "lora_loaded": payload.get("lora_loaded") if name == "sd35" else None,
                "lora_scale": payload.get("lora_scale") if name == "sd35" else None,
            }
        except Exception as exc:
            workers[name] = {"ready": False, "error": str(exc)}

    checks = {
        "hf_token": bool(os.environ.get("HF_TOKEN", "").strip()),
        "gemini_api_key": bool(
            os.environ.get("GEMINI_API_KEY", "").strip()
        ),
        "cuda": cuda_available,
        "files": files,
        "modules": modules,
        "workers": workers,
        "api_auth": bool(APP_API_KEY),
    }
    ready = (
        checks["hf_token"]
        and checks["cuda"]
        and all(files.values())
        and all(modules.values())
        and all(worker.get("ready") for worker in workers.values())
    )
    return {
        "ready": ready,
        "checks": checks,
        "cuda_name": cuda_name,
    }


def run_trellis_job(job_id: str, run_id: str, crops: List[Dict[str, Any]]) -> None:
    with trellis_jobs_lock:
        if run_id in cancelled_runs:
            trellis_jobs[job_id]["status"] = "cancelled"
            trellis_jobs[job_id]["updated_at"] = time.time()
            return
        trellis_jobs[job_id]["status"] = "running"
        trellis_jobs[job_id]["updated_at"] = time.time()

    try:
        paths = _run_paths(run_id)
        models = generate_3d_models(
            crops,
            str(paths["models"]),
            allowed_root=str(paths["root"]),
            public_prefix=f"/runs/{run_id}/models",
        )
        with trellis_jobs_lock:
            trellis_jobs[job_id] = {
                "status": "cancelled" if run_id in cancelled_runs else "completed",
                "models": [] if run_id in cancelled_runs else models,
                "run_id": run_id,
                "updated_at": time.time(),
            }
    except Exception as exc:
        with trellis_jobs_lock:
            trellis_jobs[job_id] = {
                "status": "cancelled" if run_id in cancelled_runs else "failed",
                "error": str(exc),
                "run_id": run_id,
                "updated_at": time.time(),
            }


def run_trellis_variant_job(
    job_id: str,
    run_id: str,
    source_path: str,
    prompt: str,
    name: str,
) -> None:
    with trellis_jobs_lock:
        if run_id in cancelled_runs:
            trellis_jobs[job_id]["status"] = "cancelled"
            trellis_jobs[job_id]["updated_at"] = time.time()
            return
        trellis_jobs[job_id]["status"] = "running"
        trellis_jobs[job_id]["updated_at"] = time.time()

    try:
        paths = _run_paths(run_id)
        model = generate_3d_variant(
            source_path,
            prompt,
            str(paths["models"]),
            name,
            allowed_root=str(paths["root"]),
            public_prefix=f"/runs/{run_id}/models",
        )
        source_relative = Path(source_path).resolve().relative_to(
            paths["root"].resolve()
        ).as_posix()
        _write_run_metadata(
            paths,
            f"variant_{name}.json",
            {
                "run_id": run_id,
                "source_file": source_relative,
                "prompt": prompt,
                "model": model,
            },
        )
        zip_path = _create_full_run_archive(paths)
        with trellis_jobs_lock:
            trellis_jobs[job_id] = {
                "status": "cancelled" if run_id in cancelled_runs else "completed",
                "models": [] if run_id in cancelled_runs else [model],
                "run_id": run_id,
                "kind": "variant",
                "zip_url": f"/runs/{run_id}/outputs/{zip_path.name}",
                "updated_at": time.time(),
            }
    except Exception as exc:
        with trellis_jobs_lock:
            trellis_jobs[job_id] = {
                "status": "cancelled" if run_id in cancelled_runs else "failed",
                "error": str(exc),
                "run_id": run_id,
                "kind": "variant",
                "updated_at": time.time(),
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
        job_id, run_id, payload = await trellis_queue.get()
        try:
            if payload.get("kind") == "variant":
                await asyncio.to_thread(
                    run_trellis_variant_job,
                    job_id,
                    run_id,
                    payload["source_path"],
                    payload["prompt"],
                    payload["name"],
                )
            else:
                await asyncio.to_thread(
                    run_trellis_job,
                    job_id,
                    run_id,
                    payload.get("crops", []),
                )
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
        "version": "1.4.0",
        "build": BACKEND_BUILD,
        **readiness,
        "gemini_model": GEMINI_MODEL,
        "gemini_configured": readiness["checks"]["gemini_api_key"],
        "auth_enabled": bool(APP_API_KEY),
    }


@app.post("/api/runs")
def api_create_run():
    _cleanup_state()
    run_id = uuid.uuid4().hex
    with trellis_jobs_lock:
        cancelled_runs.discard(run_id)
    paths = _run_paths(run_id, create=True)
    (paths["root"] / "run.json").write_text(
        json.dumps({"run_id": run_id, "created_at": time.time()}, indent=2),
        encoding="utf-8",
    )
    return {"status": "created", "run_id": run_id}


@app.delete("/api/runs/{run_id}")
def api_delete_run(run_id: str):
    """Delete a finished pipeline run before starting a fresh one."""
    clean_run_id = _validate_run_id(run_id)
    run_path = (Path(RUNS_DIR) / clean_run_id).resolve()
    runs_root = Path(RUNS_DIR).resolve()
    if runs_root not in run_path.parents:
        raise HTTPException(status_code=400, detail="Invalid run path")
    if not run_path.is_dir():
        return {"status": "already_deleted", "run_id": clean_run_id}

    with trellis_jobs_lock:
        was_cancelled = clean_run_id in cancelled_runs
        active_jobs = [
            job_id
            for job_id, job in trellis_jobs.items()
            if job.get("run_id") == clean_run_id
            and job.get("status") in {"queued", "running"}
        ]
        if active_jobs and not was_cancelled:
            raise HTTPException(
                status_code=409,
                detail="Cannot delete a run while its TRELLIS job is still active",
            )
        if active_jobs and was_cancelled:
            return {"status": "deletion_deferred", "run_id": clean_run_id}
        for job_id, job in list(trellis_jobs.items()):
            if job.get("run_id") == clean_run_id:
                trellis_jobs.pop(job_id, None)

    shutil.rmtree(run_path, ignore_errors=False)
    with trellis_jobs_lock:
        cancelled_runs.discard(clean_run_id)
    return {"status": "deleted", "run_id": clean_run_id}


@app.post("/api/runs/{run_id}/cancel")
def api_cancel_run(run_id: str):
    """Mark a pipeline run as cancelled and prevent queued work from starting."""
    clean_run_id = _validate_run_id(run_id)
    run_path = (Path(RUNS_DIR) / clean_run_id).resolve()
    runs_root = Path(RUNS_DIR).resolve()
    if runs_root not in run_path.parents:
        raise HTTPException(status_code=400, detail="Invalid run path")
    if not run_path.is_dir():
        return {"status": "already_deleted", "run_id": clean_run_id}

    now = time.time()
    with trellis_jobs_lock:
        cancelled_runs.add(clean_run_id)
        for job in trellis_jobs.values():
            if job.get("run_id") == clean_run_id and job.get("status") == "queued":
                job["status"] = "cancelled"
                job["updated_at"] = now
    return {"status": "cancelled", "run_id": clean_run_id}


@app.post("/api/optimize_prompt")
def api_optimize_prompt(request: PromptOptimizeRequest):
    try:
        return optimize_prompt_with_gemini(request.prompt)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/upload_image")
async def api_upload_image(
    run_id: str = Form(...),
    file: UploadFile = File(...),
):
    """Store a user image for the upload-to-3D branch of the pipeline."""
    _raise_if_run_cancelled(run_id)
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Only image files are supported")

    raw = await file.read(20 * 1024 * 1024 + 1)
    if len(raw) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image must be 20 MB or smaller")

    paths = _run_paths(run_id)
    try:
        with Image.open(io.BytesIO(raw)) as image:
            image.verify()
        with Image.open(io.BytesIO(raw)) as image:
            width, height = image.size
            if width < 32 or height < 32 or width * height > 20_000_000:
                raise ValueError("Image dimensions must be at least 32px and at most 20 megapixels")
            # Preserve a real alpha channel from PNG/WebP uploads.  Flattening
            # it to RGB here discards an already-correct foreground mask and
            # forces GroundingDINO/SAM2 to rediscover the object against the
            # original background.
            has_alpha = (
                ("A" in image.getbands() or "transparency" in image.info)
                and image.convert("RGBA").getchannel("A").getextrema() != (255, 255)
            )
            upload_image = image.convert("RGBA" if has_alpha else "RGB")
            filename = f"upload_{uuid.uuid4().hex}.png"
            output_path = paths["root"] / filename
            upload_image.save(output_path, format="PNG")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}")

    # Keep the legacy path available for archive/download tooling.
    shutil.copy2(output_path, paths["input"])
    return {
        "status": "success",
        "mode": "uploaded",
        "run_id": run_id,
        "image_url": f"/runs/{run_id}/{filename}",
        "filename": filename,
        "width": width,
        "height": height,
    }


@app.post("/api/parse_scene_graph")
def api_parse_scene_graph(request: TextPrompt):
    try:
        return parse_scene_graph_for_request(request.text)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/generate_layout")
def api_generate_layout(scene_graph: Dict[str, Any]):
    try:
        return compute_layout(scene_graph)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/generate_image")
def api_generate_image(request: ImageGenRequest):
    try:
        _raise_if_run_cancelled(request.run_id)
        paths = _run_paths(request.run_id)
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
            object_image_dir=str(paths["objects"]),
            manifest_path=str(paths["manifest"]),
        )
        _raise_if_run_cancelled(request.run_id)
        contact_sheet = paths["outputs"] / "objectwise_2d.png"
        create_object_contact_sheet(generated, str(contact_sheet))
        shutil.copy2(contact_sheet, paths["input"])
        _write_run_metadata(
            paths,
            "generation_metadata.json",
            {
                "run_id": request.run_id,
                "prompt": request.prompt,
                "layout": layout,
                "lora_scale": request.lora_scale,
                "objects": generated,
            },
        )
        images = [
            {
                "name": item["name"],
                "label": item["label"],
                "image_url": (
                    f"/runs/{request.run_id}/object_images/raw/{item['name']}.png"
                ),
                "sanitized_image_url": (
                    f"/runs/{request.run_id}/object_images/sanitized/{item['name']}.png"
                ),
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
            "run_id": request.run_id,
            "image_url": f"/runs/{request.run_id}/outputs/objectwise_2d.png",
            "images": images,
            "layout": layout,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/run_sam2")
def api_run_sam2(request: Sam2Request):
    try:
        _raise_if_run_cancelled(request.run_id)
        paths = _run_paths(request.run_id)
        source_mode = str(request.mode or "objectwise").lower()
        input_path = str(paths["input"])
        image_analysis = None
        worker_auto_detect = request.auto_detect
        if source_mode == "uploaded":
            upload_name = Path(urlparse(request.image_url).path).name
            upload_path = (paths["root"] / upload_name).resolve()
            if paths["root"] not in upload_path.parents or not upload_path.is_file():
                raise FileNotFoundError("Uploaded image was not found on the backend")
            input_path = str(upload_path)
        effective_layout = request.layout
        scene_graph = None
        warnings = []

        if source_mode == "uploaded" and request.auto_detect:
            try:
                image_analysis = analyze_uploaded_image_with_gemini(input_path)
            except Exception as exc:
                raise HTTPException(
                    status_code=502,
                    detail=f"Gemini could not analyze the uploaded image: {exc}",
                ) from exc
            if not image_analysis["reconstructable"]:
                reason = image_analysis.get("reason") or "the image is not suitable for single-image 3D"
                warning = (
                    "Gemini đánh giá ảnh upload có rủi ro cho dựng 3D: "
                    f"{reason}. Hệ thống vẫn tiếp tục theo yêu cầu của người dùng; "
                    "kết quả phụ thuộc vào chất lượng cắt nền."
                )
                warnings.append(warning)
                image_analysis["warning"] = warning
                image_analysis["proceeding_despite_risk"] = True

            # Gemini is advisory for user-provided images.  When it can name
            # objects, use those labels even if it flags a render/illustration.
            # If it cannot name anything, leave auto-detection enabled so DINO
            # gets a chance to find the foreground instead of hard-rejecting.
            if image_analysis["objects"]:
                scene_graph = parse_scene_graph_from_objects(
                    request.prompt.strip() or "uploaded physical objects",
                    image_analysis["objects"],
                    parser_source="gemini_image_analysis",
                )
                effective_layout = compute_layout(scene_graph)
                worker_auto_detect = False
            else:
                worker_auto_detect = True

        if (
            not effective_layout.get("layout")
            and request.prompt.strip()
        ):
            scene_graph = parse_scene_graph_for_request(request.prompt.strip())
            effective_layout = compute_layout(scene_graph)

        crops = run_grounded_sam2(
            input_path,
            effective_layout,
            str(paths["crops"]),
            source_mode=source_mode,
            auto_detect=worker_auto_detect,
            manifest_path=str(paths["manifest"]),
        )
        _raise_if_run_cancelled(request.run_id)
        if source_mode == "uploaded":
            failed = [
                crop.get("label", crop.get("name", "object"))
                for crop in crops
                if crop.get("detector_fallback")
            ]
            if failed:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "GroundingDINO could not locate the requested object(s): "
                        + ", ".join(map(str, failed))
                        + ". Use a clearer image or enter a more precise label in the SAM2 node."
                    ),
                )
        for crop in crops:
            crop_path = Path(str(crop.get("crop_path", ""))).resolve()
            if paths["crops"] not in crop_path.parents or not crop_path.is_file():
                raise RuntimeError("Segmentation worker returned a crop outside the active run")
            crop["crop_path"] = str(crop_path)
            crop["crop_url"] = f"/runs/{request.run_id}/crops/{crop_path.name}"
        _write_run_metadata(
            paths,
            "segmentation_metadata.json",
            {
                "run_id": request.run_id,
                "mode": source_mode,
                "prompt": request.prompt,
                "layout": effective_layout,
                "image_analysis": image_analysis,
                "warnings": warnings,
                "crops": crops,
            },
        )
        return {
            "status": "success",
            "mode": source_mode,
            "crops": crops,
            "scene_graph": scene_graph,
            "layout": effective_layout,
            "image_analysis": image_analysis,
            "warnings": warnings,
            "run_id": request.run_id,
            "preview_url": f"/runs/{request.run_id}/crops/sam2_visual.png",
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/generate_3d")
async def api_generate_3d(request: TrellisRequest):
    _cleanup_state()
    _raise_if_run_cancelled(request.run_id)
    paths = _run_paths(request.run_id)
    if trellis_queue.qsize() >= MAX_TRELLIS_QUEUE:
        raise HTTPException(status_code=429, detail="TRELLIS queue is full; try again later")
    safe_crops = []
    for crop in request.crops:
        crop_path = Path(str(crop.get("crop_path", ""))).resolve()
        if paths["crops"] not in crop_path.parents or not crop_path.is_file():
            raise HTTPException(status_code=400, detail="Invalid crop path for this run")
        safe_crops.append(dict(crop, crop_path=str(crop_path)))
    job_id = uuid.uuid4().hex
    now = time.time()
    with trellis_jobs_lock:
        trellis_jobs[job_id] = {
            "status": "queued",
            "run_id": request.run_id,
            "kind": "generate",
            "created_at": now,
            "updated_at": now,
        }

    await trellis_queue.put(
        (job_id, request.run_id, {"kind": "generate", "crops": safe_crops})
    )
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


@app.post("/api/edit_3d_variant")
async def api_edit_3d_variant(request: TrellisVariantRequest):
    """Queue TRELLIS 1's official text-conditioned run_variant operation."""
    _cleanup_state()
    _raise_if_run_cancelled(request.run_id)
    paths = _run_paths(request.run_id)
    if trellis_queue.qsize() >= MAX_TRELLIS_QUEUE:
        raise HTTPException(status_code=429, detail="TRELLIS queue is full; try again later")

    prompt = " ".join(str(request.prompt or "").split()).strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Variant prompt must not be empty")
    if len(prompt) > 1000:
        raise HTTPException(status_code=400, detail="Variant prompt must be 1000 characters or fewer")

    source_name = str(request.source_file or "").replace("\\", "/").strip("/")
    source_path = (paths["root"] / source_name).resolve()
    run_root = paths["root"].resolve()
    if run_root not in source_path.parents or source_path.suffix.lower() != ".glb":
        raise HTTPException(status_code=400, detail="source_file must be a GLB inside this run")
    if not source_path.is_file():
        raise HTTPException(status_code=404, detail="Source GLB not found")

    requested_name = re.sub(r"[^A-Za-z0-9_-]+", "_", str(request.name or "").strip())
    if not requested_name:
        requested_name = f"{source_path.stem}_variant_{uuid.uuid4().hex[:8]}"
    requested_name = requested_name[:80].strip("_-")
    if not requested_name:
        raise HTTPException(status_code=400, detail="Invalid variant model name")

    job_id = uuid.uuid4().hex
    now = time.time()
    with trellis_jobs_lock:
        trellis_jobs[job_id] = {
            "status": "queued",
            "run_id": request.run_id,
            "kind": "variant",
            "source_file": source_path.relative_to(run_root).as_posix(),
            "prompt": prompt,
            "created_at": now,
            "updated_at": now,
        }

    await trellis_queue.put(
        (
            job_id,
            request.run_id,
            {
                "kind": "variant",
                "source_path": str(source_path),
                "prompt": prompt,
                "name": requested_name,
            },
        )
    )
    return {
        "status": "queued",
        "kind": "variant",
        "job_id": job_id,
        "queue_position": get_trellis_queue_position(job_id),
    }


@app.post("/api/combine_scene")
def api_combine_scene(request: CombineRequest):
    try:
        _raise_if_run_cancelled(request.run_id)
        paths = _run_paths(request.run_id)
        scene_path = paths["outputs"] / "scene_combined.glb"
        if not combine_scene_meshes(
            request.models,
            str(scene_path),
            request.layout,
            request.scale_factor,
            allowed_root=str(paths["models"]),
        ):
            raise RuntimeError("Scene combining failed")
        _raise_if_run_cancelled(request.run_id)

        _write_run_metadata(
            paths,
            "scene_metadata.json",
            {
                "run_id": request.run_id,
                "models": request.models,
                "layout": request.layout,
                "scale_factor": request.scale_factor,
                "scene_file": "outputs/scene_combined.glb",
            },
        )
        zip_path = _create_full_run_archive(paths)

        return {
            "status": "success",
            "run_id": request.run_id,
            "scene_url": f"/runs/{request.run_id}/outputs/scene_combined.glb",
            "zip_url": f"/runs/{request.run_id}/outputs/pipeline_full_results.zip",
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/edit_material")
def api_edit_material(request: MaterialEditRequest):
    """Edit GLB PBR factors without re-running the TRELLIS pipeline."""
    try:
        _raise_if_run_cancelled(request.run_id)
        paths = _run_paths(request.run_id)
        if not request.materials:
            raise HTTPException(status_code=400, detail="materials must not be empty")
        if len(request.materials) > 256:
            raise HTTPException(status_code=400, detail="Too many material entries")

        source_name = str(request.source_file or "").replace("\\", "/").strip("/")
        source_path = (paths["root"] / source_name).resolve()
        run_root = paths["root"].resolve()
        if run_root not in source_path.parents or source_path.suffix.lower() != ".glb":
            raise HTTPException(status_code=400, detail="source_file must be a GLB inside this run")
        if not source_path.is_file():
            raise HTTPException(status_code=404, detail="Source GLB not found")

        for index, material in enumerate(request.materials):
            if not isinstance(material, dict):
                raise HTTPException(status_code=400, detail=f"Material {index} must be an object")
            color = material.get("base_color", [1.0, 1.0, 1.0, 1.0])
            if not isinstance(color, list) or len(color) not in {3, 4}:
                raise HTTPException(
                    status_code=400,
                    detail=f"Material {index} base_color must contain 3 or 4 numbers",
                )
            try:
                values = [float(channel) for channel in color]
                roughness = float(material.get("roughness", 0.8))
                metallic = float(material.get("metallic", 0.0))
            except (TypeError, ValueError) as exc:
                raise HTTPException(status_code=400, detail=f"Invalid material {index}: {exc}") from exc
            if any(channel < 0.0 or channel > 1.0 for channel in values):
                raise HTTPException(status_code=400, detail=f"Material {index} color is out of range")
            if not 0.0 <= roughness <= 1.0 or not 0.0 <= metallic <= 1.0:
                raise HTTPException(
                    status_code=400,
                    detail=f"Material {index} roughness and metallic must be between 0 and 1",
                )

        output_path = paths["outputs"] / "scene_combined_material_edited.glb"
        updated_count = patch_glb_materials(
            source_path,
            output_path,
            request.materials,
        )
        _write_run_metadata(
            paths,
            "material_edit_metadata.json",
            {
                "run_id": request.run_id,
                "source_file": source_path.relative_to(run_root).as_posix(),
                "output_file": output_path.relative_to(run_root).as_posix(),
                "materials": request.materials,
            },
        )
        # Keep the edited GLB included when the user downloads the full run.
        zip_path = _create_full_run_archive(paths)
        return {
            "status": "success",
            "run_id": request.run_id,
            "updated_materials": updated_count,
            "glb_url": f"/runs/{request.run_id}/outputs/{output_path.name}",
            "zip_url": f"/runs/{request.run_id}/outputs/{zip_path.name}",
        }
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
