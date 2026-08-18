import json
import hashlib
import os
import re
from pathlib import Path

import requests


OBJECT_IMAGE_DIR = "/kaggle/working/object_images"
OBJECT_MANIFEST = "/kaggle/working/object_images_manifest.json"
SD35_WORKER_URL = os.environ.get(
    "SD35_WORKER_URL",
    "http://127.0.0.1:8001",
).rstrip("/")
SD35_WORKER_TIMEOUT = float(os.environ.get("SD35_WORKER_TIMEOUT", "1500"))
TRELLIS_WORKER_URL = os.environ.get(
    "TRELLIS_WORKER_URL",
    "http://127.0.0.1:8002",
).rstrip("/")
SAM2_DINO_WORKER_URL = os.environ.get(
    "SAM2_DINO_WORKER_URL",
    "http://127.0.0.1:8003",
).rstrip("/")
OBJECT_SANITIZE_TIMEOUT = float(
    os.environ.get("OBJECT_SANITIZE_TIMEOUT", "600")
)
WORKER_OFFLOAD_TIMEOUT = float(
    os.environ.get("WORKER_OFFLOAD_TIMEOUT", "600")
)


_OBJECT_ALIASES = {
    "chair": ("dining chair", "wooden chair", "chair", "ghế", "ghe"),
    "table": ("dining table", "coffee table", "wooden table", "table", "bàn", "ban"),
    "sofa": ("sofa", "couch"),
    "bed": ("bed", "giường", "giuong"),
    "desk": ("desk",),
    "stool": ("stool",),
    "bench": ("bench",),
    "cabinet": ("cabinet", "tủ", "tu"),
    "wardrobe": ("wardrobe",),
    "bookshelf": ("bookshelf",),
    "shelf": ("shelf",),
    "lamp": ("floor lamp", "arc floor lamp", "standing lamp", "lamp", "đèn", "den"),
    "floor lamp": ("floor lamp", "arc floor lamp", "standing lamp", "lamp", "đèn sàn", "den san"),
    "ottoman": ("ottoman",),
    "nightstand": ("nightstand", "bedside table"),
    "dresser": ("dresser",),
}

_CLAUSE_BREAKS = re.compile(
    r"\s*(?:,|;|\band\b|\bwith\b|\bnext to\b|\bbeside\b|"
    r"\balongside\b|\bnear\b|\b(?:placed|arranged)?\s*side[- ]by[- ]side\b|"
    r"\bto the left of\b|\bto the right of\b|"
    r"\bto left of\b|\bto right of\b|\bleft of\b|\bright of\b|"
    r"\bbên cạnh\b|\bke ben\b|\bbên trái\b|\bbên phải\b|"
    r"\bphía trước\b|\bphía sau\b|\bđặt trên\b|\bdat tren\b|"
    r"\bon top of\b|\bplaced on\b|\bunderneath\b|\bunder\b|\bbelow\b|"
    r"\babove\b|\bbelow\b|\btrên\b|\btren\b|\bdưới\b|\bduoi\b)\s*",
    flags=re.IGNORECASE,
)

_GLOBAL_PROMPT_MARKERS = (
    "photorealistic",
    "product photography",
    "full objects visible",
    "clean neutral studio background",
    "clean white studio background",
    "neutral gray studio background",
    "soft even lighting",
    "sharp focus",
    "realistic materials",
)


def _object_descriptor(label: str, scene_prompt: str) -> str:
    """Extract only the target object's noun phrase from the full scene prompt."""
    text = " ".join(str(scene_prompt).split()).strip()
    lowered = text.lower()
    marker_positions = [
        lowered.find(marker)
        for marker in _GLOBAL_PROMPT_MARKERS
        if lowered.find(marker) >= 0
    ]
    if marker_positions:
        text = text[: min(marker_positions)].rstrip(" ,.;:")

    aliases = _OBJECT_ALIASES.get(label, (label,))
    alias_pattern = re.compile(
        r"(?<!\w)(?:" + "|".join(re.escape(alias) for alias in aliases) + r")(?!\w)",
        flags=re.IGNORECASE,
    )
    clauses = [clause.strip(" ,.;:") for clause in _CLAUSE_BREAKS.split(text)]
    for clause in clauses:
        if alias_pattern.search(clause):
            descriptor = re.sub(
                r"^(?:a|an|one|the|exactly one)\s+",
                "",
                clause,
                flags=re.IGNORECASE,
            ).strip(" ,.;:")
            return descriptor or label
    return label


def _generation_attempt(item: dict) -> int:
    """Return a backward-compatible zero-based diffusion attempt number."""
    if "generation_attempt" not in item:
        return 1 if item.get("retry_clean_background") else 0
    try:
        return max(0, int(item.get("generation_attempt", 0)))
    except (TypeError, ValueError):
        return 1 if item.get("retry_clean_background") else 0


def _isolation_background(label: str, attempt: int | bool) -> str:
    """Choose a stable, contrasting studio colour for every object image.

    Object-wise images are intermediate assets, not the final scene.  A plain
    chromatic cyclorama gives both SAM2 and the border-colour fallback a clear
    foreground/background separation, including for white furniture.
    """
    palettes = (
        "medium neutral gray",
        "pale blue-gray",
        "muted cool gray",
        "soft desaturated green-gray",
    )
    variant = (sum(ord(char) for char in label) + int(attempt)) % len(palettes)
    colour = palettes[variant]
    return (
        f"flat evenly-lit matte {colour} background, empty background only, "
        "no floor, no ground, no horizon line, no wall, no platform, no cast shadow"
    )


def _object_category(label: str) -> str:
    """Map descriptive scene labels to the construction rule they need.

    Gemini can return labels such as ``coffee table`` or ``bedside table``.
    They are still tables, but an exact dictionary lookup used to skip the
    table geometry and negative constraints.  This is intentionally generic
    so every table/chair/sofa/lamp variation receives the same structural
    protection.
    """
    normalized = " ".join(str(label).lower().replace("_", " ").split())
    if "table" in normalized or "desk" in normalized:
        return "table"
    if "chair" in normalized or "stool" in normalized:
        return "chair"
    if "sofa" in normalized or "couch" in normalized:
        return "sofa"
    if "lamp" in normalized or "light" in normalized:
        return "lamp"
    return normalized


def _object_prompt(
    label: str,
    scene_prompt: str,
    all_labels: list[str],
    object_description: str = "",
    generation_attempt: int = 0,
) -> tuple[str, str]:
    category = _object_category(label)
    excluded = ", ".join(sorted(set(all_labels) - {label})) or "other furniture"
    description = " ".join(str(object_description).split()).strip(" ,.;:")
    generic_descriptions = {
        "",
        label,
        f"one {label} requested by the user",
    }
    descriptor = (
        description
        if description.lower() not in generic_descriptions
        else _object_descriptor(label, scene_prompt)
    )
    # SD3.5 runs without T5-XXL on the T4 worker, leaving the two CLIP text
    # encoders with a short context window.  A verbose Gemini description used
    # to push the geometry/background rules beyond that window.  Keep the
    # object's identity, colour and material, but cap the noun phrase.
    descriptor_words = descriptor.split()
    if len(descriptor_words) > 20:
        descriptor = " ".join(descriptor_words[:20])
    detail_markers = (
        "carv", "ornate", "decor", "inlay", "engrave", "pattern", "motif",
        "floral", "beveled", "bevelled", "panel",
        "hoa văn", "chạm khắc", "họa tiết",
    )
    detail_requested = any(
        marker in f"{scene_prompt} {descriptor}".lower()
        for marker in detail_markers
    )
    if detail_requested and category == "table":
        detail_instruction = (
            "Show the requested carving only as a clear restrained border on the apron or tabletop edge. "
        )
    elif detail_requested and label == "chair":
        detail_instruction = (
            "Show the requested carving as one clear restrained panel on the backrest. "
        )
    elif detail_requested:
        detail_instruction = (
            "Keep the requested decoration clear, restrained, and attached to the object. "
        )
    else:
        detail_instruction = ""

    source_lower = f"{scene_prompt} {descriptor}".lower()
    explicit_shape = any(
        term in source_lower
        for term in ("round", "circular", "oval", "square", "rectangular", "curved")
    )
    explicit_pedestal = any(
        term in source_lower for term in ("pedestal", "trestle", "central base")
    )
    table_shape_instruction = (
        "Use a conventional rectangular tabletop "
        if not explicit_shape
        else "Preserve the explicitly requested tabletop shape "
    )
    table_base_instruction = (
        "with four straight vertical legs at the four corners"
        if not explicit_pedestal
        else "with the explicitly requested pedestal or base"
    )

    geometry_instructions = {
        "table": (
            f"Full table; {table_shape_instruction.lower()}"
            f"{table_base_instruction}, attached apron, all feet visible. "
        ),
        "chair": (
            "Full chair; one seat, one backrest, connected supports and four visible legs. "
        ),
        "sofa": (
            "Full upholstered sofa; two seat cushions, two upright back cushions, both arms and feet visible. "
        ),
        "lamp": (
            "Full freestanding floor lamp; one shade, a thin vertical stem and one base, all visible. "
        ),
        "floor lamp": (
            "Exactly one freestanding floor lamp, eye-level front three-quarter view, "
            "with the complete shade, vertical stem, base, and every support visible. "
        ),
    }
    geometry_instruction = geometry_instructions.get(
        category,
        "Complete object, eye-level front three-quarter view, all structural parts visible. ",
    )
    studio_background = _isolation_background(label, generation_attempt)
    positive = (
        f"Exactly one standalone {descriptor}. "
        f"{geometry_instruction}"
        f"{detail_instruction}"
        "Eye-level front three-quarter product photo, centered, fully visible, realistic proportions. "
        f"{studio_background}."
    )
    category_negative = {
        "sofa": "bed, bed frame, bench, empty frame, missing cushions, ",
        "lamp": "cabinet, column, rectangular block, missing shade, missing stem, missing base, ",
        "table": "rug attached to table, floor attached to legs, malformed tabletop, ",
        "chair": "stool, bench, missing backrest, ",
    }.get(category, "")
    negative = (
        f"{category_negative}extra object, duplicate, {excluded}, cropped, top-down, malformed, fused parts, "
        "floor, rug, carpet, platform, shadow, reflection, connected background, panel, "
        "icon, logo, blueprint, wireframe, text, watermark"
    )
    if detail_requested:
        negative += ", missing requested carving, blank decorative area"
    if category == "table" and not explicit_shape:
        negative += ", round tabletop, oval tabletop"
    if category == "table" and not explicit_pedestal:
        negative += ", pedestal table, central leg, three-legged table, cabriole legs"
    return positive, negative


def _stable_object_seed(
    scene_prompt: str,
    object_id: str,
    label: str,
    attempt: int,
) -> int:
    """Derive a stable seed from object identity instead of its list position.

    The previous ``42 + index * 101`` mapping made the third object use the
    same seed in every scene. A floor lamp in slot three therefore reproduced
    the same bad rectangular sample on every run.
    """
    identity = f"{scene_prompt}|{object_id}|{label}".encode("utf-8")
    base = int.from_bytes(hashlib.sha256(identity).digest()[:4], "big")
    return int((base + attempt * 1_000_003) % (2**31 - 1))


def build_object_jobs(
    scene_prompt: str,
    objects: list[dict],
    object_image_dir: str = OBJECT_IMAGE_DIR,
) -> list[dict]:
    """Build the deterministic per-object prompts shared with the SD3.5 worker."""
    if not objects:
        raise ValueError("The object-wise generator received no objects")

    Path(object_image_dir).mkdir(parents=True, exist_ok=True)
    all_labels = [str(item.get("label", "furniture")).lower() for item in objects]
    jobs = []
    for index, item in enumerate(objects, start=1):
        label = str(item.get("label", "furniture")).lower().strip() or "furniture"
        object_id = str(item.get("id", f"{label}_{index}"))
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", object_id):
            raise ValueError(f"Invalid object id: {object_id}")
        attempt = _generation_attempt(item)
        positive, negative = _object_prompt(
            label,
            scene_prompt,
            all_labels,
            object_description=str(item.get("description", "")),
            generation_attempt=attempt,
        )
        jobs.append({
            "name": object_id,
            "label": label,
            "prompt": positive,
            "negative_prompt": negative,
            "image_path": os.path.join(object_image_dir, f"{object_id}.png"),
            "seed": _stable_object_seed(scene_prompt, object_id, label, attempt),
        })
    return jobs


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


def _offload_worker(url: str, label: str) -> None:
    """Release another resident model before claiming the shared GPU."""
    try:
        response = requests.post(
            f"{url}/offload",
            timeout=WORKER_OFFLOAD_TIMEOUT,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        detail = _response_error_detail(exc.response)
        raise RuntimeError(
            f"{label} worker could not release GPU memory at {url}: "
            f"{detail or exc}"
        ) from exc


def _check_worker_ready() -> None:
    try:
        response = requests.get(f"{SD35_WORKER_URL}/health", timeout=5)
        response.raise_for_status()
        health = response.json()
    except requests.RequestException as exc:
        raise RuntimeError(
            f"SD3.5 worker is unavailable at {SD35_WORKER_URL}. "
            "Start worker_sd35.py first."
        ) from exc

    if not health.get("ready"):
        detail = health.get("error") or "model is still loading"
        raise RuntimeError(f"SD3.5 worker is not ready: {detail}")


def _sanitize_generated_jobs(jobs: list[dict]) -> list[dict]:
    """Keep exactly one detected instance in every generated object image."""
    try:
        response = requests.post(
            f"{SAM2_DINO_WORKER_URL}/sanitize",
            json={"objects": jobs},
            timeout=OBJECT_SANITIZE_TIMEOUT,
        )
        response.raise_for_status()
        sanitized = response.json().get("jobs", [])
    except requests.RequestException as exc:
        # ``requests.Response.__bool__`` is False for HTTP 4xx/5xx.  Checking
        # the response by truthiness therefore discarded the JSON error body
        # exactly when it was needed to identify the failed object.
        detail = (
            _response_error_detail(exc.response)
            if exc.response is not None
            else ""
        )
        raise RuntimeError(
            "Generated-object count validation failed. "
            f"SAM2/DINO worker response: {detail or exc}"
        ) from exc

    if len(sanitized) != len(jobs):
        raise RuntimeError(
            "Generated-object count validation returned "
            f"{len(sanitized)} jobs for {len(jobs)} objects"
        )
    return sanitized


def _generate_jobs_from_worker(
    scene_prompt: str,
    objects: list[dict],
    lora_scale: float,
    object_image_dir: str,
) -> list[dict]:
    """Ask the resident SD3.5 worker to generate one or more object images."""
    try:
        response = requests.post(
            f"{SD35_WORKER_URL}/generate",
            json={
                "scene_prompt": scene_prompt,
                "objects": objects,
                "lora_scale": lora_scale,
                "output_dir": object_image_dir,
            },
            timeout=SD35_WORKER_TIMEOUT,
        )
        response.raise_for_status()
        jobs = response.json().get("jobs", [])
    except requests.RequestException as exc:
        detail = _response_error_detail(exc.response)
        raise RuntimeError(
            f"SD3.5 worker generation failed: {detail or exc}"
        ) from exc

    if len(jobs) != len(objects):
        raise RuntimeError(
            f"SD3.5 worker returned {len(jobs)} jobs for {len(objects)} objects"
        )
    return jobs


def _objects_needing_clean_background_retry(
    objects: list[dict],
    error: Exception,
) -> list[dict]:
    """Return only the object(s) named by a sanitization failure.

    The SAM2 worker includes both the stable object id and label in errors.
    This lets us regenerate the one bad intermediate asset instead of rerunning
    the entire scene and works for any furniture category.
    """
    error_text = str(error).lower()
    matches = []
    for item in objects:
        object_id = str(item.get("id", "")).lower()
        label = str(item.get("label", "")).lower()
        if (object_id and object_id in error_text) or (label and label in error_text):
            attempt = _generation_attempt(item) + 1
            matches.append(
                dict(
                    item,
                    generation_attempt=attempt,
                    retry_clean_background=True,
                )
            )
    return matches


def generate_object_images(
    scene_prompt: str,
    objects: list[dict],
    lora_scale: float = 0.2,
    object_image_dir: str = OBJECT_IMAGE_DIR,
    manifest_path: str = OBJECT_MANIFEST,
) -> list[dict]:
    """Generate isolated object images through the resident SD3.5 worker."""
    if not objects:
        raise ValueError("The object-wise generator received no objects")

    _check_worker_ready()
    # SD3.5 is the active GPU model in this phase.  TRELLIS and SAM2/DINO are
    # resident workers too, so explicitly move both to CPU before loading the
    # diffusion pipeline back onto CUDA.
    _offload_worker(TRELLIS_WORKER_URL, "TRELLIS")
    _offload_worker(SAM2_DINO_WORKER_URL, "SAM2/DINO")
    jobs = _generate_jobs_from_worker(
        scene_prompt,
        objects,
        lora_scale,
        object_image_dir,
    )

    # Prompt wording alone cannot guarantee cardinality in a diffusion model.
    # Validate each image with GroundingDINO and retain one SAM2 instance before
    # the contact sheet is exposed to the frontend or sent to TRELLIS.
    pending_objects = [dict(item, generation_attempt=0) for item in objects]
    for attempt in range(3):
        try:
            jobs = _sanitize_generated_jobs(jobs)
        except RuntimeError as generation_error:
            retry_objects = _objects_needing_clean_background_retry(
                pending_objects,
                generation_error,
            )
            if not retry_objects or attempt >= 2:
                attempts_made = attempt + 1
                raise RuntimeError(
                    f"Object isolation failed after {attempts_made} SD3.5 "
                    f"sample{'s' if attempts_made != 1 else ''}. "
                    "The malformed asset was blocked before TRELLIS. "
                    f"Details: {generation_error}"
                ) from generation_error
        else:
            warned_names = {
                str(job.get("name", ""))
                for job in jobs
                if job.get("retry_recommended")
            }
            if not warned_names or attempt >= 2:
                break
            retry_objects = []
            for item in pending_objects:
                if str(item.get("id", "")) not in warned_names:
                    continue
                retry_objects.append(
                    dict(
                        item,
                        generation_attempt=_generation_attempt(item) + 1,
                        retry_clean_background=True,
                    )
                )
            if not retry_objects:
                break

        retry_jobs = _generate_jobs_from_worker(
            scene_prompt,
            retry_objects,
            lora_scale,
            object_image_dir,
        )
        retry_by_name = {str(job.get("name", "")): job for job in retry_jobs}
        jobs = [retry_by_name.get(str(job.get("name", "")), job) for job in jobs]
        retry_names = {str(item.get("id", "")) for item in retry_objects}
        pending_objects = [
            next(
                (
                    retry
                    for retry in retry_objects
                    if str(retry.get("id", "")) == str(item.get("id", ""))
                ),
                item,
            )
            if str(item.get("id", "")) in retry_names
            else item
            for item in pending_objects
        ]

    for job in jobs:
        image_path = job.get("image_path")
        if not image_path or not os.path.isfile(image_path):
            raise FileNotFoundError(image_path or "Missing SD3.5 image path")

    Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(jobs, file, ensure_ascii=False, indent=2)
    return jobs


def create_object_contact_sheet(objects: list[dict], output_path: str) -> None:
    """Create a visual checkpoint for the graph's SD3.5 node."""
    from PIL import Image, ImageOps

    cards = []
    for item in objects:
        image = Image.open(item["image_path"]).convert("RGB")
        cards.append(ImageOps.contain(image, (480, 480), Image.Resampling.LANCZOS))
    width = 480 * min(2, len(cards))
    sheet = Image.new("RGB", (width, 480 * ((len(cards) + 1) // 2)), "white")
    for index, image in enumerate(cards):
        x = (index % 2) * 480 + (480 - image.width) // 2
        y = (index // 2) * 480 + (480 - image.height) // 2
        sheet.paste(image, (x, y))
    sheet.save(output_path)
