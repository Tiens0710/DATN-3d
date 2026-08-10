import json
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
SAM2_DINO_WORKER_URL = os.environ.get(
    "SAM2_DINO_WORKER_URL",
    "http://127.0.0.1:8003",
).rstrip("/")
OBJECT_SANITIZE_TIMEOUT = float(
    os.environ.get("OBJECT_SANITIZE_TIMEOUT", "600")
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
    "lamp": ("lamp", "đèn", "den"),
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


def _object_prompt(label: str, scene_prompt: str, all_labels: list[str]) -> tuple[str, str]:
    excluded = ", ".join(sorted(set(all_labels) - {label})) or "other furniture"
    descriptor = _object_descriptor(label, scene_prompt)
    detail_markers = (
        "carv", "ornate", "decor", "inlay", "engrave", "pattern", "motif",
        "floral", "wood grain", "beveled", "bevelled", "panel",
        "hoa văn", "chạm khắc", "họa tiết",
    )
    detail_requested = any(
        marker in f"{scene_prompt} {descriptor}".lower()
        for marker in detail_markers
    )
    if detail_requested and label == "table":
        detail_instruction = (
            "The requested decorative craftsmanship is mandatory and clearly visible: "
            "a distinct carved border or inlay along the tabletop edge, not a plain blank top. "
        )
    elif detail_requested and label == "chair":
        detail_instruction = (
            "The requested decorative craftsmanship is mandatory and clearly visible: "
            "a distinct carved motif or patterned panel on the chair backrest, not a plain blank back. "
        )
    elif detail_requested:
        detail_instruction = (
            "The requested decorative craftsmanship is mandatory and clearly visible on the main object, "
            "not a plain blank surface. "
        )
    else:
        detail_instruction = ""
    positive = (
        f"Exactly one complete standalone {descriptor}, centered and fully visible. "
        f"One {label} only, a single subject and a single instance, not a pair or set. "
        f"{detail_instruction}"
        "Isolated product photo, empty surroundings, no scene, no extra furniture, "
        "no second object, no duplicate parts, "
        "generous margins, correct structure, realistic proportions, clean white "
        "studio background, soft even lighting, sharp focus, realistic materials."
    )
    negative = (
        f"two {label}s, pair of {label}s, {label} set, multiple objects, second {label}, "
        f"duplicate object, repeated subject, extra furniture, {excluded}, "
        "plain blank surface, featureless smooth wood, missing decoration, "
        "room scene, cropped, close-up, "
        "fused parts, intersecting parts, missing parts, duplicate legs, floating parts, "
        "deformed, blurry, low quality, room, people, text, watermark"
    )
    return positive, negative


def build_object_jobs(scene_prompt: str, objects: list[dict]) -> list[dict]:
    """Build the deterministic per-object prompts shared with the SD3.5 worker."""
    if not objects:
        raise ValueError("The object-wise generator received no objects")

    Path(OBJECT_IMAGE_DIR).mkdir(parents=True, exist_ok=True)
    all_labels = [str(item.get("label", "furniture")).lower() for item in objects]
    jobs = []
    for index, item in enumerate(objects, start=1):
        label = str(item.get("label", "furniture")).lower().strip() or "furniture"
        object_id = str(item.get("id", f"{label}_{index}"))
        positive, negative = _object_prompt(label, scene_prompt, all_labels)
        jobs.append({
            "name": object_id,
            "label": label,
            "prompt": positive,
            "negative_prompt": negative,
            "image_path": os.path.join(OBJECT_IMAGE_DIR, f"{object_id}.png"),
            "seed": 42 + index * 101,
        })
    return jobs


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
        detail = getattr(exc.response, "text", "") if exc.response else ""
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


def generate_object_images(
    scene_prompt: str,
    objects: list[dict],
    lora_scale: float = 0.2,
) -> list[dict]:
    """Generate isolated object images through the resident SD3.5 worker."""
    if not objects:
        raise ValueError("The object-wise generator received no objects")

    _check_worker_ready()
    try:
        response = requests.post(
            f"{SD35_WORKER_URL}/generate",
            json={
                "scene_prompt": scene_prompt,
                "objects": objects,
                "lora_scale": lora_scale,
            },
            timeout=SD35_WORKER_TIMEOUT,
        )
        response.raise_for_status()
        jobs = response.json().get("jobs", [])
    except requests.RequestException as exc:
        detail = getattr(exc.response, "text", "") if exc.response else ""
        raise RuntimeError(
            f"SD3.5 worker generation failed: {detail or exc}"
        ) from exc

    if len(jobs) != len(objects):
        raise RuntimeError(
            f"SD3.5 worker returned {len(jobs)} jobs for {len(objects)} objects"
        )

    # Prompt wording alone cannot guarantee cardinality in a diffusion model.
    # Validate each image with GroundingDINO and retain one SAM2 instance before
    # the contact sheet is exposed to the frontend or sent to TRELLIS.
    jobs = _sanitize_generated_jobs(jobs)

    for job in jobs:
        image_path = job.get("image_path")
        if not image_path or not os.path.isfile(image_path):
            raise FileNotFoundError(image_path or "Missing SD3.5 image path")

    with open(OBJECT_MANIFEST, "w", encoding="utf-8") as file:
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
