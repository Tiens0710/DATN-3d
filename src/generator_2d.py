import json
import os
import re
import shutil
import subprocess
from pathlib import Path


PY_PATH = "/opt/venv310/bin/python"
OBJECT_IMAGE_DIR = "/kaggle/working/object_images"
OBJECT_MANIFEST = "/kaggle/working/object_images_manifest.json"
SD35_CACHE_DIR = "/kaggle/working/sd35_medium_cache_v1"


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
    r"\balongside\b|\bnear\b|\bto the left of\b|\bto the right of\b|"
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
    positive = (
        f"Exactly one complete standalone {descriptor}, centered and fully visible. "
        "This is an isolated single-object product photo. Render only this object, "
        "with no spatial scene, no companion furniture, no duplicate parts, generous "
        "margins, correct structure, realistic proportions, clean white studio "
        "background, soft even lighting, sharp focus, realistic materials, suitable "
        "for single-image 3D reconstruction."
    )
    negative = (
        f"multiple objects, second {label}, duplicate object, extra furniture, {excluded}, "
        "room scene, spatial relationship, cropped, close-up, "
        "fused parts, intersecting parts, missing parts, duplicate legs, floating parts, "
        "deformed, blurry, low quality, room, people, text, watermark"
    )
    return positive, negative


def generate_object_images(scene_prompt: str, objects: list[dict], lora_scale: float = 0.0) -> list[dict]:
    """Generate one isolated SD3.5 image per requested object in one GPU process."""
    del lora_scale
    if not objects:
        raise ValueError("The object-wise generator received no objects")

    hf_token = os.environ.get("HF_TOKEN", "").strip()
    if not hf_token:
        raise RuntimeError("HF_TOKEN is not configured")

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

    jobs_path = "/kaggle/working/object_generation_jobs.json"
    with open(jobs_path, "w", encoding="utf-8") as file:
        json.dump(jobs, file, ensure_ascii=False, indent=2)

    script = f"""
import json
import os
import shutil
import sys
sys.path.insert(0, '/opt/venv310/lib/python3.10/site-packages')
sys.modules['triton'] = None

import torch
import diffusers
from packaging.version import Version
from diffusers import StableDiffusion3Pipeline

if Version(diffusers.__version__) < Version('0.32.0'):
    raise RuntimeError(
        'SD3.5 Medium requires Diffusers >= 0.32.0. Found ' + diffusers.__version__ +
        '. Run the API notebook dependency cell, then restart FastAPI.'
    )

print('Diffusers version:', diffusers.__version__)

with open({jobs_path!r}, encoding='utf-8') as file:
    jobs = json.load(file)

pipe = StableDiffusion3Pipeline.from_pretrained(
    'stabilityai/stable-diffusion-3.5-medium',
    text_encoder_3=None,
    tokenizer_3=None,
    torch_dtype=torch.float16,
    token={hf_token!r},
    cache_dir={SD35_CACHE_DIR!r},
).to('cuda')

for job in jobs:
    print('Generating isolated object:', job['name'], job['label'])
    image = pipe(
        prompt=job['prompt'],
        negative_prompt=job['negative_prompt'],
        num_inference_steps=35,
        guidance_scale=4.5,
        generator=torch.Generator('cuda').manual_seed(job['seed']),
        width=768,
        height=768,
    ).images[0]
    image.save(job['image_path'])
    torch.cuda.empty_cache()
"""
    result = subprocess.run([PY_PATH, "-c", script], capture_output=True, text=True, timeout=900)
    cache_mismatch = (
        "expected shape tensor" in result.stderr
        or "size mismatch" in result.stderr
    )
    if result.returncode != 0 and cache_mismatch:
        # A partly downloaded snapshot can pair a new config with old weights.
        # Repair only this model's dedicated cache, then retry once.
        shutil.rmtree(SD35_CACHE_DIR, ignore_errors=True)
        retry_script = script.replace(
            f"cache_dir={SD35_CACHE_DIR!r},",
            f"cache_dir={SD35_CACHE_DIR!r}, force_download=True,",
        )
        result = subprocess.run(
            [PY_PATH, "-c", retry_script],
            capture_output=True,
            text=True,
            timeout=1500,
        )
    if result.returncode != 0:
        raise RuntimeError("SD3.5 object-wise generation failed:\n" + result.stderr + "\n" + result.stdout)

    for job in jobs:
        if not os.path.isfile(job["image_path"]):
            raise FileNotFoundError(job["image_path"])
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
