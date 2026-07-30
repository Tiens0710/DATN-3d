import json
import os
import subprocess


PY_PATH = "/opt/venv310/bin/python"
TRELLIS_ROOT = "/kaggle/working/TRELLIS"


def generate_3d_models(crops: list, multi_glb_dir: str) -> list:
    """Generate one GLB per transparent SAM2 crop with TRELLIS."""
    if not crops:
        raise ValueError("No SAM2 crops were provided to TRELLIS")

    if not os.path.isdir(os.path.join(TRELLIS_ROOT, "trellis")):
        raise RuntimeError(
            "TRELLIS source is missing. Run the backend notebook setup cells."
        )

    os.makedirs(multi_glb_dir, exist_ok=True)
    meta_json_path = "/kaggle/working/objects_meta_api.json"
    objects_dict = {crop["name"]: crop for crop in crops}
    with open(meta_json_path, "w", encoding="utf-8") as file:
        json.dump(objects_dict, file, ensure_ascii=False, indent=2)

    script = f"""
import json
import os
import sys

sys.path.insert(0, "/opt/venv310/lib/python3.10/site-packages")
sys.path.insert(0, {TRELLIS_ROOT!r})
sys.modules["triton"] = None

os.environ["SPCONV_ALGO"] = "native"
os.environ["ATTN_BACKEND"] = "xformers"
os.environ["SPARSE_ATTN"] = "xformers"
os.environ["MPLBACKEND"] = "agg"

import torch
import nvdiffrast.torch
import spconv
import utils3d
import xformers
from diff_gaussian_rasterization import _C
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import postprocessing_utils

with open({meta_json_path!r}, encoding="utf-8") as file:
    objects = json.load(file)

print("Loading TRELLIS-image-large...")
pipeline = TrellisImageTo3DPipeline.from_pretrained(
    "JeffreyXiang/TRELLIS-image-large"
)
pipeline.to("cuda")

for name, info in objects.items():
    crop_path = info["crop_path"]
    if not os.path.isfile(crop_path):
        raise FileNotFoundError(crop_path)

    print(f"Generating TRELLIS model for {{name}}...")
    image = Image.open(crop_path).convert("RGB")
    image = pipeline.preprocess_image(image)
    outputs = pipeline.run(
        image,
        seed=42,
        formats=["gaussian", "mesh"],
        preprocess_image=False,
        sparse_structure_sampler_params={{
            "steps": 12,
            "cfg_strength": 7.5,
        }},
        slat_sampler_params={{
            "steps": 12,
            "cfg_strength": 3.0,
        }},
    )

    glb = postprocessing_utils.to_glb(
        outputs["gaussian"][0],
        outputs["mesh"][0],
        simplify=0.95,
        texture_size=1024,
        verbose=False,
    )
    output_path = os.path.join({multi_glb_dir!r}, f"{{name}}.glb")
    glb.export(output_path)
    print("Saved:", output_path)
    torch.cuda.empty_cache()
"""

    environment = os.environ.copy()
    environment.setdefault("SPCONV_ALGO", "native")
    environment.setdefault("ATTN_BACKEND", "xformers")
    environment.setdefault("SPARSE_ATTN", "xformers")
    environment.setdefault("MPLBACKEND", "agg")

    result = subprocess.run(
        [PY_PATH, "-c", script],
        capture_output=True,
        text=True,
        timeout=1200,
        env=environment,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "TRELLIS failed. Run the notebook preflight cell before starting "
            f"the server.\nSTDERR:\n{result.stderr}\nSTDOUT:\n{result.stdout}"
        )

    models = []
    for crop in crops:
        name = crop["name"]
        model_path = os.path.join(multi_glb_dir, f"{name}.glb")
        if not os.path.isfile(model_path):
            raise FileNotFoundError(model_path)
        models.append(
            {
                "name": name,
                "label": crop["label"],
                "model_url": f"/multi_object_glb/{name}.glb",
                "model_path": model_path,
                "final_box": crop["final_box"],
            }
        )
    return models
