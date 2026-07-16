import os
import subprocess


PY_PATH = "/opt/venv310/bin/python"


def generate_2d_image(prompt: str, lora_scale: float, output_path: str) -> bool:
    """Generate a 1024px image with the default SD3.5 Medium pipeline."""
    del lora_scale  # Kept in the API for backward compatibility.

    safe_prompt = prompt.replace("\\", "\\\\").replace('"', '\\"')
    hf_token = os.environ.get("HF_TOKEN", "").strip()
    if not hf_token:
        raise RuntimeError("HF_TOKEN is not configured")

    script = f"""
import sys
sys.path.insert(0, "/opt/venv310/lib/python3.10/site-packages")

import torch
sys.modules["triton"] = None

from diffusers import StableDiffusion3Pipeline

MODEL_ID = "stabilityai/stable-diffusion-3.5-medium"

print("Loading default SD3.5 Medium pipeline...")
pipe = StableDiffusion3Pipeline.from_pretrained(
    MODEL_ID,
    text_encoder_3=None,
    tokenizer_3=None,
    torch_dtype=torch.float16,
    token="{hf_token}",
)
pipe = pipe.to("cuda")

generator = torch.Generator("cuda").manual_seed(42)

image = pipe(
    prompt="{safe_prompt}",
    negative_prompt="blurry, distorted, deformed, low quality, text, watermark",
    num_inference_steps=30,
    guidance_scale=4.5,
    generator=generator,
    height=1024,
    width=1024,
).images[0]

image.save("{output_path}")
print("SD3.5 image saved:", "{output_path}")
"""

    result = subprocess.run(
        [PY_PATH, "-c", script],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"SD3.5 generation failed:\n{result.stderr}\n{result.stdout}"
        )
    return os.path.exists(output_path)
