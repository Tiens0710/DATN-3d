import os
import subprocess

PY_PATH = "/opt/venv310/bin/python"

def generate_2d_image(prompt: str, lora_scale: float, output_path: str) -> bool:
    """
    Chạy mô hình SDXL base + LoRA sinh ảnh 2D từ prompt.
    """
    safe_prompt = prompt.replace('\\', '\\\\').replace('"', '\\"')
    script = f"""
import sys
sys.path.insert(0, "/opt/venv310/lib/python3.10/site-packages")

import os, torch
sys.modules['triton'] = None  # Chặn lỗi bitsandbytes

from diffusers import AutoPipelineForText2Image
from peft import PeftModel
from PIL import Image

print("Loading SDXL...")
pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
).to("cuda")

LORA_OUT = "/kaggle/input/datasets/tiens0710/pretrain-2d"
if os.path.exists(LORA_OUT):
    print("Loading LoRA...")
    pipe.unet = PeftModel.from_pretrained(pipe.unet, LORA_OUT)
    for name, module in pipe.unet.named_modules():
        if hasattr(module, "scaling") and "default" in module.scaling:
            module.scaling["default"] = {lora_scale}
    pipe.unet = pipe.unet.merge_and_unload()
    print("✅ Merge LoRA xong.")

pipe.enable_attention_slicing()
pipe.enable_vae_slicing()

generator = torch.Generator("cuda").manual_seed(42)

print("Inference SDXL...")
img = pipe(
    prompt="{safe_prompt}",
    negative_prompt="blurry, distorted, deformed, low quality, watermark, text, cropped, extra chairs, duplicate",
    num_inference_steps=30,
    guidance_scale=7.5,
    generator=generator,
    height=1024,
    width=1024
).images[0]

img.save("{output_path}")
print("✅ Sinh ảnh xong!")
"""
    r = subprocess.run([PY_PATH, "-c", script], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"Lỗi sinh ảnh SDXL: {r.stderr}")
    return os.path.exists(output_path)
