import os
import json
import subprocess

PY_PATH = "/opt/venv310/bin/python"

def generate_3d_models(crops: list, multi_glb_dir: str) -> list:
    """
    Sử dụng TRELLIS để tạo các mô hình 3D (.glb) riêng lẻ từ ảnh tách nền.
    """
    meta_json_path = "/kaggle/working/objects_meta_api.json"
    objects_dict = {crop["name"]: crop for crop in crops}
    with open(meta_json_path, 'w') as f:
        json.dump(objects_dict, f)

    script = f"""
import sys, subprocess
sys.path.insert(0, "/opt/venv310/lib/python3.10/site-packages")

# ── MOCK: Chặn triton (bitsandbytes) ──
sys.modules['triton'] = None

# ── Đồng bộ bộ ba Torch 2.1.0 + Torchvision 0.16.0 + xFormers 0.0.22.post7 ──
# ── Đồng bộ bộ ba Torch 2.1.0 + Torchvision 0.16.0 + xFormers 0.0.22.post7 ──
needs_sync = False
try:
    import torch
    import torchvision
    import xformers
    import numpy as np
    if not torch.__version__.startswith("2.1.0"):
        needs_sync = True
    if xformers.__version__ != "0.0.22.post7":
        needs_sync = True
    if np.__version__ != "1.26.4":
        needs_sync = True
except Exception:
    needs_sync = True

if needs_sync:
    print("Dong bo lai phien ban thu vien (Torch 2.1.0 + xFormers)...")
    # Clean up numpy directories first to avoid "Frankenstein" mix of numpy 1.x and 2.x
    import shutil, glob
    for p in glob.glob("/opt/venv310/lib/python3.10/site-packages/numpy*"):
        try:
            if os.path.isdir(p): shutil.rmtree(p)
            else: os.remove(p)
        except Exception: pass
    r1 = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--force-reinstall", "--no-cache-dir", "torch==2.1.0", "torchvision==0.16.0", "xformers==0.0.22.post7", "numpy==1.26.4", "--index-url", "https://download.pytorch.org/whl/cu121"],
        capture_output=True, text=True, timeout=300
    )
    if r1.returncode != 0:
        print("Loi dong bo torch/xformers:", r1.stderr)
else:
    print("Moi truong Torch 2.1.0 + xFormers da hop le, bo qua dong bo.")

# Thiết lập PATH chứa nvcc để compile CUDA extension
import os
compile_env = os.environ.copy()
compile_env["CUDA_HOME"] = "/usr/local/cuda"
compile_env["PATH"] = "/usr/local/cuda/bin:" + compile_env.get("PATH", "")

needs_compile_nvdiffrast = needs_sync
if not needs_compile_nvdiffrast:
    try:
        import nvdiffrast.torch as dr
        print("Nvdiffrast da duoc compile, bo qua.")
    except Exception:
        needs_compile_nvdiffrast = True

if needs_compile_nvdiffrast:
    print("Dang compile lai nvdiffrast...")
    r2 = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-deps", "--no-cache-dir", "--no-build-isolation", "git+https://github.com/NVlabs/nvdiffrast.git"],
        capture_output=True, text=True, timeout=180, env=compile_env
    )
    if r2.returncode != 0:
        print("Loi compile nvdiffrast:", r2.stderr)
    else:
        print("Nvdiffrast OK!")

needs_compile_utils3d = needs_sync
if not needs_compile_utils3d:
    try:
        import utils3d
        print("Utils3d da duoc compile, bo qua.")
    except Exception:
        needs_compile_utils3d = True

if needs_compile_utils3d:
    print("Dang compile lai utils3d...")
    r3 = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-deps", "--no-cache-dir", "--no-build-isolation", "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8"],
        capture_output=True, text=True, timeout=180, env=compile_env
    )
    if r3.returncode != 0:
        print("Loi compile utils3d:", r3.stderr)
    else:
        print("Utils3d OK!")

needs_compile_diff_gaussian = needs_sync
if not needs_compile_diff_gaussian:
    try:
        from diff_gaussian_rasterization import _C
        print("diff_gaussian_rasterization da duoc compile, bo qua.")
    except Exception:
        needs_compile_diff_gaussian = True

if needs_compile_diff_gaussian:
    print("Dang compile lai diff_gaussian_rasterization...")
    import shutil
    if os.path.exists("/tmp/mip-splatting"):
        try: shutil.rmtree("/tmp/mip-splatting")
        except Exception: pass
    r_clone = subprocess.run(
        ["git", "clone", "--recursive", "https://github.com/autonomousvision/mip-splatting.git", "/tmp/mip-splatting"],
        capture_output=True, text=True
    )
    r4 = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-deps", "--no-cache-dir", "--no-build-isolation", "/tmp/mip-splatting/submodules/diff-gaussian-rasterization"],
        capture_output=True, text=True, timeout=300, env=compile_env
    )
    if r4.returncode != 0:
        print("Loi compile diff_gaussian_rasterization:", r4.stderr)
    else:
        print("diff_gaussian_rasterization OK!")

print("Dong bo hoan tat!")

# ── CẤU HÌNH BACKEND (giống code cũ chạy thành công) ─────────────
import os, json, torch
os.environ["SPCONV_ALGO"]  = "native"
os.environ["ATTN_BACKEND"] = "xformers"
os.environ["SPARSE_ATTN"]  = "xformers"
os.environ["MPLBACKEND"]   = "agg"

# ── Clone TRELLIS nếu chưa có ────────────────────────────────────────
if not os.path.exists("/kaggle/working/TRELLIS/trellis"):
    print("Auto-cloning TRELLIS repository...")
    import shutil, subprocess
    if os.path.exists("/kaggle/working/TRELLIS"):
        try:
            shutil.rmtree("/kaggle/working/TRELLIS")
        except Exception as e:
            print("Warning: could not clean TRELLIS folder:", e)
    res_clone = subprocess.run(
        "GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/spaces/trellis-community/TRELLIS /kaggle/working/TRELLIS",
        shell=True, capture_output=True, text=True
    )
    if res_clone.returncode != 0:
        print("Git clone failed!")
        print("STDOUT:", res_clone.stdout)
        print("STDERR:", res_clone.stderr)
        raise RuntimeError(f"Git clone failed: {{res_clone.stderr}}")

sys.path.insert(0, "/kaggle/working/TRELLIS")
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import postprocessing_utils
from PIL import Image

with open("{meta_json_path}") as f:
    objects = json.load(f)

# Load pipeline và chuyển sang GPU (KHÔNG dùng dtype - Pipeline.to() không hỗ trợ)
pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
pipeline.to("cuda")

for name, info in objects.items():
    print(f"Dựng mô hình 3D cho {{name}}...")
    img = Image.open(info["crop_path"]).convert("RGB")
    image = pipeline.preprocess_image(img)
    
    outputs = pipeline.run(
        image,
        seed=42,
        formats=["gaussian", "mesh"],
        preprocess_image=False,
        sparse_structure_sampler_params={{"steps": 12, "cfg_strength": 7.5}},
        slat_sampler_params={{"steps": 12, "cfg_strength": 3.0}},
    )
    
    glb = postprocessing_utils.to_glb(
        outputs["gaussian"][0], outputs["mesh"][0],
        simplify=0.95, texture_size=1024, verbose=False
    )
    out_path = f"{multi_glb_dir}/{{name}}.glb"
    glb.export(out_path)
    print(f"Saved: {{out_path}}")
    torch.cuda.empty_cache()
"""
    r = subprocess.run([PY_PATH, "-c", script], capture_output=True, text=True, timeout=1200)
    if r.returncode != 0:
        raise RuntimeError(f"Lỗi chạy TRELLIS: {r.stderr}\nLogs:\n{r.stdout}")
        
    models = []
    for crop in crops:
        name = crop["name"]
        models.append({
            "name": name,
            "label": crop["label"],
            "model_url": f"/multi_object_glb/{name}.glb",
            "model_path": f"{multi_glb_dir}/{name}.glb",
            "final_box": crop["final_box"]
        })
    return models
