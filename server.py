import os
import shutil
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict, List, Any

# Import modules from src package
from src.parser import parse_scene_graph
from src.layout import compute_layout
from src.generator_2d import generate_2d_image
from src.segmenter import run_grounded_sam2
from src.generator_3d import generate_3d_models
from src.combiner import combine_scene_meshes

app = FastAPI(title="DATN 3D Scene Reconstruction API", version="1.0.0")

# CORS middleware config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thư mục làm việc trên Kaggle
KAGGLE_WORKING = "/kaggle/working"
CROPS_DIR = os.path.join(KAGGLE_WORKING, "crops")
MULTI_GLB_DIR = os.path.join(KAGGLE_WORKING, "multi_object_glb")
OUT_DIR = os.path.join(KAGGLE_WORKING, "outputs/trellis")

# Tạo các thư mục nếu chưa có
os.makedirs(CROPS_DIR, exist_ok=True)
os.makedirs(MULTI_GLB_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# Mount các thư mục tĩnh để cung cấp file ảnh/model cho Frontend qua HTTP
app.mount("/crops", StaticFiles(directory=CROPS_DIR), name="crops")
app.mount("/multi_object_glb", StaticFiles(directory=MULTI_GLB_DIR), name="multi_object_glb")
app.mount("/outputs", StaticFiles(directory=OUT_DIR), name="outputs")

# =========================================================================
# API Models
# =========================================================================
class TextPrompt(BaseModel):
    text: str

class ImageGenRequest(BaseModel):
    prompt: str
    layout: Dict[str, Any]
    lora_scale: float = 0.6

class Sam2Request(BaseModel):
    image_url: str
    layout: Dict[str, Any]

class TrellisRequest(BaseModel):
    crops: List[Dict[str, Any]]

class CombineRequest(BaseModel):
    models: List[Dict[str, Any]]
    layout: Dict[str, Any]
    scale_factor: float = 0.01

# =========================================================================
# API Endpoints
# =========================================================================

@app.get("/api/health")
def health_check():
    return {"status": "ok", "version": "1.0.0"}

@app.post("/api/parse_scene_graph")
def api_parse_scene_graph(request: TextPrompt):
    try:
        return parse_scene_graph(request.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate_layout")
def api_generate_layout(scene_graph: Dict[str, Any]):
    try:
        return compute_layout(scene_graph)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate_image")
def api_generate_image(request: ImageGenRequest):
    try:
        output_image_path = os.path.join(KAGGLE_WORKING, "input.png")
        success = generate_2d_image(request.prompt, request.lora_scale, output_image_path)
        if success:
            shutil.copy(output_image_path, os.path.join(OUT_DIR, "input_2d.png"))
            return {"status": "success", "image_url": "/outputs/input_2d.png"}
        else:
            raise HTTPException(status_code=500, detail="Image generation failed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/run_sam2")
def api_run_sam2(request: Sam2Request):
    try:
        input_image_path = os.path.join(KAGGLE_WORKING, "input.png")
        crops_data = run_grounded_sam2(input_image_path, request.layout, CROPS_DIR)
        return {"status": "success", "crops": crops_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate_3d")
def api_generate_3d(request: TrellisRequest):
    try:
        models = generate_3d_models(request.crops, MULTI_GLB_DIR)
        return {"status": "success", "models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/combine_scene")
def api_combine_scene(request: CombineRequest):
    try:
        output_scene_path = os.path.join(OUT_DIR, "scene_combined.glb")
        success = combine_scene_meshes(request.models, output_scene_path, request.scale_factor)
        if success:
            return {"status": "success", "scene_url": "/outputs/scene_combined.glb"}
        else:
            raise HTTPException(status_code=500, detail="Scene combining failed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
