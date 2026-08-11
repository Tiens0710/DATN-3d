# DATN SpatialFlow: Text/Image-to-3D Pipeline

Do an tot nghiep xay dung pipeline tao tai san 3D tu mo ta van ban tieng Viet/tieng Anh hoac anh dau vao. He thong dung giao dien node graph, backend FastAPI tren Kaggle GPU va cac worker mo hinh thuong tru.

## Kien truc hien tai

1. Gemini toi uu prompt va trich xuat scene graph mo.
2. SD3.5 Medium + LoRA sinh rieng mot anh cho tung vat the.
3. GroundingDINO + SAM2 dinh vi, tach nen va kiem tra so luong vat the.
4. TRELLIS Image-to-3D sinh mot GLB cho tung crop.
5. Trimesh dat cac GLB theo toan bo quan he trong scene graph va xuat scene.

Nhanh anh dau vao bo qua prompt optimizer va SD3.5. Gemini Vision kiem tra anh co
phu hop cho single-image 3D va nhan dien cac vat the tien canh; GroundingDINO +
SAM2 sau do tach tung vat the rieng. Anh phang, tranh ve, khuon mat, anh cat qua
nhieu hoac detection fallback se bi tu choi thay vi tao GLB phang/trung lap.

## Co lap tung luot chay

Frontend tao mot `run_id` moi cho moi lan bam Run. Tat ca tep cua luot chay duoc luu tai:

```text
/kaggle/working/runs/<run_id>/
  input.png
  object_images/
  object_images_manifest.json
  crops/
  models/
  outputs/
```

Khong dung lai `input.png`, manifest, crop hay GLB cua luot truoc. Run cu duoc don theo `RUN_TTL_SECONDS` va `MAX_STORED_RUNS`.

## Cau hinh bat buoc tren Kaggle

Dung notebook [api-3d.ipynb](api-3d.ipynb) de tao Python 3.10 environment, cai CUDA extensions, tai checkpoint va khoi dong ba worker cung FastAPI.

Secrets:

```text
HF_TOKEN
GEMINI_API_KEY
NGROK_TOKEN
APP_API_KEY
```

Bien moi truong production nen dat:

```text
APP_API_KEY=<mot chuoi dai ngau nhien>
ALLOWED_ORIGINS=https://<ten-du-an>.vercel.app
MAX_TRELLIS_QUEUE=4
JOB_TTL_SECONDS=21600
RUN_TTL_SECONDS=21600
MAX_STORED_RUNS=20
SD35_LORA_SCALE=0.2
```

Nhap cung `APP_API_KEY` trong nut **Setup** cua giao dien. Khi `APP_API_KEY` duoc dat, moi API xu ly deu yeu cau header `X-API-Key`; health check van cong khai de ket noi.

## Khoi dong

1. Chay cac cell 1 den 10 trong `api-3d.ipynb`.
2. Cho health cua `sd35`, `sam2_dino` va `trellis` deu bao `ready: true`.
3. Mo frontend, nhap URL Ngrok va API key, sau do bam Connect.
4. Chon nhanh Van ban hoac Hinh anh va bam Run.

`/api/health` chi bao `ready: true` khi CUDA, checkpoint, module va ca ba worker deu san sang.

## API chinh

```text
POST /api/runs
POST /api/optimize_prompt
POST /api/parse_scene_graph
POST /api/generate_layout
POST /api/generate_image
POST /api/upload_image
POST /api/run_sam2
POST /api/generate_3d
GET  /api/generate_3d/status/{job_id}
POST /api/combine_scene
GET  /api/health
```

Nhung endpoint tao file yeu cau `run_id`. Client khong duoc phep yeu cau backend doc crop hoac GLB nam ngoai thu muc cua run dang hoat dong.

## Kiem thu

```bash
python -m unittest discover -s tests -v
```

Test tap trung vao so luong vat the, tham chieu khong bi dem lap, quan he lap lai va layout nhieu vat the.
