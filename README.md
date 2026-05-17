# LP Recognition Service

Self-contained FastAPI service for Vietnamese license plate recognition.
Bundles the pre-trained YOLOv5 weights and OCR helpers from the upstream
[License-Plate-Recognition](https://github.com/Marsmallotr/License-Plate-Recognition)
project. Targets the Jetson Orin Nano (Docker + systemd); also runs on CPU
for local development.

```
service/
├── app/        # FastAPI app, pipeline, schemas
├── function/   # OCR helpers (helper.py, utils_rotate.py)
├── model/      # Pre-trained YOLOv5 weights (.pt)
├── systemd/    # lp-service.service unit
├── Dockerfile
├── requirements.txt
└── README.md
```

## Endpoints

| Method | Path               | Body                                | Response                                          |
|--------|--------------------|-------------------------------------|---------------------------------------------------|
| GET    | `/healthz`         | –                                   | `{status, device, cuda_available, models_loaded}` |
| POST   | `/recognize`       | `multipart/form-data` field `file`  | `text/plain`, e.g. `30A-12345` or `unknown`       |
| POST   | `/recognize/batch` | `multipart/form-data` field `files` | `{"plates": ["30A-12345", "unknown", ...]}`       |

`/recognize` returns a plain-text body (per the service contract). `/recognize/batch`
preserves input order in the `plates` array.

## Bundled weights

All four weight files are committed under `model/`:

| File                       | Size  | Use                                |
|----------------------------|-------|------------------------------------|
| `LP_detector_nano_61.pt`   | 3.6 MB| Plate detection (Jetson default)   |
| `LP_ocr_nano_62.pt`        | 3.8 MB| Character OCR (Jetson default)     |
| `LP_detector.pt`           | 40 MB | Plate detection (full, more accurate) |
| `LP_ocr.pt`                | 41 MB | Character OCR (full, more accurate)|

The service uses the **nano** weights by default. Override with env vars
(`LP_DETECTOR_WEIGHTS`, `LP_OCR_WEIGHTS`) to switch to the full models.

> **Note**: the upstream repo tracks these via plain git (not LFS). If you
> clone fresh and the `model/` directory is empty, run
> `git checkout HEAD -- model/` inside the upstream repo to restore them.

## Local development (CPU)

```bash
cd service
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Runtime extras shipped by the L4T base image, needed locally:
pip install opencv-python torch torchvision numpy

# YOLOv5 v6.1 (pinned commit) — required by torch.hub.load(source="local")
git clone https://github.com/ultralytics/yolov5.git /tmp/yolov5
git -C /tmp/yolov5 checkout 3752807c0b8af03d42de478fbcbf338ec4546a6c

YOLOV5_ROOT=/tmp/yolov5 \
DEVICE=cpu \
uvicorn app.main:app --reload
```

(`SERVICE_ROOT`, `LP_DETECTOR_WEIGHTS`, `LP_OCR_WEIGHTS` default to paths
inside this folder — no env override needed for the bundled weights.)

Smoke tests:

```bash
curl -s http://localhost:8000/healthz | jq

# upstream test_image/ is gitignored — use any plate photo you have, e.g.:
curl -s -X POST http://localhost:8000/recognize \
  -F "file=@/path/to/plate.jpg"

curl -s -X POST http://localhost:8000/recognize/batch \
  -F "files=@/path/to/plate1.jpg" \
  -F "files=@/path/to/plate2.jpg" | jq
```

## Build & run on Jetson Orin Nano

1. Confirm the L4T release on the device:

   ```bash
   cat /etc/nv_tegra_release
   # JetPack 6 / L4T r36  → nvcr.io/nvidia/l4t-ml:r36.2.0-py3 (default)
   # JetPack 5 / L4T r35  → switch FROM to nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3
   ```

   The default base (`l4t-ml:r36.2.0-py3`) is ~12 GB; first pull on the
   Jetson will take a while. It ships PyTorch + torchvision + OpenCV +
   numpy + pandas pre-built for L4T r36, so the service install is fast.

2. Build from the `service/` directory (build context is self-contained):

   ```bash
   cd service
   docker build -t lp-service:latest .
   ```

3. Sanity-check CUDA inside the container:

   ```bash
   docker run --rm --runtime nvidia lp-service:latest \
     python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   # Expect: True Orin
   ```

4. Run:

   ```bash
   docker run --rm --runtime nvidia -p 8000:8000 \
     --name lp-service lp-service:latest
   ```

   First boot takes ~30–90 s (model load + CUDA kernel JIT). Watch for
   `Application startup complete.` then re-run the curl tests against
   `http://<jetson-ip>:8000`.

5. Recommended Jetson tuning for sustained throughput:

   ```bash
   sudo nvpmodel -m 0    # max-N power mode
   sudo jetson_clocks    # lock clocks
   ```

## Auto-start via systemd

```bash
sudo cp systemd/lp-service.service /etc/systemd/system/lp-service.service
sudo systemctl daemon-reload
sudo systemctl enable --now lp-service
sudo systemctl status lp-service
```

The unit auto-restarts the container on failure and allows up to 5 minutes
for the cold-start model load.

## Configuration

Env vars (defaults shown):

| Var                   | Default                                              | Notes                                |
|-----------------------|------------------------------------------------------|--------------------------------------|
| `SERVICE_ROOT`        | parent of `app/` (auto-detected)                     | Used to resolve other defaults       |
| `YOLOV5_ROOT`         | `/opt/lp/yolov5`                                     | YOLOv5 v6.1 checkout                 |
| `LP_DETECTOR_WEIGHTS` | `<service_root>/model/LP_detector_nano_61.pt`        |                                      |
| `LP_OCR_WEIGHTS`      | `<service_root>/model/LP_ocr_nano_62.pt`             |                                      |
| `OCR_CONF`            | `0.60`                                               | Min confidence for OCR detections    |
| `DETECTOR_IMGSZ`      | `640`                                                | YOLO inference resolution            |
| `DEVICE`              | `cuda`                                               | Falls back to CPU if CUDA missing    |
| `MAX_BATCH`           | `8`                                                  | Max images per batch request         |
| `MAX_IMAGE_BYTES`     | `10000000`                                           | Per-image size cap (10 MB)           |

## Notes & caveats

- **YOLOv5 v6.1 pinned** (commit `3752807c…`): newer master is
  API-incompatible with the bundled `.pt` files. Do not bump without
  retraining.
- **Single uvicorn worker**: the Orin Nano's 8 GB shared memory can't host
  two model copies. Inference is serialized inside the worker via a
  `threading.Lock` (YOLOv5 `AutoShape` is not concurrency-safe).
- **No filesystem writes per request** — the upstream `lp_image.py` writes
  `crop.jpg`; this service skips that.
- **`helper.read_plate` returns `"unknown"`** for plates with <7 or >10
  detected characters; this is surfaced as-is in the API response.
