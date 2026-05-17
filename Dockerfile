# Jetson Orin Nano, JetPack 6 / L4T r36.x.
# l4t-ml bundles PyTorch + torchvision + OpenCV + numpy + pandas for L4T.
# (NVIDIA's l4t-pytorch repo has no r36 tags; l4t-ml is the JetPack-6 successor.)
# For JetPack 5 use: nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3
FROM nvcr.io/nvidia/l4t-ml:r36.2.0-py3

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      git \
      libgl1 \
      libglib2.0-0 \
      ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# YOLOv5 v6.1 pinned — matches the .pt files trained against this release.
ARG YOLOV5_SHA=3752807c0b8af03d42de478fbcbf338ec4546a6c
RUN git clone https://github.com/ultralytics/yolov5.git /opt/lp/yolov5 \
 && git -C /opt/lp/yolov5 checkout ${YOLOV5_SHA} \
 && rm -rf /opt/lp/yolov5/.git

WORKDIR /opt/lp/service

COPY requirements.txt ./requirements.txt
# Resolve deps normally — the base image already has torch/torchvision/numpy/
# opencv/pandas, so pip will skip those and only install the FastAPI side and
# yolov5 v6.1's small runtime helpers.
RUN pip3 install -r requirements.txt

# Service code + helper functions + pre-trained weights (all bundled).
COPY app ./app
COPY function ./function
COPY model ./model

ENV SERVICE_ROOT=/opt/lp/service \
    YOLOV5_ROOT=/opt/lp/yolov5 \
    LP_DETECTOR_WEIGHTS=/opt/lp/service/model/LP_detector_nano_61.pt \
    LP_OCR_WEIGHTS=/opt/lp/service/model/LP_ocr_nano_62.pt \
    DEVICE=cuda \
    PYTHONPATH=/opt/lp/service

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
