from typing import List

import numpy as np
import torch

from app.config import settings
from app.model_loader import INFERENCE_LOCK, LoadedModels

# These imports resolve because model_loader._ensure_upstream_on_path()
# inserts License-Plate-Recognition/ onto sys.path at startup.
from function.helper import read_plate
from function.utils_rotate import deskew

UNKNOWN = "unknown"


def _detect(detector, img_bgr: np.ndarray):
    return detector(img_bgr, size=settings.detector_imgsz)


def _ocr_with_rotations(ocr, crop: np.ndarray) -> str:
    for change_contrast in (0, 1):
        for center_thres in (0, 1):
            deskewed = deskew(crop, change_contrast, center_thres)
            lp = read_plate(ocr, deskewed)
            if lp != UNKNOWN:
                return lp
    return UNKNOWN


def recognize_best(models: LoadedModels, img_bgr: np.ndarray) -> str:
    """Return the highest-confidence successfully-decoded plate, or 'unknown'."""
    with INFERENCE_LOCK, torch.no_grad():
        plates = _detect(models.detector, img_bgr)
        df = plates.pandas().xyxy[0]

        if df.empty:
            # Original pipeline: fall back to OCR on the whole frame.
            return read_plate(models.ocr, img_bgr)

        df = df.sort_values("confidence", ascending=False)
        h, w = img_bgr.shape[:2]
        for _, row in df.iterrows():
            x1 = max(int(row["xmin"]), 0)
            y1 = max(int(row["ymin"]), 0)
            x2 = min(int(row["xmax"]), w)
            y2 = min(int(row["ymax"]), h)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = img_bgr[y1:y2, x1:x2]
            lp = _ocr_with_rotations(models.ocr, crop)
            if lp != UNKNOWN:
                return lp
        return UNKNOWN


def recognize_many(models: LoadedModels, imgs: List[np.ndarray]) -> List[str]:
    return [recognize_best(models, img) for img in imgs]
