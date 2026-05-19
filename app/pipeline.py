import logging
from typing import List

import numpy as np
import torch

from app.config import settings
from app.model_loader import INFERENCE_LOCK, LoadedModels

# These imports resolve because model_loader._ensure_upstream_on_path()
# inserts License-Plate-Recognition/ onto sys.path at startup.
from function.helper import read_plate
from function.preprocess import enhance_for_ocr
from function.utils_rotate import deskew

logger = logging.getLogger("lp-service.pipeline")

UNKNOWN = "unknown"


def _detect(detector, img_bgr: np.ndarray):
    return detector(img_bgr, size=settings.detector_imgsz)


def _ocr_with_rotations(ocr, crop: np.ndarray, *, label: str) -> str:
    ch, cw = crop.shape[:2]
    for change_contrast in (0, 1):
        for center_thres in (0, 1):
            deskewed = deskew(crop, change_contrast, center_thres)
            lp = read_plate(ocr, deskewed)
            logger.info(
                "ocr[%s] contrast=%d center=%d size=%dx%d -> %s",
                label, change_contrast, center_thres, cw, ch, lp,
            )
            if lp != UNKNOWN:
                return lp
    return UNKNOWN


def recognize_best(models: LoadedModels, img_bgr: np.ndarray) -> str:
    """Return the highest-confidence successfully-decoded plate, or 'unknown'."""
    with INFERENCE_LOCK, torch.no_grad():
        h, w = img_bgr.shape[:2]
        logger.info("recognize: input %dx%d", w, h)

        plates = _detect(models.detector, img_bgr)
        df = plates.pandas().xyxy[0]
        logger.info("detect: %d candidate(s)", len(df))

        if df.empty:
            # Original pipeline: fall back to OCR on the whole frame.
            lp = read_plate(models.ocr, img_bgr)
            logger.info("recognize -> %s (whole-frame fallback)", lp)
            return lp

        df = df.sort_values("confidence", ascending=False)
        for idx, (_, row) in enumerate(df.iterrows()):
            x1 = max(int(row["xmin"]), 0)
            y1 = max(int(row["ymin"]), 0)
            x2 = min(int(row["xmax"]), w)
            y2 = min(int(row["ymax"]), h)
            conf = float(row["confidence"])
            if x2 <= x1 or y2 <= y1:
                logger.info("crop[%d]: degenerate bbox, skipped", idx)
                continue
            crop = img_bgr[y1:y2, x1:x2]
            logger.info(
                "crop[%d]: bbox=(%d,%d,%d,%d) size=%dx%d conf=%.3f",
                idx, x1, y1, x2, y2, x2 - x1, y2 - y1, conf,
            )

            lp = _ocr_with_rotations(models.ocr, crop, label="raw")
            if lp != UNKNOWN:
                logger.info("recognize -> %s (crop[%d], raw)", lp, idx)
                return lp

            if settings.ocr_preprocess:
                logger.info("crop[%d]: raw failed, trying preprocessed", idx)
                lp = _ocr_with_rotations(
                    models.ocr, enhance_for_ocr(crop), label="prep"
                )
                if lp != UNKNOWN:
                    logger.info("recognize -> %s (crop[%d], preprocessed)", lp, idx)
                    return lp

                logger.info("crop[%d]: preprocessed failed, trying inverted", idx)
                lp = _ocr_with_rotations(
                    models.ocr,
                    enhance_for_ocr(crop, invert=True),
                    label="prep-inv",
                )
                if lp != UNKNOWN:
                    logger.info(
                        "recognize -> %s (crop[%d], preprocessed+inverted)", lp, idx
                    )
                    return lp

        logger.info("recognize -> unknown (all candidates exhausted)")
        return UNKNOWN


def recognize_many(models: LoadedModels, imgs: List[np.ndarray]) -> List[str]:
    return [recognize_best(models, img) for img in imgs]
