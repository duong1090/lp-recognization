import logging
import re
from dataclasses import dataclass
from typing import List, Tuple, Optional

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

# Vietnamese plate: 2-digit province + 1-2 letter series + numeric tail
# e.g. "51A12345", "29AB1234", "51A-12345"
_PLATE_RE = re.compile(r'^(\d{2}[A-Z]{1,2}-?)(.+)$')


def _refine_plate(plate: str) -> str:
    """Fix common OCR misreads in the numeric tail: B→8, D→0."""
    if plate == UNKNOWN:
        return plate
    m = _PLATE_RE.match(plate)
    if not m:
        return plate
    return m.group(1) + m.group(2).replace('B', '8').replace('D', '0')


@dataclass
class RecognitionResult:
    plate: str
    bbox: Optional[Tuple[int, int, int, int]] = None  # (x1, y1, x2, y2)
    confidence: float = 0.0


def _detect(detector, img_bgr: np.ndarray):
    return detector(img_bgr, size=settings.detector_imgsz)


def _ocr_with_rotations(ocr, crop: np.ndarray, *, label: str, verbose: bool = True) -> str:
    ch, cw = crop.shape[:2]
    for change_contrast in (0, 1):
        for center_thres in (0, 1):
            deskewed = deskew(crop, change_contrast, center_thres)
            lp = read_plate(ocr, deskewed)
            if verbose:
                logger.info(
                    "ocr[%s] contrast=%d center=%d size=%dx%d -> %s",
                    label, change_contrast, center_thres, cw, ch, lp,
                )
            if lp != UNKNOWN:
                return _refine_plate(lp)
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
            lp = _refine_plate(read_plate(models.ocr, img_bgr))
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


def recognize_best_with_bbox(
    models: LoadedModels, img_bgr: np.ndarray, *, verbose: bool = True
) -> RecognitionResult:
    """Same as recognize_best but returns bbox and confidence alongside the plate."""
    with INFERENCE_LOCK, torch.no_grad():
        h, w = img_bgr.shape[:2]

        plates = _detect(models.detector, img_bgr)
        df = plates.pandas().xyxy[0]

        if df.empty:
            lp = _refine_plate(read_plate(models.ocr, img_bgr))
            return RecognitionResult(plate=lp)

        # Only reached when at least one candidate found — always log this
        logger.info("detect: %d candidate(s) [%dx%d]", len(df), w, h)
        df = df.sort_values("confidence", ascending=False)
        for idx, (_, row) in enumerate(df.iterrows()):
            x1 = max(int(row["xmin"]), 0)
            y1 = max(int(row["ymin"]), 0)
            x2 = min(int(row["xmax"]), w)
            y2 = min(int(row["ymax"]), h)
            conf = float(row["confidence"])
            if x2 <= x1 or y2 <= y1:
                if verbose:
                    logger.info("crop[%d]: degenerate bbox, skipped", idx)
                continue
            crop = img_bgr[y1:y2, x1:x2]
            bbox = (x1, y1, x2, y2)
            if verbose:
                logger.info(
                    "crop[%d]: bbox=(%d,%d,%d,%d) size=%dx%d conf=%.3f",
                    idx, x1, y1, x2, y2, x2 - x1, y2 - y1, conf,
                )

            lp = _ocr_with_rotations(models.ocr, crop, label="raw", verbose=verbose)
            if lp != UNKNOWN:
                logger.info("recognize_bbox -> %s (crop[%d], raw)", lp, idx)
                return RecognitionResult(plate=lp, bbox=bbox, confidence=conf)

            if settings.ocr_preprocess:
                lp = _ocr_with_rotations(
                    models.ocr, enhance_for_ocr(crop), label="prep", verbose=verbose
                )
                if lp != UNKNOWN:
                    logger.info("recognize_bbox -> %s (crop[%d], preprocessed)", lp, idx)
                    return RecognitionResult(plate=lp, bbox=bbox, confidence=conf)

                lp = _ocr_with_rotations(
                    models.ocr,
                    enhance_for_ocr(crop, invert=True),
                    label="prep-inv",
                    verbose=verbose,
                )
                if lp != UNKNOWN:
                    logger.info("recognize_bbox -> %s (crop[%d], preprocessed+inverted)", lp, idx)
                    return RecognitionResult(plate=lp, bbox=bbox, confidence=conf)

        if verbose:
            logger.info("recognize_bbox -> unknown (all candidates exhausted)")
        return RecognitionResult(plate=UNKNOWN)


def recognize_many(models: LoadedModels, imgs: List[np.ndarray]) -> List[str]:
    return [recognize_best(models, img) for img in imgs]
