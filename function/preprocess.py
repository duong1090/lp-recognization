"""OpenCV preprocessing for plate crops before OCR.

Applied as a fallback when raw-crop OCR returns 'unknown'. Order:
upscale (more pixels for later stages) → polarity normalize (invert dark-
background plates so characters appear dark-on-light, matching the OCR
model's dominant training distribution) → bilateral denoise → CLAHE on
the LAB L-channel.
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger("lp-service.preprocess")

_UPSCALE_MAX_SIDE_PX = 200  # only upscale crops smaller than this on long side
_UPSCALE_FACTOR = 2.0
_BILATERAL = dict(d=5, sigmaColor=50, sigmaSpace=50)
_CLAHE_CLIP = 2.0
_CLAHE_TILES = (8, 8)
# Median grayscale value below which we treat the crop as dark-background
# (blue government / red military plates). Tuned for Vietnamese plates.
_DARK_BG_MEDIAN_THRESHOLD = 127


def enhance_for_ocr(crop_bgr: np.ndarray) -> np.ndarray:
    img = crop_bgr
    h, w = img.shape[:2]
    upscaled = False
    if max(h, w) < _UPSCALE_MAX_SIDE_PX:
        img = cv2.resize(
            img,
            (int(w * _UPSCALE_FACTOR), int(h * _UPSCALE_FACTOR)),
            interpolation=cv2.INTER_CUBIC,
        )
        upscaled = True

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    median_v = float(np.median(gray))
    inverted = median_v < _DARK_BG_MEDIAN_THRESHOLD
    if inverted:
        img = cv2.bitwise_not(img)

    img = cv2.bilateralFilter(img, **_BILATERAL)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=_CLAHE_CLIP, tileGridSize=_CLAHE_TILES)
    l_channel = clahe.apply(l_channel)
    out = cv2.cvtColor(cv2.merge((l_channel, a, b)), cv2.COLOR_LAB2BGR)
    logger.info(
        "enhance: %dx%d -> %dx%d (upscale=%s, invert=%s, median=%.0f)",
        w, h, out.shape[1], out.shape[0], upscaled, inverted, median_v,
    )
    return out
