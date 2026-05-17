import logging
import sys
import threading
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch

from app.config import settings

logger = logging.getLogger(__name__)

INFERENCE_LOCK = threading.Lock()


@dataclass
class LoadedModels:
    detector: torch.nn.Module
    ocr: torch.nn.Module
    device: str


def _ensure_function_pkg_on_path() -> None:
    """Make `function.helper` / `function.utils_rotate` importable.

    The helper package lives at service_root/function/. Adding service_root
    to sys.path lets the upstream imports resolve unchanged.
    """
    root = str(settings.service_root)
    if root not in sys.path:
        sys.path.insert(0, root)


def _resolve_device() -> str:
    requested = settings.device.lower()
    if requested == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available; falling back to CPU")
        return "cpu"
    return requested


def load_models() -> LoadedModels:
    _ensure_function_pkg_on_path()
    device = _resolve_device()

    logger.info(
        "loading models (detector=%s, ocr=%s, yolov5=%s, device=%s)",
        settings.lp_detector_weights,
        settings.lp_ocr_weights,
        settings.yolov5_root,
        device,
    )

    detector = torch.hub.load(
        str(settings.yolov5_root),
        "custom",
        path=str(settings.lp_detector_weights),
        source="local",
        force_reload=False,
    )
    ocr = torch.hub.load(
        str(settings.yolov5_root),
        "custom",
        path=str(settings.lp_ocr_weights),
        source="local",
        force_reload=False,
    )
    ocr.conf = settings.ocr_conf

    if device == "cuda":
        detector.to("cuda")
        ocr.to("cuda")

    detector.eval()
    ocr.eval()

    _warmup(detector, ocr)
    return LoadedModels(detector=detector, ocr=ocr, device=device)


def _warmup(detector: torch.nn.Module, ocr: torch.nn.Module) -> None:
    dummy = np.zeros(
        (settings.detector_imgsz, settings.detector_imgsz, 3), dtype=np.uint8
    )
    try:
        with INFERENCE_LOCK, torch.no_grad():
            detector(dummy, size=settings.detector_imgsz)
            ocr(dummy)
        logger.info("warm-up inference complete")
    except Exception:
        logger.exception("warm-up inference failed (continuing)")
