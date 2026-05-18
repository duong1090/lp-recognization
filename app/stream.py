"""RTSP frame reader for the real-time WebSocket endpoint.

Runs `cv2.VideoCapture(url).read()` in a background daemon thread and keeps
only the most recent frame in a single-slot buffer. The WebSocket handler
pulls the latest frame, runs inference, and sends the result back. Frames
produced while inference is busy are silently overwritten — this is the
intended drop-old backpressure strategy.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("lp-service.stream")

# Module-level slot enforces a single active stream at a time. The Jetson
# memory budget and the global INFERENCE_LOCK in pipeline.py mean we can't
# usefully run multiple concurrent streams in this worker.
STREAM_SLOT: threading.Lock = threading.Lock()


class RtspFrameReader:
    def __init__(self, url: str, reconnect_max: int = 3) -> None:
        self.url = url
        self.reconnect_max = reconnect_max

        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._frame_id = 0
        self._latest: Optional[np.ndarray] = None
        self.fatal_error: Optional[str] = None

    def start(self) -> None:
        cap = cv2.VideoCapture(self.url)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"cv2.VideoCapture could not open {self.url!r}")
        # Minimize internal buffering so cap.read() returns near-live frames.
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        self._cap = cap
        self._thread = threading.Thread(
            target=self._loop, name="rtsp-reader", daemon=True
        )
        self._thread.start()

    def _loop(self) -> None:
        assert self._cap is not None
        reconnects = 0
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok or frame is None:
                if reconnects >= self.reconnect_max:
                    self.fatal_error = (
                        f"rtsp read failed after {reconnects} reconnect attempts"
                    )
                    logger.warning("stream %r giving up: %s", self.url, self.fatal_error)
                    return
                reconnects += 1
                logger.info(
                    "stream %r read failed, reconnect attempt %d/%d",
                    self.url,
                    reconnects,
                    self.reconnect_max,
                )
                try:
                    self._cap.release()
                except Exception:
                    pass
                self._cap = cv2.VideoCapture(self.url)
                if self._cap.isOpened():
                    try:
                        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    except Exception:
                        pass
                continue
            reconnects = 0
            with self._lock:
                self._frame_id += 1
                self._latest = frame

    def latest(self) -> Optional[Tuple[int, np.ndarray]]:
        with self._lock:
            if self._latest is None:
                return None
            return self._frame_id, self._latest

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
