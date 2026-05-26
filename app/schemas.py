from typing import List, Optional

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    device: str
    cuda_available: bool
    models_loaded: bool


class BatchPlatesResponse(BaseModel):
    plates: List[str]


class StreamInit(BaseModel):
    rtsp_url: str


class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class StreamResult(BaseModel):
    plate: str
    frame_id: int
    ts: float
    latency_ms: float
    bbox: Optional[BoundingBox] = None
    confidence: float = 0.0
    frame_width: int = 0
    frame_height: int = 0


class StreamError(BaseModel):
    error: str
    detail: Optional[str] = None
