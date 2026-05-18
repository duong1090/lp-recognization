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


class StreamResult(BaseModel):
    plate: str
    frame_id: int
    ts: float
    latency_ms: float


class StreamError(BaseModel):
    error: str
    detail: Optional[str] = None
