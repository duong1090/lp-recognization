from typing import List

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    device: str
    cuda_available: bool
    models_loaded: bool


class BatchPlatesResponse(BaseModel):
    plates: List[str]
