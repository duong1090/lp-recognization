import cv2
import numpy as np
from fastapi import HTTPException, UploadFile, status

from app.config import settings


async def decode_upload(upload: UploadFile) -> np.ndarray:
    data = await upload.read()
    if len(data) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"empty upload: {upload.filename}",
        )
    if len(data) > settings.max_image_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"{upload.filename} exceeds {settings.max_image_bytes} bytes",
        )
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None or img.ndim != 3 or img.shape[2] != 3:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"could not decode image: {upload.filename}",
        )
    return img
