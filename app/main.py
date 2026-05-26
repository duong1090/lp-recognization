import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import List

import torch
from fastapi import (
    FastAPI,
    File,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import PlainTextResponse
from pydantic import ValidationError

from app import pipeline
from app.config import settings
from app.image_io import decode_upload
from app.model_loader import load_models
from app.schemas import BatchPlatesResponse, HealthResponse, StreamInit
from app.stream import STREAM_SLOT, RtspFrameReader

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("lp-service")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("starting up — loading models")
    app.state.models = load_models()
    logger.info("models loaded on %s", app.state.models.device)
    yield
    logger.info("shutting down")


app = FastAPI(title="License Plate Recognition", lifespan=lifespan)


@app.get("/healthz", response_model=HealthResponse)
async def healthz() -> HealthResponse:
    models = getattr(app.state, "models", None)
    return HealthResponse(
        status="ok" if models is not None else "loading",
        device=models.device if models is not None else "unknown",
        cuda_available=torch.cuda.is_available(),
        models_loaded=models is not None,
    )


@app.post("/recognize", response_class=PlainTextResponse)
async def recognize(file: UploadFile = File(...)) -> str:
    img = await decode_upload(file)
    return await run_in_threadpool(pipeline.recognize_best, app.state.models, img)


@app.post("/recognize/batch", response_model=BatchPlatesResponse)
async def recognize_batch(files: List[UploadFile] = File(...)) -> BatchPlatesResponse:
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="no files supplied"
        )
    if len(files) > settings.max_batch:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"max {settings.max_batch} images per request",
        )
    imgs = [await decode_upload(f) for f in files]
    plates = await run_in_threadpool(pipeline.recognize_many, app.state.models, imgs)
    return BatchPlatesResponse(plates=plates)


@app.websocket("/ws/rtsp")
async def ws_rtsp(websocket: WebSocket) -> None:
    """Real-time license plate recognition over an RTSP stream.

    Protocol:
      1. Client connects to /ws/rtsp.
      2. Client sends one JSON message: {"rtsp_url": "rtsp://..."}.
      3. Server streams JSON results: {"plate", "frame_id", "ts", "latency_ms"}.
      4. Either side may disconnect to end the session.

    Only one concurrent stream is supported per worker (see STREAM_SLOT).
    """
    await websocket.accept()

    if not STREAM_SLOT.acquire(blocking=False):
        await websocket.send_json({"error": "busy"})
        await websocket.close(code=1013)
        return

    reader: RtspFrameReader | None = None
    try:
        try:
            init = StreamInit(**await websocket.receive_json())
        except (ValidationError, ValueError) as e:
            await websocket.send_json({"error": "bad_init", "detail": str(e)})
            await websocket.close(code=1003)
            return

        reader = RtspFrameReader(init.rtsp_url, reconnect_max=settings.rtsp_reconnect_max)
        try:
            await run_in_threadpool(reader.start)
        except Exception as e:
            logger.warning("rtsp open failed for %r: %s", init.rtsp_url, e)
            await websocket.send_json({"error": "rtsp_open_failed", "detail": str(e)})
            await websocket.close(code=1011)
            return

        last_id = 0
        while True:
            latest = reader.latest()
            if latest is None or latest[0] == last_id:
                if reader.fatal_error:
                    await websocket.send_json(
                        {"error": "stream_lost", "detail": reader.fatal_error}
                    )
                    await websocket.close(code=1011)
                    return
                await asyncio.sleep(0.01)
                continue

            frame_id, frame = latest
            last_id = frame_id
            t0 = time.perf_counter()
            result = await run_in_threadpool(
                pipeline.recognize_best_with_bbox, app.state.models, frame
            )
            h, w = frame.shape[:2]
            msg = {
                "plate": result.plate,
                "frame_id": frame_id,
                "ts": time.time(),
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                "confidence": result.confidence,
                "frame_width": w,
                "frame_height": h,
            }
            if result.bbox:
                msg["bbox"] = {
                    "x1": result.bbox[0],
                    "y1": result.bbox[1],
                    "x2": result.bbox[2],
                    "y2": result.bbox[3],
                }
            await websocket.send_json(msg)
    except WebSocketDisconnect:
        logger.info("ws client disconnected")
    finally:
        if reader is not None:
            reader.stop()
        STREAM_SLOT.release()
