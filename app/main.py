import logging
from contextlib import asynccontextmanager
from typing import List

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import PlainTextResponse

from app import pipeline
from app.config import settings
from app.image_io import decode_upload
from app.model_loader import load_models
from app.schemas import BatchPlatesResponse, HealthResponse

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
