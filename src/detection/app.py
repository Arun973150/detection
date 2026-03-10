"""FastAPI application for image authenticity detection."""

import base64
import logging
from contextlib import asynccontextmanager
from io import BytesIO

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile

from detection.config import load_config
from detection.device import get_device
from detection.pipeline import Pipeline
from detection.schemas import AnalysisResponse, HealthResponse
from detection.utils.heatmap import create_heatmap_overlay
from detection.utils.image_io import load_image

logger = logging.getLogger(__name__)

pipeline: Pipeline | None = None


def build_pipeline(config: dict) -> Pipeline:
    """Build the full pipeline with all layers from config."""
    from detection.layers import create_all_layers

    device = get_device(config.get("device", {}).get("prefer", "cuda"))
    p = Pipeline(config)

    for layer in create_all_layers(config, device):
        p.register_layer(layer)

    p.setup()
    return p


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    config = load_config()
    pipeline = build_pipeline(config)
    logger.info("Pipeline initialized")
    yield
    pipeline = None


app = FastAPI(
    title="Image Authenticity Detection API",
    version="0.1.0",
    lifespan=lifespan,
)


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(file: UploadFile = File(...)):
    """Analyze an image for authenticity."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    contents = await file.read()
    try:
        image = load_image(contents)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    result = pipeline.run(image)
    response_data = result.to_dict()

    # Encode heatmap as base64 PNG if available
    heatmap_b64 = None
    if result.merged_heatmap is not None:
        overlay = create_heatmap_overlay(image, result.merged_heatmap)
        from PIL import Image as PILImage

        pil_img = PILImage.fromarray(overlay)
        buf = BytesIO()
        pil_img.save(buf, format="PNG")
        heatmap_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return AnalysisResponse(
        verdict=response_data["verdict"],
        confidence=response_data["confidence"],
        overall_score=response_data["overall_score"],
        flags=response_data["flags"],
        processing_time_ms=response_data["processing_time_ms"],
        breakdown=response_data["breakdown"],
        heatmap_base64=heatmap_b64,
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        gpu_available=torch.cuda.is_available(),
        layers_loaded=len(pipeline.layers) if pipeline else 0,
    )
