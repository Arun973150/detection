"""Pydantic models for API request/response."""

from pydantic import BaseModel, Field


class DetectorInfo(BaseModel):
    name: str
    score: float
    confidence: float
    error: str | None = None
    details: dict | None = None


class LayerInfo(BaseModel):
    layer: str
    score: float
    flags: list[str] = Field(default_factory=list)
    detectors: list[DetectorInfo] = Field(default_factory=list)


class AnalysisResponse(BaseModel):
    verdict: str
    confidence: float
    overall_score: float
    flags: list[str] = Field(default_factory=list)
    processing_time_ms: float
    breakdown: list[LayerInfo] = Field(default_factory=list)
    heatmap_base64: str | None = None


class HealthResponse(BaseModel):
    status: str
    gpu_available: bool
    layers_loaded: int
