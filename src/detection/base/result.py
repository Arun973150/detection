"""Result dataclasses used across all layers and detectors."""

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class DetectorResult:
    """Result from a single detector algorithm."""

    detector_name: str
    score: float  # 0.0 = authentic, 1.0 = fake/manipulated
    confidence: float  # how confident the detector is in its score
    heatmap: Optional[np.ndarray] = None  # (H, W) float32, 0-1 suspicion map
    details: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None  # non-None if detector failed gracefully


@dataclass
class LayerResult:
    """Aggregated result from a pipeline layer (multiple detectors)."""

    layer_name: str
    order: float
    aggregated_score: float
    detector_results: list[DetectorResult] = field(default_factory=list)
    heatmap: Optional[np.ndarray] = None  # merged heatmap from all detectors in layer
    flags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    """Final output from the full detection pipeline."""

    verdict: str  # "authentic", "ai_generated", "manipulated", "deepfake", "uncertain"
    confidence: float
    overall_score: float
    layer_results: list[LayerResult] = field(default_factory=list)
    merged_heatmap: Optional[np.ndarray] = None
    flags: list[str] = field(default_factory=list)
    processing_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        result = {
            "verdict": self.verdict,
            "confidence": round(self.confidence, 4),
            "overall_score": round(self.overall_score, 4),
            "flags": self.flags,
            "processing_time_ms": round(self.processing_time_ms, 1),
            "breakdown": [],
        }
        for lr in self.layer_results:
            layer_info = {
                "layer": lr.layer_name,
                "score": round(lr.aggregated_score, 4),
                "flags": lr.flags,
                "detectors": [],
            }
            for dr in lr.detector_results:
                det_info = {
                    "name": dr.detector_name,
                    "score": round(dr.score, 4),
                    "confidence": round(dr.confidence, 4),
                }
                if dr.error:
                    det_info["error"] = dr.error
                if dr.details:
                    det_info["details"] = dr.details
                layer_info["detectors"].append(det_info)
            result["breakdown"].append(layer_info)
        return result
