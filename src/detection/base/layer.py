"""Abstract base class for pipeline layers."""

import logging
from abc import ABC, abstractmethod

import numpy as np

from detection.base.detector import BaseDetector
from detection.base.result import DetectorResult, LayerResult

logger = logging.getLogger(__name__)


class BaseLayer(ABC):
    """A pipeline stage that runs one or more detectors and aggregates their outputs.

    Each layer receives the image and a context dict accumulated from prior layers.
    """

    name: str = "base_layer"
    order: float = 0.0

    def __init__(self, config: dict, device: str = "cpu"):
        self.config = config
        self.device = device
        self.detectors: list[BaseDetector] = []
        self.enabled: bool = config.get("enabled", True)

    @abstractmethod
    def setup(self) -> None:
        """Instantiate detectors based on config. Called once at startup."""
        ...

    @abstractmethod
    def analyze(self, image: np.ndarray, context: dict) -> LayerResult:
        """Run all detectors and aggregate results.

        Args:
            image: RGB numpy array, uint8, shape (H, W, 3)
            context: accumulated results from prior layers (mutable dict)

        Returns:
            LayerResult with per-detector results and aggregated score.
        """
        ...

    def _run_detector_safe(
        self, detector: BaseDetector, image: np.ndarray, metadata: dict | None = None
    ) -> DetectorResult:
        """Run a single detector with error handling."""
        try:
            return detector.analyze(image, metadata)
        except Exception as e:
            logger.error(f"Detector {detector.name} failed: {e}", exc_info=True)
            return DetectorResult(
                detector_name=detector.name,
                score=0.0,
                confidence=0.0,
                error=str(e),
            )

    def _run_with_purification(
        self, detector: BaseDetector, image: np.ndarray, context: dict
    ) -> DetectorResult:
        """Run detector on original + purified variants, flag divergence."""
        original_result = self._run_detector_safe(detector, image)
        purified_images = context.get("purified_images", {})

        if not purified_images:
            return original_result

        purified_scores = []
        for variant_name, variant_img in purified_images.items():
            pr = self._run_detector_safe(detector, variant_img)
            purified_scores.append(pr.score)

        if not purified_scores:
            return original_result

        max_divergence = max(abs(s - original_result.score) for s in purified_scores)
        threshold = self.config.get("divergence_threshold", 0.20)

        if max_divergence > threshold:
            # Use max score across all variants (most conservative)
            max_score = max(max(purified_scores), original_result.score)
            original_result.score = max_score
            original_result.details["adversarial_flag"] = True
            original_result.details["max_divergence"] = round(max_divergence, 4)

        return original_result

    def _aggregate_scores(
        self, results: list[DetectorResult], weights: dict[str, float] | None = None
    ) -> float:
        """Weighted average of detector scores, redistributing weight from failed detectors."""
        valid = [(r, weights.get(r.detector_name, 1.0) if weights else 1.0)
                 for r in results if r.error is None]
        if not valid:
            return 0.0

        total_weight = sum(w for _, w in valid)
        if total_weight == 0:
            return 0.0

        return sum(r.score * w for r, w in valid) / total_weight
