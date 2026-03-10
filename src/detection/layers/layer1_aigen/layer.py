"""Layer 1: AI-Generated Image Detection — 4 model ensemble with purification awareness."""

import numpy as np

from detection.base.layer import BaseLayer
from detection.base.result import LayerResult
from detection.layers.layer1_aigen.drct import DRCTDetector
from detection.layers.layer1_aigen.fatformer import FatFormerDetector
from detection.layers.layer1_aigen.npr import NPRDetector
from detection.layers.layer1_aigen.univfd import UnivFDDetector


class AIGeneratedDetectionLayer(BaseLayer):
    name = "layer1_aigen"
    order = 1.0

    def setup(self) -> None:
        detector_configs = [
            ("univfd", UnivFDDetector),
            ("npr", NPRDetector),
            ("drct", DRCTDetector),
            ("fatformer", FatFormerDetector),
        ]
        for key, cls in detector_configs:
            cfg = self.config.get(key, {})
            if cfg.get("enabled", True):
                self.detectors.append(cls(cfg, self.device))

    def analyze(self, image: np.ndarray, context: dict) -> LayerResult:
        results = []
        has_purification = "purified_images" in context

        for detector in self.detectors:
            if has_purification:
                result = self._run_with_purification(detector, image, context)
            else:
                result = self._run_detector_safe(detector, image)
            results.append(result)

        # Build weights map from config
        weights = {}
        for det in self.detectors:
            det_config = self.config.get(det.name, {})
            weights[det.name] = det_config.get("weight", 1.0)

        agg_score = self._aggregate_scores(results, weights)

        # Check for adversarial flags
        flags = []
        for r in results:
            if r.details.get("adversarial_flag"):
                flags.append(f"adversarial_detected_{r.detector_name}")

        return LayerResult(
            layer_name=self.name,
            order=self.order,
            aggregated_score=agg_score,
            detector_results=results,
            flags=flags,
        )
