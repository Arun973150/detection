"""Layer 4: Watermark Detection — SynthID g-function, invisible-watermark, C2PA."""

import numpy as np

from detection.base.layer import BaseLayer
from detection.base.result import LayerResult
from detection.layers.layer4_watermark.c2pa_check import C2PADetector
from detection.layers.layer4_watermark.gfunction import GFunctionDetector
from detection.layers.layer4_watermark.invisible_wm import InvisibleWatermarkDetector


class WatermarkDetectionLayer(BaseLayer):
    name = "layer4_watermark"
    order = 4.0

    def setup(self) -> None:
        if self.config.get("gfunction", {}).get("enabled", True):
            self.detectors.append(GFunctionDetector(self.config.get("gfunction", {}), self.device))

        if self.config.get("invisible_watermark", {}).get("enabled", True):
            self.detectors.append(
                InvisibleWatermarkDetector(self.config.get("invisible_watermark", {}), self.device)
            )

        if self.config.get("c2pa", {}).get("enabled", True):
            self.detectors.append(C2PADetector(self.config.get("c2pa", {}), self.device))

    def analyze(self, image: np.ndarray, context: dict) -> LayerResult:
        results = []
        for detector in self.detectors:
            result = self._run_detector_safe(detector, image, context)
            results.append(result)

        # For watermarks, any positive detection is strong evidence
        # Use max score rather than average
        scores = [r.score for r in results if r.error is None]
        agg_score = max(scores) if scores else 0.0

        flags = []
        for r in results:
            state = r.details.get("state", "")
            if state == "watermark_detected":
                flags.append(f"watermark_detected_{r.detector_name}")
            if r.details.get("c2pa_present"):
                flags.append("c2pa_credentials_found")
            if r.details.get("ai_generated"):
                flags.append("c2pa_ai_generated")

        return LayerResult(
            layer_name=self.name,
            order=self.order,
            aggregated_score=agg_score,
            detector_results=results,
            flags=flags,
        )
