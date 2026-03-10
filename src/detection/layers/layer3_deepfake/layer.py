"""Layer 3: Deepfake Detection — Reality Defender API + local fallback."""

import numpy as np

from detection.base.layer import BaseLayer
from detection.base.result import LayerResult
from detection.layers.layer3_deepfake.local_fallback import LocalDeepfakeDetector
from detection.layers.layer3_deepfake.reality_defender import RealityDefenderDetector


class DeepfakeDetectionLayer(BaseLayer):
    name = "layer3_deepfake"
    order = 3.0

    def setup(self) -> None:
        rd_cfg = self.config.get("reality_defender", {})
        if rd_cfg.get("enabled", True):
            self.detectors.append(RealityDefenderDetector(rd_cfg, self.device))

        fb_cfg = self.config.get("local_fallback", {})
        if fb_cfg.get("enabled", True):
            self.detectors.append(LocalDeepfakeDetector(fb_cfg, self.device))

    def analyze(self, image: np.ndarray, context: dict) -> LayerResult:
        results = []

        for detector in self.detectors:
            result = self._run_detector_safe(detector, image)
            results.append(result)

            # If Reality Defender succeeds, its result is primary
            if detector.name == "reality_defender" and result.error is None:
                break  # Don't need local fallback

        # Filter to successful results
        successful = [r for r in results if r.error is None]

        # If no faces found by any detector, score is 0
        all_no_faces = all(
            r.details.get("faces_found", 0) == 0 or r.details.get("method") == "no_faces_skipped"
            for r in results if r.error is None
        )

        if all_no_faces:
            return LayerResult(
                layer_name=self.name,
                order=self.order,
                aggregated_score=0.0,
                detector_results=results,
                flags=["no_faces_detected"],
            )

        agg_score = self._aggregate_scores(results)

        flags = []
        if any(r.error and "API" in str(r.error) for r in results):
            flags.append("api_fallback_used")

        return LayerResult(
            layer_name=self.name,
            order=self.order,
            aggregated_score=agg_score,
            detector_results=results,
            flags=flags,
        )
