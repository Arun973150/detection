"""Layer 2: Manipulation Localization — pixel-level heatmap models."""

import numpy as np

from detection.base.layer import BaseLayer
from detection.base.result import LayerResult
from detection.layers.layer2_localization.forensim import ForensimDetector
from detection.layers.layer2_localization.hifi_ifdl import HiFiIFDLDetector
from detection.layers.layer2_localization.pim import PIMDetector
from detection.layers.layer2_localization.trufor import TruForDetector
from detection.utils.heatmap import merge_heatmaps


class ManipulationLocalizationLayer(BaseLayer):
    name = "layer2_localization"
    order = 2.0

    def setup(self) -> None:
        detector_configs = [
            ("trufor", TruForDetector),
            ("pim", PIMDetector),
            ("hifi_ifdl", HiFiIFDLDetector),
            ("forensim", ForensimDetector),
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

        # Merge heatmaps
        heatmaps = [r.heatmap for r in results if r.heatmap is not None]
        resolution = tuple(self.config.get("heatmap_resolution", [512, 512]))

        weights_map = {}
        for det in self.detectors:
            det_cfg = self.config.get(det.name, {})
            weights_map[det.name] = det_cfg.get("weight", 1.0)

        heatmap_weights = [
            weights_map.get(r.detector_name, 1.0) for r in results if r.heatmap is not None
        ]
        merged = merge_heatmaps(heatmaps, target_size=resolution, weights=heatmap_weights) if heatmaps else None

        agg_score = self._aggregate_scores(results, weights_map)

        flags = []
        for r in results:
            if r.details.get("adversarial_flag"):
                flags.append(f"adversarial_detected_{r.detector_name}")

        return LayerResult(
            layer_name=self.name,
            order=self.order,
            aggregated_score=agg_score,
            detector_results=results,
            heatmap=merged,
            flags=flags,
        )
