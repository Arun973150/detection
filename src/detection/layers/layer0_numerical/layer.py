"""Layer 0: Numerical Pre-Analysis — pure signal processing, no ML models."""

import numpy as np

from detection.base.layer import BaseLayer
from detection.base.result import LayerResult
from detection.layers.layer0_numerical.chromatic import ChromaticAberrationDetector
from detection.layers.layer0_numerical.dct import DCTDetector
from detection.layers.layer0_numerical.ela import ELADetector
from detection.layers.layer0_numerical.laplacian import LaplacianDetector
from detection.layers.layer0_numerical.metadata import MetadataDetector
from detection.layers.layer0_numerical.prnu import PRNUDetector
from detection.layers.layer0_numerical.srm import SRMDetector
from detection.utils.heatmap import merge_heatmaps


class NumericalPreAnalysisLayer(BaseLayer):
    name = "layer0_numerical"
    order = 0.0

    def setup(self) -> None:
        self.detectors = [
            ELADetector(self.config.get("ela", {}), self.device),
            DCTDetector(self.config.get("dct", {}), self.device),
            SRMDetector(self.config.get("srm", {}), self.device),
            PRNUDetector(self.config.get("prnu", {}), self.device),
            LaplacianDetector(self.config.get("laplacian", {}), self.device),
            ChromaticAberrationDetector(self.config.get("chromatic", {}), self.device),
            MetadataDetector(self.config.get("metadata", {}), self.device),
        ]

    def analyze(self, image: np.ndarray, context: dict) -> LayerResult:
        results = []
        for detector in self.detectors:
            result = self._run_detector_safe(detector, image, context)
            results.append(result)

        # Merge heatmaps from detectors that produced them
        heatmaps = [r.heatmap for r in results if r.heatmap is not None]
        merged = merge_heatmaps(heatmaps, method="max") if heatmaps else None

        # Aggregate score
        agg_score = self._aggregate_scores(results)

        # Collect flags
        flags = []
        for r in results:
            flags.extend(r.details.get("flags", []))

        return LayerResult(
            layer_name=self.name,
            order=self.order,
            aggregated_score=agg_score,
            detector_results=results,
            heatmap=merged,
            flags=flags,
        )
