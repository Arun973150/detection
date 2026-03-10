"""Layer 2.6: Semantic Consistency Checks (NEW).

Physics-based analysis: lighting direction, shadow consistency, reflection
plausibility, and perspective vanishing point consistency. All classical CV.
"""

import numpy as np

from detection.base.layer import BaseLayer
from detection.base.result import DetectorResult, LayerResult
from detection.layers.layer2_6_semantic.lighting import analyze_lighting_consistency
from detection.layers.layer2_6_semantic.reflections import analyze_reflections
from detection.layers.layer2_6_semantic.shadows import analyze_shadow_consistency
from detection.layers.layer2_6_semantic.vanishing import analyze_vanishing_points


class SemanticConsistencyLayer(BaseLayer):
    name = "layer2_6_semantic"
    order = 2.6

    def setup(self) -> None:
        pass  # No models to load — pure classical CV

    def analyze(self, image: np.ndarray, context: dict) -> LayerResult:
        # Lighting direction consistency
        lighting_cfg = self.config.get("lighting", {})
        lighting_result = analyze_lighting_consistency(
            image,
            num_regions=lighting_cfg.get("num_regions", 4),
            consistency_threshold=lighting_cfg.get("consistency_threshold", 30),
        )

        # Shadow direction consistency
        shadow_cfg = self.config.get("shadows", {})
        shadow_result = analyze_shadow_consistency(
            image,
            min_shadow_area=shadow_cfg.get("min_shadow_area", 500),
            direction_variance_threshold=shadow_cfg.get("direction_variance_threshold", 25),
        )

        # Reflection plausibility
        reflection_result = analyze_reflections(image)

        # Vanishing point consistency
        vp_cfg = self.config.get("vanishing", {})
        vp_result = analyze_vanishing_points(
            image,
            hough_threshold=vp_cfg.get("hough_threshold", 100),
            max_vanishing_points=vp_cfg.get("max_vanishing_points", 3),
            consistency_threshold=vp_cfg.get("consistency_threshold", 50),
        )

        # Build detector results
        detector_results = [
            DetectorResult(
                detector_name="lighting_consistency",
                score=lighting_result["score"],
                confidence=0.5,
                details=lighting_result,
            ),
            DetectorResult(
                detector_name="shadow_consistency",
                score=shadow_result["score"],
                confidence=0.5,
                details=shadow_result,
            ),
            DetectorResult(
                detector_name="reflection_plausibility",
                score=reflection_result["score"],
                confidence=0.3,  # Lower confidence — heuristic
                details=reflection_result,
            ),
            DetectorResult(
                detector_name="vanishing_point_consistency",
                score=vp_result["score"],
                confidence=0.5,
                details=vp_result,
            ),
        ]

        # Weighted combination
        weights = {
            "lighting_consistency": 0.30,
            "shadow_consistency": 0.30,
            "reflection_plausibility": 0.10,
            "vanishing_point_consistency": 0.30,
        }
        agg_score = self._aggregate_scores(detector_results, weights)

        flags = []
        if not lighting_result.get("consistent", True):
            flags.append("inconsistent_lighting")
        if not shadow_result.get("consistent", True):
            flags.append("inconsistent_shadows")
        if not vp_result.get("consistent", True):
            flags.append("inconsistent_perspective")

        return LayerResult(
            layer_name=self.name,
            order=self.order,
            aggregated_score=agg_score,
            detector_results=detector_results,
            flags=flags,
        )
