"""Layer 5: Ensemble Fusion — combines all layer outputs into final verdict."""

import logging

import numpy as np

from detection.base.layer import BaseLayer
from detection.base.result import DetectorResult, LayerResult
from detection.utils.heatmap import merge_heatmaps

logger = logging.getLogger(__name__)

# Default layer weights (configurable via YAML)
DEFAULT_WEIGHTS = {
    "layer0_numerical": 0.10,
    "layer0_5_purification": 0.00,  # Flags only
    "layer1_aigen": 0.25,
    "layer2_localization": 0.20,
    "layer2_5_text": 0.10,
    "layer2_6_semantic": 0.10,
    "layer3_deepfake": 0.15,
    "layer4_watermark": 0.10,
}

# Layers with conditional weighting (zeroed when not applicable)
CONDITIONAL_LAYERS = {
    "layer2_5_text": "no_text_regions",      # Zero weight if flagged
    "layer3_deepfake": "no_faces_detected",   # Zero weight if flagged
}


class EnsembleFusionLayer(BaseLayer):
    name = "layer5_fusion"
    order = 5.0

    def setup(self) -> None:
        pass

    def analyze(self, image: np.ndarray, context: dict) -> LayerResult:
        thresholds = self.config.get("thresholds", {})
        adversarial_boost = self.config.get("adversarial_boost", 0.15)

        # Collect all layer results from context
        layer_results: dict[str, LayerResult] = {}
        for key, value in context.items():
            if isinstance(value, LayerResult):
                layer_results[key] = value

        # Determine active weights
        weights = dict(DEFAULT_WEIGHTS)
        inactive_weight = 0.0

        for layer_name, flag in CONDITIONAL_LAYERS.items():
            lr = layer_results.get(layer_name)
            if lr and flag in lr.flags:
                inactive_weight += weights.get(layer_name, 0.0)
                weights[layer_name] = 0.0

        # Redistribute inactive weight proportionally to active layers
        active_total = sum(w for w in weights.values() if w > 0)
        if active_total > 0 and inactive_weight > 0:
            for k in weights:
                if weights[k] > 0:
                    weights[k] += inactive_weight * (weights[k] / active_total)

        # Compute weighted score
        weighted_sum = 0.0
        total_weight = 0.0

        for layer_name, weight in weights.items():
            if weight <= 0:
                continue
            lr = layer_results.get(layer_name)
            if lr is None:
                continue
            weighted_sum += lr.aggregated_score * weight
            total_weight += weight

        overall_score = weighted_sum / total_weight if total_weight > 0 else 0.0

        # Check for adversarial perturbation flags
        all_flags = []
        adversarial_detected = False
        for lr in layer_results.values():
            all_flags.extend(lr.flags)
            if any("adversarial" in f for f in lr.flags):
                adversarial_detected = True

        if adversarial_detected:
            overall_score = min(1.0, overall_score + adversarial_boost)
            all_flags.append("adversarial_perturbation_detected")

        # Apply conflict resolution
        overall_score, verdict_hint = self._resolve_conflicts(
            overall_score, layer_results, all_flags
        )

        # Determine verdict
        verdict = self._determine_verdict(overall_score, layer_results, all_flags, thresholds, verdict_hint)

        # Compute confidence
        confidence = self._compute_confidence(overall_score, layer_results, weights)

        # Merge all heatmaps
        heatmaps = []
        heatmap_weights = []
        for layer_name, lr in layer_results.items():
            if lr.heatmap is not None:
                heatmaps.append(lr.heatmap)
                heatmap_weights.append(weights.get(layer_name, 0.1))

        smoothing = self.config.get("heatmap_smoothing_sigma", 2.0)
        merged_heatmap = None
        if heatmaps:
            merged_heatmap = merge_heatmaps(heatmaps, weights=heatmap_weights)
            if smoothing > 0:
                import cv2
                ksize = int(smoothing * 6) | 1
                merged_heatmap = cv2.GaussianBlur(merged_heatmap, (ksize, ksize), smoothing)

        return LayerResult(
            layer_name=self.name,
            order=self.order,
            aggregated_score=overall_score,
            heatmap=merged_heatmap,
            flags=all_flags,
            metadata={
                "verdict": verdict,
                "confidence": confidence,
                "weights_used": {k: round(v, 4) for k, v in weights.items() if v > 0},
                "adversarial_detected": adversarial_detected,
            },
        )

    def _resolve_conflicts(
        self, score: float, layers: dict[str, LayerResult], flags: list[str]
    ) -> tuple[float, str | None]:
        """Resolve inter-layer conflicts and return adjusted score + verdict hint."""
        hint = None

        l1 = layers.get("layer1_aigen")
        l2 = layers.get("layer2_localization")
        l2_5 = layers.get("layer2_5_text")
        l3 = layers.get("layer3_deepfake")

        # Conflict: AI-gen high but localization low -> clean synthetic image
        if l1 and l2 and l1.aggregated_score > 0.7 and l2.aggregated_score < 0.2:
            hint = "ai_generated"

        # Conflict: localization high but AI-gen low -> real photo was edited
        if l1 and l2 and l1.aggregated_score < 0.3 and l2.aggregated_score > 0.6:
            hint = "manipulated"

        # Override: garbled text found -> strong AI signal even if models miss it
        if l2_5 and "garbled_text_detected" in flags:
            if l1 and l1.aggregated_score > 0.3:
                score = max(score, 0.7)
                hint = "ai_generated"

        # Override: deepfake detected
        if l3 and l3.aggregated_score > 0.7 and "no_faces_detected" not in flags:
            hint = "deepfake"

        return score, hint

    def _determine_verdict(
        self, score: float, layers: dict[str, LayerResult],
        flags: list[str], thresholds: dict, hint: str | None,
    ) -> str:
        """Determine final verdict label."""
        if hint == "deepfake":
            return "deepfake"
        if hint:
            return hint

        t_ai = thresholds.get("ai_generated", 0.75)
        t_manip = thresholds.get("manipulated", 0.65)
        t_deepfake = thresholds.get("deepfake", 0.70)
        t_suspicious = thresholds.get("suspicious", 0.40)

        # Check deepfake first
        l3 = layers.get("layer3_deepfake")
        if l3 and l3.aggregated_score > t_deepfake and "no_faces_detected" not in flags:
            return "deepfake"

        if score > t_ai:
            # Distinguish AI-generated vs manipulated
            l1 = layers.get("layer1_aigen")
            l2 = layers.get("layer2_localization")
            if l2 and l2.aggregated_score > t_manip:
                return "manipulated"
            return "ai_generated"

        if score > t_suspicious:
            return "uncertain"

        return "authentic"

    def _compute_confidence(
        self, score: float, layers: dict[str, LayerResult], weights: dict[str, float]
    ) -> float:
        """Compute confidence based on inter-model agreement."""
        scores = []
        for layer_name, lr in layers.items():
            w = weights.get(layer_name, 0)
            if w > 0:
                scores.append(lr.aggregated_score)

        if not scores:
            return 0.0

        # High agreement (low variance) = high confidence
        variance = float(np.var(scores))
        mean_score = float(np.mean(scores))

        # Confidence is higher when:
        # 1. Models agree (low variance)
        # 2. Score is far from the decision boundary (0.5)
        agreement = max(0.0, 1.0 - variance * 4)  # Scale variance
        decisiveness = abs(mean_score - 0.5) * 2  # 0 at boundary, 1 at extremes

        confidence = 0.6 * agreement + 0.4 * decisiveness
        return round(min(1.0, max(0.0, confidence)), 4)
