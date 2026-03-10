"""Pipeline orchestrator — runs layers sequentially, detectors in parallel within layers."""

import logging

import numpy as np

from detection.base.layer import BaseLayer
from detection.base.result import PipelineResult
from detection.utils.timing import Timer

logger = logging.getLogger(__name__)


class Pipeline:
    """Main detection pipeline that chains layers and produces a final verdict."""

    def __init__(self, config: dict):
        self.config = config
        self.layers: list[BaseLayer] = []

    def register_layer(self, layer: BaseLayer) -> None:
        """Register a layer and maintain order."""
        self.layers.append(layer)
        self.layers.sort(key=lambda l: l.order)

    def setup(self) -> None:
        """Initialize all registered layers."""
        for layer in self.layers:
            if layer.enabled:
                logger.info(f"Setting up layer: {layer.name} (order={layer.order})")
                layer.setup()

    def run(self, image: np.ndarray) -> PipelineResult:
        """Run the full detection pipeline on an image.

        Args:
            image: RGB uint8 numpy array (H, W, 3)

        Returns:
            PipelineResult with verdict, confidence, heatmap, and breakdown.
        """
        with Timer("pipeline.run") as timer:
            context: dict = {}
            layer_results = []

            for layer in self.layers:
                if not layer.enabled:
                    continue

                logger.info(f"Running layer: {layer.name}")
                try:
                    result = layer.analyze(image, context)
                    layer_results.append(result)
                    context[layer.name] = result

                    # Layer 0.5 stores purified images in context
                    if hasattr(result, "metadata") and "purified_images" in result.metadata:
                        context["purified_images"] = result.metadata["purified_images"]

                except Exception as e:
                    logger.error(f"Layer {layer.name} failed completely: {e}", exc_info=True)

            # The last layer (fusion) should produce the final verdict
            # If fusion layer ran, its result contains the verdict
            fusion_result = context.get("layer5_fusion")

        if fusion_result:
            return PipelineResult(
                verdict=fusion_result.metadata.get("verdict", "uncertain"),
                confidence=fusion_result.metadata.get("confidence", 0.0),
                overall_score=fusion_result.aggregated_score,
                layer_results=layer_results,
                merged_heatmap=fusion_result.heatmap,
                flags=self._collect_flags(layer_results),
                processing_time_ms=timer.elapsed_ms,
            )

        # Fallback if fusion layer didn't run
        return self._fallback_result(layer_results, timer.elapsed_ms)

    def _collect_flags(self, layer_results) -> list[str]:
        """Collect all flags from all layers."""
        flags = []
        for lr in layer_results:
            flags.extend(lr.flags)
        return flags

    def _fallback_result(self, layer_results, elapsed_ms: float) -> PipelineResult:
        """Generate a basic result without the fusion layer."""
        scores = [lr.aggregated_score for lr in layer_results if lr.aggregated_score > 0]
        avg_score = sum(scores) / len(scores) if scores else 0.0

        if avg_score > 0.75:
            verdict = "ai_generated"
        elif avg_score > 0.40:
            verdict = "uncertain"
        else:
            verdict = "authentic"

        return PipelineResult(
            verdict=verdict,
            confidence=1.0 - abs(avg_score - 0.5) * 2,
            overall_score=avg_score,
            layer_results=layer_results,
            flags=self._collect_flags(layer_results),
            processing_time_ms=elapsed_ms,
        )
