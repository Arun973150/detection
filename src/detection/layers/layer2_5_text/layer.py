"""Layer 2.5: Text/Typography Analysis (NEW).

Uses OCR to extract text regions and analyzes them for signs of AI generation:
garbled characters, inconsistent fonts, broken perspective alignment.
"""

import logging

import numpy as np

from detection.base.layer import BaseLayer
from detection.base.result import DetectorResult, LayerResult
from detection.layers.layer2_5_text.font_analysis import analyze_font_consistency
from detection.layers.layer2_5_text.garble_detect import analyze_garble
from detection.layers.layer2_5_text.ocr_engine import OCREngine
from detection.layers.layer2_5_text.perspective import analyze_text_perspective

logger = logging.getLogger(__name__)


class TextTypographyLayer(BaseLayer):
    name = "layer2_5_text"
    order = 2.5

    def setup(self) -> None:
        engine = self.config.get("ocr_engine", "easyocr")
        languages = self.config.get("languages", ["en"])
        self.ocr = OCREngine(engine=engine, languages=languages)

    def analyze(self, image: np.ndarray, context: dict) -> LayerResult:
        min_regions = self.config.get("min_text_regions", 1)

        # Extract text regions
        try:
            regions = self.ocr.extract_text(image)
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return LayerResult(
                layer_name=self.name, order=self.order,
                aggregated_score=0.0, flags=["ocr_failed"],
            )

        # If no text found, this layer contributes nothing
        if len(regions) < min_regions:
            return LayerResult(
                layer_name=self.name, order=self.order,
                aggregated_score=0.0, flags=["no_text_regions"],
                metadata={"text_regions_found": len(regions)},
            )

        # Run sub-analyses
        conf_threshold = self.config.get("garble_confidence_threshold", 0.3)
        garble_result = analyze_garble(regions, confidence_threshold=conf_threshold)

        stroke_threshold = self.config.get("font_stroke_variance_threshold", 0.4)
        font_result = analyze_font_consistency(image, regions, stroke_variance_threshold=stroke_threshold)

        angle_threshold = self.config.get("perspective_angle_threshold", 15.0)
        perspective_result = analyze_text_perspective(regions, angle_threshold=angle_threshold)

        # Build detector results
        detector_results = [
            DetectorResult(
                detector_name="garble_detect",
                score=garble_result["score"],
                confidence=0.7,
                details=garble_result,
            ),
            DetectorResult(
                detector_name="font_analysis",
                score=font_result["score"],
                confidence=0.5,
                details=font_result,
            ),
            DetectorResult(
                detector_name="text_perspective",
                score=perspective_result["score"],
                confidence=0.5,
                details=perspective_result,
            ),
        ]

        # Weighted combination
        weights = {"garble_detect": 0.5, "font_analysis": 0.25, "text_perspective": 0.25}
        agg_score = self._aggregate_scores(detector_results, weights)

        flags = []
        if garble_result["score"] > 0.5:
            flags.append("garbled_text_detected")
        if font_result["score"] > 0.5:
            flags.append("inconsistent_fonts")
        if perspective_result["score"] > 0.5:
            flags.append("perspective_mismatch")

        return LayerResult(
            layer_name=self.name,
            order=self.order,
            aggregated_score=agg_score,
            detector_results=detector_results,
            flags=flags,
            metadata={
                "text_regions_found": len(regions),
                "texts": [r.text for r in regions[:10]],  # First 10 texts
            },
        )
