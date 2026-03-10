"""Error Level Analysis (ELA) detector.

Re-saves image at controlled JPEG quality and computes pixel-wise difference.
Edited regions re-compress differently, producing visible inconsistencies.
"""

from io import BytesIO
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from detection.base.detector import BaseDetector
from detection.base.result import DetectorResult


class ELADetector(BaseDetector):
    name = "ela"

    def analyze(self, image: np.ndarray, metadata: Optional[dict] = None) -> DetectorResult:
        quality_levels = self.config.get("quality_levels", [90, 75, 50])
        threshold = self.config.get("threshold", 0.15)

        ela_maps = []
        for quality in quality_levels:
            ela_map = self._compute_ela(image, quality)
            ela_maps.append(ela_map)

        # Average ELA across quality levels
        combined = np.mean(ela_maps, axis=0)

        # Normalize to [0, 1]
        max_val = combined.max()
        if max_val > 0:
            heatmap = combined / max_val
        else:
            heatmap = combined

        mean_ela = float(np.mean(heatmap))
        score = min(1.0, mean_ela / threshold) if threshold > 0 else 0.0

        return DetectorResult(
            detector_name=self.name,
            score=score,
            confidence=0.7,
            heatmap=heatmap.astype(np.float32),
            details={
                "mean_ela": round(mean_ela, 4),
                "max_ela": round(float(heatmap.max()), 4),
                "quality_levels": quality_levels,
            },
        )

    def _compute_ela(self, image: np.ndarray, quality: int) -> np.ndarray:
        """Compute ELA difference map at a given JPEG quality."""
        pil_img = Image.fromarray(image)
        buffer = BytesIO()
        pil_img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        recompressed = np.array(Image.open(buffer).convert("RGB"))

        diff = np.abs(image.astype(np.float32) - recompressed.astype(np.float32))
        # Convert to single-channel magnitude
        return np.mean(diff, axis=2)
