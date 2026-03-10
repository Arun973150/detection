"""Laplacian Variance Mapping for local sharpness inconsistency detection.

Computes per-patch Laplacian variance to detect inconsistent sharpness levels,
which are common in compositing, inpainting, and face swapping.
"""

from typing import Optional

import cv2
import numpy as np

from detection.base.detector import BaseDetector
from detection.base.result import DetectorResult


class LaplacianDetector(BaseDetector):
    name = "laplacian"

    def analyze(self, image: np.ndarray, metadata: Optional[dict] = None) -> DetectorResult:
        patch_size = self.config.get("patch_size", 64)
        stride = self.config.get("stride", 32)

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape

        # Compute Laplacian variance per patch
        variances = []
        positions = []

        for i in range(0, h - patch_size, stride):
            for j in range(0, w - patch_size, stride):
                patch = gray[i : i + patch_size, j : j + patch_size]
                lap = cv2.Laplacian(patch, cv2.CV_64F)
                var = float(lap.var())
                variances.append(var)
                positions.append((i, j))

        if not variances:
            return DetectorResult(
                detector_name=self.name, score=0.0, confidence=0.0,
                error="Image too small for analysis",
            )

        variances = np.array(variances)
        median_var = np.median(variances)

        # Build inconsistency heatmap
        heatmap = np.zeros((h, w), dtype=np.float32)
        deviations = []

        for (i, j), var in zip(positions, variances):
            if median_var > 0:
                deviation = abs(var - median_var) / median_var
            else:
                deviation = 0.0
            deviations.append(deviation)
            heatmap[i : i + patch_size, j : j + patch_size] = np.maximum(
                heatmap[i : i + patch_size, j : j + patch_size], min(1.0, deviation)
            )

        # Score: high variance in sharpness = likely manipulated
        coefficient_of_variation = float(np.std(variances) / (median_var + 1e-8))
        score = min(1.0, max(0.0, (coefficient_of_variation - 0.5) / 1.5))

        return DetectorResult(
            detector_name=self.name,
            score=score,
            confidence=0.6,
            heatmap=heatmap,
            details={
                "median_variance": round(float(median_var), 2),
                "cv": round(coefficient_of_variation, 4),
                "num_patches": len(variances),
                "max_deviation": round(float(max(deviations)), 4),
            },
        )
