"""Spatial Rich Model (SRM) filtering detector.

Applies high-pass filter kernels from steganalysis to suppress image content
and amplify residual noise signals. Synthetic images have different residual
statistics than camera-captured images.
"""

from typing import Optional

import cv2
import numpy as np

from detection.base.detector import BaseDetector
from detection.base.result import DetectorResult

# Standard SRM filter kernels (subset of the 30 rich model filters)
SRM_KERNELS = [
    # 1st order edge
    np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]], dtype=np.float32),
    np.array([[0, 0, 0], [0, -1, 0], [0, 1, 0]], dtype=np.float32),
    # 2nd order edge
    np.array([[0, 0, 0], [1, -2, 1], [0, 0, 0]], dtype=np.float32),
    np.array([[0, 1, 0], [0, -2, 0], [0, 1, 0]], dtype=np.float32),
    # 3rd order
    np.array([[0, 0, 0], [-1, 3, -3], [0, 0, 1]], dtype=np.float32),
    # Square 3x3
    np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]], dtype=np.float32),
    # Square 5x5 (center portion)
    np.array([[0, 0, -1, 0, 0],
              [0, 0, 2, 0, 0],
              [-1, 2, -4, 2, -1],
              [0, 0, 2, 0, 0],
              [0, 0, -1, 0, 0]], dtype=np.float32),
    # Diagonal
    np.array([[0, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float32),
    np.array([[0, 0, 0], [0, -1, 0], [1, 0, 0]], dtype=np.float32),
    # SPAM-like
    np.array([[0, 1, 0], [0, -2, 0], [0, 1, 0]], dtype=np.float32) / 2,
    np.array([[0, 0, 0], [1, -2, 1], [0, 0, 0]], dtype=np.float32) / 2,
    # KV kernel
    np.array([[-1, 2, -2, 2, -1],
              [2, -6, 8, -6, 2],
              [-2, 8, -12, 8, -2],
              [2, -6, 8, -6, 2],
              [-1, 2, -2, 2, -1]], dtype=np.float32) / 12,
]


class SRMDetector(BaseDetector):
    name = "srm"

    def analyze(self, image: np.ndarray, metadata: Optional[dict] = None) -> DetectorResult:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)

        residual_stats = []
        residual_maps = []

        for kernel in SRM_KERNELS:
            filtered = cv2.filter2D(gray, -1, kernel)
            residual_maps.append(np.abs(filtered))

            # Compute statistics of the residual
            residual_stats.append({
                "mean": float(np.mean(np.abs(filtered))),
                "std": float(np.std(filtered)),
                "kurtosis": float(self._kurtosis(filtered)),
            })

        # Combine residual maps
        combined_residual = np.mean(residual_maps, axis=0)

        # Natural images have higher residual variance than synthetic
        # AI images tend to have smoother, more uniform residuals
        mean_residual = float(np.mean(combined_residual))
        std_residual = float(np.std(combined_residual))

        # Low residual variance suggests synthetic origin
        # This is a simplified heuristic — proper SRM uses trained classifiers
        if std_residual > 0:
            uniformity = mean_residual / std_residual
        else:
            uniformity = 0.0

        # Higher uniformity (low variance relative to mean) suggests synthetic
        score = min(1.0, max(0.0, (uniformity - 1.0) / 3.0))

        # Create heatmap from residual variance
        h, w = gray.shape
        patch_size = 32
        heatmap = np.zeros((h, w), dtype=np.float32)
        for i in range(0, h - patch_size, patch_size // 2):
            for j in range(0, w - patch_size, patch_size // 2):
                patch = combined_residual[i : i + patch_size, j : j + patch_size]
                heatmap[i : i + patch_size, j : j + patch_size] = np.std(patch)

        max_val = heatmap.max()
        if max_val > 0:
            heatmap = 1.0 - (heatmap / max_val)  # Invert: low variance = suspicious

        return DetectorResult(
            detector_name=self.name,
            score=score,
            confidence=0.5,
            heatmap=heatmap,
            details={
                "mean_residual": round(mean_residual, 4),
                "std_residual": round(std_residual, 4),
                "uniformity": round(uniformity, 4),
                "num_kernels": len(SRM_KERNELS),
            },
        )

    @staticmethod
    def _kurtosis(data: np.ndarray) -> float:
        """Compute excess kurtosis."""
        mean = np.mean(data)
        std = np.std(data)
        if std < 1e-8:
            return 0.0
        return float(np.mean(((data - mean) / std) ** 4) - 3.0)
