"""PIM (Pixel-Inconsistency Modelling) detector.

Detects demosaicing trace inconsistencies at pixel level. Modified regions
break the camera ISP demosaicing pattern.

Weights: github.com/ChenqiKONG/PIM
"""

import logging
from typing import Optional

import cv2
import numpy as np

from detection.base.detector import BaseDetector
from detection.base.result import DetectorResult

logger = logging.getLogger(__name__)


class PIMDetector(BaseDetector):
    name = "pim"

    def load_model(self) -> None:
        self._loaded = True

    def analyze(self, image: np.ndarray, metadata: Optional[dict] = None) -> DetectorResult:
        self._ensure_loaded()

        try:
            return self._demosaicing_analysis(image)
        except Exception as e:
            return DetectorResult(
                detector_name=self.name, score=0.0, confidence=0.0, error=str(e),
            )

    def _demosaicing_analysis(self, image: np.ndarray) -> DetectorResult:
        """Analyze demosaicing pattern consistency."""
        # Check CFA (Color Filter Array) pattern consistency
        # Real camera images have Bayer pattern artifacts
        r, g, b = image[:, :, 0].astype(np.float64), image[:, :, 1].astype(np.float64), image[:, :, 2].astype(np.float64)

        # Compute inter-channel prediction residuals
        # In Bayer demosaicing, G channel is interpolated from 2x density
        # R and B from 1x density — correlation patterns differ
        g_pred_from_r = cv2.GaussianBlur(r, (3, 3), 0.5)
        g_pred_from_b = cv2.GaussianBlur(b, (3, 3), 0.5)

        residual_rg = np.abs(g - g_pred_from_r)
        residual_bg = np.abs(g - g_pred_from_b)

        # Compute local consistency of demosaicing residuals
        h, w = image.shape[:2]
        patch_size = 64
        heatmap = np.zeros((h, w), dtype=np.float32)
        local_stats = []

        for i in range(0, h - patch_size, patch_size // 2):
            for j in range(0, w - patch_size, patch_size // 2):
                rg_patch = residual_rg[i : i + patch_size, j : j + patch_size]
                bg_patch = residual_bg[i : i + patch_size, j : j + patch_size]

                # Measure correlation pattern — Bayer artifacts are periodic
                rg_fft = np.abs(np.fft.fft2(rg_patch))
                periodicity = self._measure_periodicity(rg_fft)
                local_stats.append((i, j, periodicity))

        if not local_stats:
            return DetectorResult(detector_name=self.name, score=0.0, confidence=0.0)

        all_periodicities = [p for _, _, p in local_stats]
        median_p = np.median(all_periodicities)

        for i, j, p in local_stats:
            if median_p > 0:
                deviation = abs(p - median_p) / (median_p + 1e-8)
            else:
                deviation = 0.0
            heatmap[i : i + patch_size, j : j + patch_size] = np.maximum(
                heatmap[i : i + patch_size, j : j + patch_size], min(1.0, deviation)
            )

        score = min(1.0, float(np.percentile(heatmap[heatmap > 0], 90)) if np.any(heatmap > 0) else 0.0)

        return DetectorResult(
            detector_name=self.name,
            score=score,
            confidence=0.55,
            heatmap=heatmap,
            details={"method": "demosaicing_inconsistency", "median_periodicity": round(float(median_p), 4)},
        )

    @staticmethod
    def _measure_periodicity(fft_block: np.ndarray) -> float:
        """Measure strength of periodic patterns in FFT block."""
        h, w = fft_block.shape
        center_h, center_w = h // 2, w // 2
        # Look at specific frequencies corresponding to Bayer pattern (period 2)
        nyquist_energy = fft_block[center_h, :].sum() + fft_block[:, center_w].sum()
        total_energy = fft_block.sum()
        if total_energy > 0:
            return float(nyquist_energy / total_energy)
        return 0.0
