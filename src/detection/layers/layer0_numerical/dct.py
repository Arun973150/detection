"""DCT Coefficient Analysis detector.

Examines frequency domain representation for artifacts typical of
AI-generated images (unusual energy in mid-to-high frequency bands)
and double-JPEG compression artifacts.
"""

from typing import Optional

import cv2
import numpy as np
from scipy.fft import dctn

from detection.base.detector import BaseDetector
from detection.base.result import DetectorResult


class DCTDetector(BaseDetector):
    name = "dct"

    def analyze(self, image: np.ndarray, metadata: Optional[dict] = None) -> DetectorResult:
        block_size = self.config.get("block_size", 8)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)

        h, w = gray.shape
        # Trim to multiple of block_size
        h_trim = (h // block_size) * block_size
        w_trim = (w // block_size) * block_size
        gray = gray[:h_trim, :w_trim]

        # Compute block-wise DCT and analyze coefficient distribution
        coeff_hist = np.zeros(64, dtype=np.float64)
        num_blocks = 0

        for i in range(0, h_trim, block_size):
            for j in range(0, w_trim, block_size):
                block = gray[i : i + block_size, j : j + block_size]
                dct_block = dctn(block, type=2, norm="ortho")
                flat = np.abs(dct_block.flatten())
                coeff_hist += flat
                num_blocks += 1

        if num_blocks > 0:
            coeff_hist /= num_blocks

        # Analyze frequency distribution
        # Low freq = first 16, mid = 16-48, high = 48-64
        low_energy = float(np.mean(coeff_hist[:16]))
        mid_energy = float(np.mean(coeff_hist[16:48]))
        high_energy = float(np.mean(coeff_hist[48:]))

        # AI-generated images often have unusual mid/high frequency ratios
        if low_energy > 0:
            mid_ratio = mid_energy / low_energy
            high_ratio = high_energy / low_energy
        else:
            mid_ratio = 0.0
            high_ratio = 0.0

        # Heuristic score: unusual frequency distribution suggests synthetic origin
        # Natural images typically have mid_ratio < 0.3, high_ratio < 0.1
        score = min(1.0, max(0.0, (mid_ratio - 0.2) * 2 + (high_ratio - 0.05) * 5))

        # Check for double-JPEG artifacts (periodic patterns in DCT histogram)
        periodicity = self._detect_periodicity(coeff_hist)

        return DetectorResult(
            detector_name=self.name,
            score=score,
            confidence=0.6,
            details={
                "low_energy": round(low_energy, 4),
                "mid_energy": round(mid_energy, 4),
                "high_energy": round(high_energy, 4),
                "mid_ratio": round(mid_ratio, 4),
                "high_ratio": round(high_ratio, 4),
                "periodicity_score": round(periodicity, 4),
            },
        )

    def _detect_periodicity(self, hist: np.ndarray) -> float:
        """Detect periodic patterns in DCT histogram indicating double JPEG compression."""
        if len(hist) < 4:
            return 0.0
        # Compute autocorrelation of the histogram
        centered = hist - np.mean(hist)
        autocorr = np.correlate(centered, centered, mode="full")
        autocorr = autocorr[len(autocorr) // 2 :]
        if autocorr[0] > 0:
            autocorr = autocorr / autocorr[0]
        # Look for strong peaks at regular intervals (sign of quantization)
        peaks = []
        for i in range(2, min(len(autocorr), 32)):
            if autocorr[i] > 0.3:
                peaks.append(autocorr[i])
        return float(np.mean(peaks)) if peaks else 0.0
