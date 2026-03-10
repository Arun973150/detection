"""Photo Response Non-Uniformity (PRNU) analysis.

Every camera sensor has unique fixed-pattern noise. Real photos contain this
fingerprint while synthetic images do not. Checks for presence and spatial
consistency of sensor noise patterns.
"""

from typing import Optional

import cv2
import numpy as np

from detection.base.detector import BaseDetector
from detection.base.result import DetectorResult

try:
    import pywt

    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False


class PRNUDetector(BaseDetector):
    name = "prnu"

    def analyze(self, image: np.ndarray, metadata: Optional[dict] = None) -> DetectorResult:
        if not HAS_PYWT:
            return DetectorResult(
                detector_name=self.name, score=0.0, confidence=0.0,
                error="PyWavelets not installed",
            )

        wavelet = self.config.get("wavelet", "db4")
        levels = self.config.get("levels", 3)

        # Extract noise residual using wavelet denoising
        noise_residual = self._extract_noise(image, wavelet, levels)

        # Analyze spatial consistency of noise
        consistency = self._check_consistency(noise_residual)

        # Compute noise strength — real images have stronger, more structured PRNU
        noise_strength = float(np.std(noise_residual))

        # Very low noise strength suggests synthetic (no sensor)
        # Very high suggests heavy processing or manipulation
        if noise_strength < 0.5:
            score = 0.7  # Likely synthetic
        elif noise_strength > 5.0:
            score = 0.4  # Possibly manipulated
        else:
            score = max(0.0, 0.3 - consistency * 0.3)  # Consistent = authentic

        # Create heatmap showing noise inconsistencies
        heatmap = self._noise_inconsistency_map(noise_residual)

        return DetectorResult(
            detector_name=self.name,
            score=score,
            confidence=0.55,
            heatmap=heatmap,
            details={
                "noise_strength": round(noise_strength, 4),
                "consistency": round(consistency, 4),
                "wavelet": wavelet,
            },
        )

    def _extract_noise(self, image: np.ndarray, wavelet: str, levels: int) -> np.ndarray:
        """Extract noise residual via wavelet denoising."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float64)

        # Wavelet decomposition
        coeffs = pywt.wavedec2(gray, wavelet, level=levels)

        # Zero out approximation coefficients (keep only detail/noise)
        coeffs[0] = np.zeros_like(coeffs[0])

        # Reconstruct from detail coefficients only
        noise = pywt.waverec2(coeffs, wavelet)

        # Trim to original size (wavelet reconstruction may pad)
        h, w = gray.shape
        return noise[:h, :w]

    def _check_consistency(self, noise: np.ndarray) -> float:
        """Check spatial consistency of noise pattern (0 = inconsistent, 1 = consistent)."""
        h, w = noise.shape
        patch_size = min(64, h // 4, w // 4)
        if patch_size < 8:
            return 0.5

        stds = []
        for i in range(0, h - patch_size, patch_size):
            for j in range(0, w - patch_size, patch_size):
                patch = noise[i : i + patch_size, j : j + patch_size]
                stds.append(np.std(patch))

        if not stds:
            return 0.5

        # Low coefficient of variation = consistent noise = likely authentic
        mean_std = np.mean(stds)
        if mean_std < 1e-8:
            return 0.0
        cv = np.std(stds) / mean_std
        return float(max(0.0, 1.0 - cv))

    def _noise_inconsistency_map(self, noise: np.ndarray) -> np.ndarray:
        """Create a heatmap of noise inconsistencies."""
        h, w = noise.shape
        patch_size = 32
        heatmap = np.zeros((h, w), dtype=np.float32)

        stds = []
        positions = []
        for i in range(0, h - patch_size, patch_size // 2):
            for j in range(0, w - patch_size, patch_size // 2):
                patch = noise[i : i + patch_size, j : j + patch_size]
                stds.append(np.std(patch))
                positions.append((i, j))

        if not stds:
            return heatmap

        median_std = np.median(stds)
        for (i, j), s in zip(positions, stds):
            # Deviation from median noise level = suspicious
            if median_std > 0:
                deviation = abs(s - median_std) / median_std
            else:
                deviation = 0
            heatmap[i : i + patch_size, j : j + patch_size] = min(1.0, deviation)

        return heatmap
