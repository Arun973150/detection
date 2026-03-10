"""SynthID-style g-function watermark scoring.

Samples DCT frequency bins at pseudo-random positions and tests whether
their signs follow an expected pattern. Score > 0.5 suggests watermarking.
"""

from typing import Optional

import numpy as np
from scipy.fft import dctn

from detection.base.detector import BaseDetector
from detection.base.result import DetectorResult


class GFunctionDetector(BaseDetector):
    name = "gfunction"

    def analyze(self, image: np.ndarray, metadata: Optional[dict] = None) -> DetectorResult:
        num_bins = self.config.get("num_frequency_bins", 256)
        significance = self.config.get("significance_threshold", 0.6)

        try:
            score, details = self._compute_gfunction(image, num_bins)

            # Determine state
            if score > significance:
                state = "watermarked"
            elif score < 1 - significance:
                state = "not_watermarked"
            else:
                state = "uncertain"

            return DetectorResult(
                detector_name=self.name,
                score=max(0.0, (score - 0.5) * 2) if score > 0.5 else 0.0,
                confidence=abs(score - 0.5) * 2,
                details={
                    "method": "gfunction_scoring",
                    "raw_score": round(score, 4),
                    "state": state,
                    **details,
                },
            )
        except Exception as e:
            return DetectorResult(
                detector_name=self.name, score=0.0, confidence=0.0, error=str(e),
            )

    def _compute_gfunction(self, image: np.ndarray, num_bins: int) -> tuple[float, dict]:
        """Compute g-function score on DCT coefficients."""
        import cv2

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float64)

        # Full-image DCT
        dct = dctn(gray, type=2, norm="ortho")

        # Sample frequency bins at pseudo-random positions
        rng = np.random.RandomState(42)  # Fixed seed (in practice, this is the secret key)
        h, w = dct.shape

        # Generate sampling positions (avoiding DC component)
        positions = []
        for _ in range(num_bins):
            y = rng.randint(1, min(h, 256))
            x = rng.randint(1, min(w, 256))
            positions.append((y, x))

        # Sample coefficient signs
        signs = []
        for y, x in positions:
            if y < h and x < w:
                signs.append(1 if dct[y, x] > 0 else 0)

        if not signs:
            return 0.5, {"sampled_bins": 0}

        # Generate expected pattern (pseudo-random binary sequence)
        expected = rng.randint(0, 2, size=len(signs))

        # Compute agreement rate
        agreement = np.mean(np.array(signs) == expected)

        return float(agreement), {
            "sampled_bins": len(signs),
            "agreement_rate": round(float(agreement), 4),
        }
