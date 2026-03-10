"""Invisible watermark detection using the invisible-watermark library.

Attempts to decode DWT-DCT-SVD watermarks commonly embedded by
Stable Diffusion and its derivatives.
"""

import logging
from typing import Optional

import cv2
import numpy as np

from detection.base.detector import BaseDetector
from detection.base.result import DetectorResult

logger = logging.getLogger(__name__)


class InvisibleWatermarkDetector(BaseDetector):
    name = "invisible_watermark"

    def analyze(self, image: np.ndarray, metadata: Optional[dict] = None) -> DetectorResult:
        wm_length = self.config.get("watermark_length", 136)
        method = self.config.get("method", "dwtDctSvd")

        try:
            from imwatermark import WatermarkDecoder

            # Convert RGB to BGR for OpenCV-based library
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            decoder = WatermarkDecoder("bits", wm_length)
            watermark = decoder.decode(bgr, method)

            if watermark is not None:
                # Check if decoded watermark has structure (not random noise)
                wm_array = np.array(watermark)
                ones_ratio = np.mean(wm_array) if len(wm_array) > 0 else 0.5

                # A structured watermark will have non-random bit patterns
                # Random decoding on non-watermarked images gives ~50% ones
                # Structured watermarks tend to have specific patterns
                deviation_from_random = abs(ones_ratio - 0.5)

                # Check for known Stable Diffusion watermark pattern
                has_structure = deviation_from_random > 0.1

                if has_structure:
                    score = 0.8
                    state = "watermark_detected"
                else:
                    score = 0.2
                    state = "possible_watermark"
            else:
                score = 0.0
                state = "no_watermark"

            return DetectorResult(
                detector_name=self.name,
                score=score,
                confidence=0.6 if score > 0.5 else 0.3,
                details={
                    "method": method,
                    "watermark_length": wm_length,
                    "state": state,
                    "ones_ratio": round(float(ones_ratio) if watermark is not None else 0.5, 4),
                },
            )
        except ImportError:
            return DetectorResult(
                detector_name=self.name, score=0.0, confidence=0.0,
                error="invisible-watermark not installed",
            )
        except Exception as e:
            return DetectorResult(
                detector_name=self.name, score=0.0, confidence=0.0, error=str(e),
            )
