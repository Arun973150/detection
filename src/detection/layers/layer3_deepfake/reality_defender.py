"""Reality Defender API client for deepfake detection."""

import logging
from typing import Optional

import numpy as np

from detection.base.detector import BaseDetector
from detection.base.result import DetectorResult

logger = logging.getLogger(__name__)


class RealityDefenderDetector(BaseDetector):
    name = "reality_defender"

    def load_model(self) -> None:
        self._api_key = self.config.get("api_key", "")
        self._endpoint = self.config.get("endpoint", "https://api.realitydefender.com/v1/detect")
        self._timeout = self.config.get("timeout_s", 10)
        self._loaded = bool(self._api_key and not self._api_key.startswith("${"))

    def analyze(self, image: np.ndarray, metadata: Optional[dict] = None) -> DetectorResult:
        self._ensure_loaded()

        if not self._loaded:
            return DetectorResult(
                detector_name=self.name, score=0.0, confidence=0.0,
                error="API key not configured",
            )

        try:
            import httpx
            from io import BytesIO
            from PIL import Image

            # Encode image
            pil_img = Image.fromarray(image)
            buf = BytesIO()
            pil_img.save(buf, format="JPEG", quality=90)
            buf.seek(0)

            # Send to API
            response = httpx.post(
                self._endpoint,
                headers={"Authorization": f"Bearer {self._api_key}"},
                files={"file": ("image.jpg", buf, "image/jpeg")},
                timeout=self._timeout,
            )
            response.raise_for_status()
            data = response.json()

            score = float(data.get("probability", data.get("score", 0.0)))
            return DetectorResult(
                detector_name=self.name,
                score=score,
                confidence=0.8,
                details={
                    "method": "reality_defender_api",
                    "response": {k: v for k, v in data.items() if k != "raw"},
                },
            )
        except Exception as e:
            logger.warning(f"Reality Defender API call failed: {e}")
            return DetectorResult(
                detector_name=self.name, score=0.0, confidence=0.0,
                error=f"API call failed: {e}",
            )
