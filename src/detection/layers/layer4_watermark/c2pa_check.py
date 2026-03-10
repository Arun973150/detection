"""C2PA Content Credentials watermark detection.

Checks for cryptographic provenance metadata from Adobe, OpenAI, Microsoft.
"""

import logging
from typing import Optional

import numpy as np

from detection.base.detector import BaseDetector
from detection.base.result import DetectorResult

logger = logging.getLogger(__name__)


class C2PADetector(BaseDetector):
    name = "c2pa"

    def analyze(self, image: np.ndarray, metadata: Optional[dict] = None) -> DetectorResult:
        file_path = (metadata or {}).get("file_path")

        if not file_path:
            return DetectorResult(
                detector_name=self.name, score=0.0, confidence=0.0,
                details={"method": "c2pa", "error": "No file path available"},
            )

        try:
            from c2pa import Reader

            reader = Reader.from_file(file_path)
            manifest = reader.get_active_manifest()

            if manifest:
                assertions = manifest.get("assertions", [])
                ai_generated = any(
                    "ai_generated" in str(a).lower() or "ai_generative" in str(a).lower()
                    for a in assertions
                )
                tool = manifest.get("claim_generator", "unknown")

                if ai_generated:
                    score = 0.95
                else:
                    score = 0.1  # Has C2PA but not marked AI

                return DetectorResult(
                    detector_name=self.name,
                    score=score,
                    confidence=0.9,
                    details={
                        "method": "c2pa",
                        "c2pa_present": True,
                        "ai_generated": ai_generated,
                        "claim_generator": tool,
                    },
                )
            else:
                return DetectorResult(
                    detector_name=self.name, score=0.0, confidence=0.2,
                    details={"method": "c2pa", "c2pa_present": False},
                )
        except ImportError:
            return DetectorResult(
                detector_name=self.name, score=0.0, confidence=0.0,
                error="c2pa-python not installed",
            )
        except Exception:
            return DetectorResult(
                detector_name=self.name, score=0.0, confidence=0.2,
                details={"method": "c2pa", "c2pa_present": False},
            )
