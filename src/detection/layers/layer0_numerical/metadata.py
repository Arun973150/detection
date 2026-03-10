"""EXIF Metadata and C2PA Provenance Check.

Parses EXIF metadata for consistency checks and looks for C2PA Content
Credentials indicating AI generation or editing provenance.
"""

from typing import Optional

import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS

from detection.base.detector import BaseDetector
from detection.base.result import DetectorResult

# Known AI tool signatures in EXIF software field
AI_SOFTWARE_SIGNATURES = [
    "stable diffusion", "midjourney", "dall-e", "dalle", "firefly",
    "comfyui", "automatic1111", "invokeai", "novelai", "leonardo",
    "playground", "ideogram", "flux", "imagen",
]


class MetadataDetector(BaseDetector):
    name = "metadata"

    def analyze(self, image: np.ndarray, metadata: Optional[dict] = None) -> DetectorResult:
        # Metadata needs the original file, not the numpy array
        # Check if file path is available in metadata
        file_path = (metadata or {}).get("file_path")
        exif_data = {}
        c2pa_result = {}
        flags = []

        if file_path:
            exif_data = self._extract_exif(file_path)
            c2pa_result = self._check_c2pa(file_path)

        # Analyze EXIF
        has_exif = bool(exif_data)
        has_camera = bool(exif_data.get("Make") or exif_data.get("Model"))
        software = exif_data.get("Software", "").lower()
        ai_software = any(sig in software for sig in AI_SOFTWARE_SIGNATURES)

        # Score based on metadata signals
        score = 0.0
        confidence = 0.4

        if ai_software:
            score = 0.9
            confidence = 0.85
            flags.append("ai_software_detected")

        if c2pa_result.get("ai_generated"):
            score = max(score, 0.95)
            confidence = 0.9
            flags.append("c2pa_ai_generated")

        if not has_exif:
            # Missing EXIF is mildly suspicious (could be stripped)
            score = max(score, 0.2)
            flags.append("no_exif_data")

        if has_camera and not self._validate_camera_consistency(exif_data):
            score = max(score, 0.5)
            flags.append("inconsistent_camera_metadata")

        return DetectorResult(
            detector_name=self.name,
            score=score,
            confidence=confidence,
            details={
                "has_exif": has_exif,
                "has_camera_info": has_camera,
                "software": exif_data.get("Software", ""),
                "camera": f"{exif_data.get('Make', '')} {exif_data.get('Model', '')}".strip(),
                "ai_software_detected": ai_software,
                "c2pa": c2pa_result,
                "flags": flags,
            },
        )

    def _extract_exif(self, file_path: str) -> dict:
        """Extract EXIF data from image file."""
        try:
            img = Image.open(file_path)
            exif = img.getexif()
            if not exif:
                return {}
            return {TAGS.get(k, k): v for k, v in exif.items() if isinstance(v, (str, int, float))}
        except Exception:
            return {}

    def _check_c2pa(self, file_path: str) -> dict:
        """Check for C2PA Content Credentials."""
        try:
            from c2pa import Reader

            reader = Reader.from_file(file_path)
            manifest = reader.get_active_manifest()
            if manifest:
                actions = manifest.get("assertions", [])
                ai_generated = any(
                    "ai_generated" in str(a).lower() or "ai_generative" in str(a).lower()
                    for a in actions
                )
                return {"present": True, "ai_generated": ai_generated}
        except ImportError:
            return {"present": False, "error": "c2pa-python not installed"}
        except Exception:
            pass
        return {"present": False}

    def _validate_camera_consistency(self, exif: dict) -> bool:
        """Basic consistency checks on camera EXIF data."""
        make = exif.get("Make", "")
        model = exif.get("Model", "")
        software = exif.get("Software", "")

        # Check if software matches camera maker
        if make and software:
            make_lower = make.lower()
            # Camera manufacturers typically have their own software
            if any(ai in software.lower() for ai in AI_SOFTWARE_SIGNATURES):
                return False

        return True
