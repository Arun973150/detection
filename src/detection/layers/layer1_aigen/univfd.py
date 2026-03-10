"""UnivFD (Universal Fake Detector) — CLIP-based AI image detector.

Uses frozen CLIP ViT-L/14 features with a trained linear probe to detect
AI-generated images across generator families.

Weights: github.com/Yuheng-Li/UniversalFakeDetect
"""

import logging
from typing import Optional

import numpy as np

from detection.base.detector import BaseDetector
from detection.base.result import DetectorResult

logger = logging.getLogger(__name__)


class UnivFDDetector(BaseDetector):
    name = "univfd"

    def load_model(self) -> None:
        try:
            import torch
            import open_clip

            clip_model_name = self.config.get("clip_model", "ViT-L/14")
            model_path = self.config.get("model_path", "")

            # Load CLIP backbone (frozen)
            self._clip_model, _, self._preprocess = open_clip.create_model_and_transforms(
                clip_model_name, pretrained="openai"
            )
            self._clip_model = self._clip_model.to(self.device).eval()

            # Load linear probe weights
            if model_path:
                try:
                    state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
                    in_features = state_dict.get("weight", state_dict.get("fc.weight")).shape[1]
                    self._probe = torch.nn.Linear(in_features, 1).to(self.device)
                    self._probe.load_state_dict(state_dict)
                    self._probe.eval()
                except FileNotFoundError:
                    logger.warning(f"UnivFD weights not found at {model_path}, using random probe")
                    self._probe = torch.nn.Linear(768, 1).to(self.device)
            else:
                self._probe = torch.nn.Linear(768, 1).to(self.device)

            self._loaded = True
        except ImportError as e:
            logger.error(f"UnivFD dependencies not available: {e}")

    def analyze(self, image: np.ndarray, metadata: Optional[dict] = None) -> DetectorResult:
        self._ensure_loaded()

        if not self._loaded:
            return DetectorResult(
                detector_name=self.name, score=0.0, confidence=0.0,
                error="Model not loaded",
            )

        try:
            import torch
            from PIL import Image

            pil_img = Image.fromarray(image)
            img_tensor = self._preprocess(pil_img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                features = self._clip_model.encode_image(img_tensor)
                features = features / features.norm(dim=-1, keepdim=True)
                logit = self._probe(features.float())
                score = torch.sigmoid(logit).item()

            return DetectorResult(
                detector_name=self.name,
                score=score,
                confidence=0.75,
                details={"method": "clip_linear_probe"},
            )
        except Exception as e:
            return DetectorResult(
                detector_name=self.name, score=0.0, confidence=0.0, error=str(e),
            )
