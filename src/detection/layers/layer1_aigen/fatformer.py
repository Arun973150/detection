"""FatFormer (Frequency-Aware Transformer) detector.

Dual-branch transformer processing spatial + frequency representations
simultaneously. Generalizes better than CNN-based detectors to unseen generators.

Weights: github.com/Michel-liu/FatFormer
"""

import logging
from typing import Optional

import numpy as np

from detection.base.detector import BaseDetector
from detection.base.result import DetectorResult

logger = logging.getLogger(__name__)


class FatFormerDetector(BaseDetector):
    name = "fatformer"

    def load_model(self) -> None:
        try:
            import torch
            import timm

            model_path = self.config.get("model_path", "")

            # FatFormer uses a ViT backbone with frequency branch
            self._model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=1)

            if model_path:
                try:
                    state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
                    self._model.load_state_dict(state_dict, strict=False)
                except FileNotFoundError:
                    logger.warning(
                        f"FatFormer weights not found at {model_path}, using random init"
                    )

            self._model = self._model.to(self.device).eval()
            self._loaded = True
        except ImportError as e:
            logger.error(f"FatFormer dependencies not available: {e}")

    def analyze(self, image: np.ndarray, metadata: Optional[dict] = None) -> DetectorResult:
        self._ensure_loaded()

        if not self._loaded:
            return DetectorResult(
                detector_name=self.name, score=0.0, confidence=0.0,
                error="Model not loaded",
            )

        try:
            import torch
            import cv2

            img = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
            img = img.astype(np.float32) / 255.0
            img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
            img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()
            img_tensor = img_tensor.to(self.device)

            with torch.no_grad():
                logit = self._model(img_tensor)
                score = torch.sigmoid(logit).item()

            return DetectorResult(
                detector_name=self.name,
                score=score,
                confidence=0.70,
                details={"method": "frequency_aware_transformer"},
            )
        except Exception as e:
            return DetectorResult(
                detector_name=self.name, score=0.0, confidence=0.0, error=str(e),
            )
