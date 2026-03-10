"""Local deepfake detection fallback using EfficientNet + face detection."""

import logging
from typing import Optional

import cv2
import numpy as np

from detection.base.detector import BaseDetector
from detection.base.result import DetectorResult

logger = logging.getLogger(__name__)


class LocalDeepfakeDetector(BaseDetector):
    name = "local_deepfake"

    def load_model(self) -> None:
        try:
            import torch
            import timm

            model_name = self.config.get("model", "efficientnet_b4")
            model_path = self.config.get("model_path", "")

            self._model = timm.create_model(model_name, pretrained=False, num_classes=1)

            if model_path:
                try:
                    state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
                    self._model.load_state_dict(state_dict, strict=False)
                except FileNotFoundError:
                    logger.warning(f"Deepfake model weights not found at {model_path}")

            self._model = self._model.to(self.device).eval()

            # Face detector
            self._face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )

            self._loaded = True
        except ImportError as e:
            logger.error(f"Local deepfake detector dependencies not available: {e}")

    def analyze(self, image: np.ndarray, metadata: Optional[dict] = None) -> DetectorResult:
        self._ensure_loaded()

        if not self._loaded:
            return DetectorResult(
                detector_name=self.name, score=0.0, confidence=0.0,
                error="Model not loaded",
            )

        # Detect faces first
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        min_face = self.config.get("min_face_size", 40)
        faces = self._face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_face, min_face)
        )

        if len(faces) == 0:
            return DetectorResult(
                detector_name=self.name, score=0.0, confidence=0.0,
                details={"faces_found": 0, "method": "no_faces_skipped"},
            )

        # Analyze each face
        try:
            import torch

            face_scores = []
            for x, y, fw, fh in faces:
                # Extract face with margin
                margin = int(max(fw, fh) * 0.2)
                y1 = max(0, y - margin)
                x1 = max(0, x - margin)
                y2 = min(image.shape[0], y + fh + margin)
                x2 = min(image.shape[1], x + fw + margin)
                face_crop = image[y1:y2, x1:x2]

                # Preprocess
                face_resized = cv2.resize(face_crop, (224, 224))
                face_norm = face_resized.astype(np.float32) / 255.0
                face_norm = (face_norm - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
                tensor = torch.from_numpy(face_norm.transpose(2, 0, 1)).unsqueeze(0).float()
                tensor = tensor.to(self.device)

                with torch.no_grad():
                    logit = self._model(tensor)
                    score = torch.sigmoid(logit).item()
                    face_scores.append(score)

            max_score = max(face_scores) if face_scores else 0.0

            return DetectorResult(
                detector_name=self.name,
                score=max_score,
                confidence=0.6,
                details={
                    "faces_found": len(faces),
                    "face_scores": [round(s, 4) for s in face_scores],
                    "method": "efficientnet_face_analysis",
                },
            )
        except Exception as e:
            return DetectorResult(
                detector_name=self.name, score=0.0, confidence=0.0, error=str(e),
            )
