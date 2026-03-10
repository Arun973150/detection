"""Abstract base class for individual detection algorithms."""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from detection.base.result import DetectorResult


class BaseDetector(ABC):
    """A single detection algorithm (e.g., ELA, UnivFD, TruFor).

    Subclasses implement analyze() to process an image and return a DetectorResult.
    Models are loaded lazily on first call to analyze() unless load_model() is called explicitly.
    """

    name: str = "base_detector"

    def __init__(self, config: dict, device: str = "cpu"):
        self.config = config
        self.device = device
        self._model = None
        self._loaded = False

    @abstractmethod
    def analyze(self, image: np.ndarray, metadata: Optional[dict] = None) -> DetectorResult:
        """Run detection on a single image.

        Args:
            image: RGB numpy array, uint8, shape (H, W, 3)
            metadata: optional dict of prior results from earlier layers

        Returns:
            DetectorResult with score, optional heatmap, optional details dict
        """
        ...

    def load_model(self) -> None:
        """Load model weights into memory. Override for ML-based detectors."""
        self._loaded = True

    def unload_model(self) -> None:
        """Free model memory."""
        self._model = None
        self._loaded = False

    def _ensure_loaded(self) -> None:
        """Lazily load model on first use."""
        if not self._loaded:
            self.load_model()
