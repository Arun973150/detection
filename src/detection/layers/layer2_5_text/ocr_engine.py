"""OCR engine abstraction for text extraction from images."""

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TextRegion:
    """A detected text region in an image."""
    bbox: list[list[int]]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    text: str
    confidence: float


class OCREngine:
    """Abstraction over EasyOCR / PaddleOCR for text extraction."""

    def __init__(self, engine: str = "easyocr", languages: list[str] | None = None):
        self.engine_name = engine
        self.languages = languages or ["en"]
        self._reader = None

    def _init_reader(self):
        if self._reader is not None:
            return

        if self.engine_name == "easyocr":
            try:
                import easyocr
                self._reader = easyocr.Reader(self.languages, gpu=False)
                return
            except ImportError:
                logger.warning("easyocr not installed, trying paddleocr")

        if self.engine_name == "paddleocr" or self._reader is None:
            try:
                from paddleocr import PaddleOCR
                self._reader = PaddleOCR(lang=self.languages[0], use_gpu=False, show_log=False)
                self.engine_name = "paddleocr"
                return
            except ImportError:
                logger.error("No OCR engine available (install easyocr or paddleocr)")

    def extract_text(self, image: np.ndarray) -> list[TextRegion]:
        """Extract text regions from image.

        Args:
            image: RGB uint8 numpy array.

        Returns:
            List of TextRegion objects with bounding boxes, text, and confidence.
        """
        self._init_reader()
        if self._reader is None:
            return []

        try:
            if self.engine_name == "easyocr":
                return self._extract_easyocr(image)
            else:
                return self._extract_paddleocr(image)
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return []

    def _extract_easyocr(self, image: np.ndarray) -> list[TextRegion]:
        results = self._reader.readtext(image)
        regions = []
        for bbox, text, conf in results:
            regions.append(TextRegion(
                bbox=[[int(p[0]), int(p[1])] for p in bbox],
                text=text,
                confidence=float(conf),
            ))
        return regions

    def _extract_paddleocr(self, image: np.ndarray) -> list[TextRegion]:
        result = self._reader.ocr(image, cls=True)
        regions = []
        if result and result[0]:
            for line in result[0]:
                bbox, (text, conf) = line
                regions.append(TextRegion(
                    bbox=[[int(p[0]), int(p[1])] for p in bbox],
                    text=text,
                    confidence=float(conf),
                ))
        return regions
