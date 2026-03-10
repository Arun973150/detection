"""Image loading, decoding, and preprocessing utilities."""

from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def load_image(source: str | Path | bytes | BytesIO) -> np.ndarray:
    """Load an image from file path, bytes, or BytesIO and return RGB uint8 numpy array.

    Args:
        source: File path, raw bytes, or BytesIO object.

    Returns:
        RGB numpy array with shape (H, W, 3) and dtype uint8.

    Raises:
        ValueError: If the image cannot be decoded.
    """
    if isinstance(source, (str, Path)):
        img = Image.open(source).convert("RGB")
    elif isinstance(source, bytes):
        img = Image.open(BytesIO(source)).convert("RGB")
    elif isinstance(source, BytesIO):
        img = Image.open(source).convert("RGB")
    else:
        raise ValueError(f"Unsupported source type: {type(source)}")

    return np.array(img)


def resize_image(image: np.ndarray, max_size: int = 1024) -> np.ndarray:
    """Resize image so longest edge is at most max_size, preserving aspect ratio."""
    h, w = image.shape[:2]
    if max(h, w) <= max_size:
        return image
    scale = max_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert RGB image to grayscale."""
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def normalize_for_model(image: np.ndarray, mean: tuple = (0.485, 0.456, 0.406),
                         std: tuple = (0.229, 0.224, 0.225)) -> np.ndarray:
    """Normalize image for ImageNet-pretrained models. Returns float32 CHW array."""
    img = image.astype(np.float32) / 255.0
    img = (img - np.array(mean)) / np.array(std)
    return img.transpose(2, 0, 1)  # HWC -> CHW


def image_to_jpeg_bytes(image: np.ndarray, quality: int = 75) -> bytes:
    """Encode RGB image as JPEG bytes at given quality."""
    img_pil = Image.fromarray(image)
    buffer = BytesIO()
    img_pil.save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def jpeg_bytes_to_image(data: bytes) -> np.ndarray:
    """Decode JPEG bytes back to RGB numpy array."""
    return np.array(Image.open(BytesIO(data)).convert("RGB"))
