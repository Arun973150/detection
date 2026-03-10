"""Random resized cropping purification.

Applies random crops and resizes to disrupt spatially-aligned adversarial
perturbations.
"""

import cv2
import numpy as np


def random_crop_purify(
    image: np.ndarray, scales: list[float], seed: int = 42
) -> dict[str, np.ndarray]:
    """Generate random-cropped-and-resized variants.

    Args:
        image: RGB uint8 array.
        scales: Crop scale factors (e.g., [0.8, 0.9, 0.95]).
        seed: Random seed for reproducibility.

    Returns:
        Dict mapping variant name to cropped-resized image.
    """
    rng = np.random.RandomState(seed)
    h, w = image.shape[:2]
    variants = {}

    for scale in scales:
        crop_h, crop_w = int(h * scale), int(w * scale)
        top = rng.randint(0, h - crop_h + 1)
        left = rng.randint(0, w - crop_w + 1)
        cropped = image[top : top + crop_h, left : left + crop_w]
        resized = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
        variants[f"crop_s{scale}"] = resized

    return variants
