"""JPEG recompression purification.

Re-saves image at multiple JPEG quality levels to destroy adversarial
perturbations that rely on precise pixel values.
"""

from io import BytesIO

import numpy as np
from PIL import Image


def jpeg_purify(image: np.ndarray, qualities: list[int]) -> dict[str, np.ndarray]:
    """Generate JPEG-recompressed variants of the image.

    Args:
        image: RGB uint8 array.
        qualities: List of JPEG quality levels (e.g., [95, 85, 75, 65]).

    Returns:
        Dict mapping variant name to purified image array.
    """
    variants = {}
    pil_img = Image.fromarray(image)

    for q in qualities:
        buf = BytesIO()
        pil_img.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        recompressed = np.array(Image.open(buf).convert("RGB"))
        variants[f"jpeg_q{q}"] = recompressed

    return variants
