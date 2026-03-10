"""Bit-depth reduction purification.

Reduces color precision to destroy adversarial perturbations that depend
on least-significant bit manipulation.
"""

import numpy as np


def bitdepth_purify(image: np.ndarray, bit_depths: list[int]) -> dict[str, np.ndarray]:
    """Generate bit-depth-reduced variants.

    Args:
        image: RGB uint8 array.
        bit_depths: Target bit depths per channel (e.g., [6, 5, 4]).

    Returns:
        Dict mapping variant name to reduced image.
    """
    variants = {}
    for bits in bit_depths:
        shift = 8 - bits
        reduced = (image >> shift) << shift
        # Re-expand to full range for visual consistency
        reduced = reduced.astype(np.float32)
        max_val = (255 >> shift) << shift
        if max_val > 0:
            reduced = (reduced / max_val * 255).astype(np.uint8)
        variants[f"bitdepth_{bits}bit"] = reduced
    return variants
