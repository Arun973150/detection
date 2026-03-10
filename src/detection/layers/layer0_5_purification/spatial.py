"""Spatial smoothing purification.

Applies Gaussian blur and median filtering to destroy high-frequency
adversarial perturbations while preserving image content.
"""

import cv2
import numpy as np


def gaussian_purify(image: np.ndarray, sigmas: list[float]) -> dict[str, np.ndarray]:
    """Generate Gaussian-blurred variants."""
    variants = {}
    for sigma in sigmas:
        ksize = int(sigma * 6) | 1  # Ensure odd kernel size
        blurred = cv2.GaussianBlur(image, (ksize, ksize), sigma)
        variants[f"gaussian_s{sigma}"] = blurred
    return variants


def median_purify(image: np.ndarray, kernel_sizes: list[int]) -> dict[str, np.ndarray]:
    """Generate median-filtered variants."""
    variants = {}
    for k in kernel_sizes:
        k = k if k % 2 == 1 else k + 1  # Ensure odd
        filtered = cv2.medianBlur(image, k)
        variants[f"median_k{k}"] = filtered
    return variants
