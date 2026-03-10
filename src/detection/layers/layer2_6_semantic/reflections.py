"""Reflection plausibility analysis.

Detects reflective surfaces and checks if reflections are geometrically
plausible. Heuristic and lower-confidence than other semantic checks.
"""

import cv2
import numpy as np


def analyze_reflections(image: np.ndarray) -> dict:
    """Analyze reflection plausibility in the image.

    Looks for symmetric patterns near horizontal boundaries that could indicate
    water/mirror reflections and checks their geometric consistency.

    Args:
        image: RGB uint8 array.

    Returns:
        Dict with reflection analysis results.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float64)
    h, w = gray.shape

    # Check for horizontal symmetry (common in water reflections)
    # Split image into upper and lower halves
    half_h = h // 2
    upper = gray[:half_h, :]
    lower = gray[half_h : half_h * 2, :]

    # Flip lower half vertically
    lower_flipped = lower[::-1, :]

    # Ensure same size
    min_h = min(upper.shape[0], lower_flipped.shape[0])
    upper = upper[:min_h, :]
    lower_flipped = lower_flipped[:min_h, :]

    # Compute similarity (normalized cross-correlation)
    if upper.std() > 0 and lower_flipped.std() > 0:
        correlation = float(np.corrcoef(upper.flatten(), lower_flipped.flatten())[0, 1])
    else:
        correlation = 0.0

    has_reflection = correlation > 0.3

    # If reflection detected, check consistency
    score = 0.0
    details = {
        "has_reflection_pattern": has_reflection,
        "symmetry_correlation": round(correlation, 4),
    }

    if has_reflection:
        # Check if the reflection is properly attenuated
        # Real reflections are typically darker/blurrier than the source
        upper_brightness = float(np.mean(upper))
        lower_brightness = float(np.mean(gray[half_h:]))
        brightness_ratio = lower_brightness / (upper_brightness + 1e-8)

        # Real reflections: ratio < 1 (reflection darker)
        # If ratio > 1 or == 1 exactly, suspicious
        if brightness_ratio > 0.95:
            score = 0.4  # Suspiciously bright reflection
            details["suspicious_brightness"] = True
        else:
            score = 0.0
            details["suspicious_brightness"] = False

        details["brightness_ratio"] = round(brightness_ratio, 4)

        # Check blur consistency: reflections should be blurrier
        upper_sharpness = float(cv2.Laplacian(upper.astype(np.uint8), cv2.CV_64F).var())
        lower_sharpness = float(
            cv2.Laplacian(gray[half_h:].astype(np.uint8), cv2.CV_64F).var()
        )
        sharpness_ratio = lower_sharpness / (upper_sharpness + 1e-8)

        if sharpness_ratio > 1.1:
            score = max(score, 0.5)  # Reflection sharper than source
            details["suspicious_sharpness"] = True
        else:
            details["suspicious_sharpness"] = False

        details["sharpness_ratio"] = round(sharpness_ratio, 4)

    return {
        "score": round(score, 4),
        **details,
    }
