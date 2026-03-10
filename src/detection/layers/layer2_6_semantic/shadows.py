"""Shadow direction consistency analysis.

Detects shadow regions and estimates cast directions. Inconsistent shadow
directions across the scene indicate compositing or AI generation.
"""

import cv2
import numpy as np


def analyze_shadow_consistency(
    image: np.ndarray,
    min_shadow_area: int = 500,
    direction_variance_threshold: float = 25.0,
) -> dict:
    """Analyze shadow direction consistency.

    Args:
        image: RGB uint8 array.
        min_shadow_area: Minimum shadow region area in pixels.
        direction_variance_threshold: Max acceptable direction variance (degrees).

    Returns:
        Dict with shadow analysis results.
    """
    # Convert to HSV for shadow detection
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Shadows: low value, low saturation
    v_channel = hsv[:, :, 2]
    s_channel = hsv[:, :, 1]

    # Adaptive threshold for shadows
    v_threshold = np.percentile(v_channel, 20)
    shadow_mask = (v_channel < v_threshold).astype(np.uint8) * 255

    # Clean up with morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_OPEN, kernel)
    shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_CLOSE, kernel)

    # Find shadow contours
    contours, _ = cv2.findContours(shadow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    shadow_directions = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_shadow_area:
            continue

        # Fit ellipse to get shadow direction
        if len(contour) < 5:
            continue

        try:
            ellipse = cv2.fitEllipse(contour)
            center, axes, angle = ellipse
            # Shadow cast direction is along the major axis
            shadow_directions.append({
                "center": [float(center[0]), float(center[1])],
                "angle": float(angle),
                "area": float(area),
                "aspect_ratio": float(max(axes) / (min(axes) + 1e-8)),
            })
        except cv2.error:
            continue

    if len(shadow_directions) < 2:
        return {
            "score": 0.0,
            "consistent": True,
            "num_shadows": len(shadow_directions),
        }

    # Filter to elongated shadows (likely cast shadows, not ambient)
    cast_shadows = [s for s in shadow_directions if s["aspect_ratio"] > 1.5]

    if len(cast_shadows) < 2:
        return {
            "score": 0.0,
            "consistent": True,
            "num_shadows": len(shadow_directions),
            "num_cast_shadows": len(cast_shadows),
        }

    # Analyze direction consistency
    angles = [s["angle"] for s in cast_shadows]
    angles_rad = np.radians(angles)

    # Circular statistics (shadows can point in opposite directions from same light)
    # Use doubled angles to handle 180-degree ambiguity
    doubled = angles_rad * 2
    mean_cos = np.mean(np.cos(doubled))
    mean_sin = np.mean(np.sin(doubled))
    circular_variance = 1 - np.sqrt(mean_cos ** 2 + mean_sin ** 2)
    angle_spread = float(circular_variance * 90)  # 0-90 range after halving

    consistent = angle_spread < direction_variance_threshold
    score = min(1.0, max(0.0, (angle_spread - direction_variance_threshold / 2) / direction_variance_threshold))

    return {
        "score": round(score, 4),
        "consistent": consistent,
        "angle_spread": round(angle_spread, 2),
        "num_shadows": len(shadow_directions),
        "num_cast_shadows": len(cast_shadows),
        "shadow_details": cast_shadows[:5],  # First 5
    }
