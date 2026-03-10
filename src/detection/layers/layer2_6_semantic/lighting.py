"""Lighting direction estimation and consistency checking.

Estimates dominant light source direction from shading gradients across
image regions. Inconsistent lighting directions suggest compositing.
"""

import cv2
import numpy as np


def analyze_lighting_consistency(
    image: np.ndarray, num_regions: int = 4, consistency_threshold: float = 30.0
) -> dict:
    """Analyze lighting direction consistency across image regions.

    Args:
        image: RGB uint8 array.
        num_regions: Number of regions to split image into (2x2=4, 3x3=9, etc.).
        consistency_threshold: Max acceptable angle variance in degrees.

    Returns:
        Dict with lighting analysis results.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float64)
    h, w = gray.shape

    # Compute gradient direction (proxy for light direction)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)

    # Split into regions
    grid_size = int(np.sqrt(num_regions))
    if grid_size < 2:
        grid_size = 2

    region_h = h // grid_size
    region_w = w // grid_size

    region_directions = []

    for i in range(grid_size):
        for j in range(grid_size):
            y1, y2 = i * region_h, (i + 1) * region_h
            x1, x2 = j * region_w, (j + 1) * region_w

            gx = grad_x[y1:y2, x1:x2]
            gy = grad_y[y1:y2, x1:x2]

            # Weight by gradient magnitude (strong gradients = more reliable)
            magnitude = np.sqrt(gx ** 2 + gy ** 2)
            threshold = np.percentile(magnitude, 75)
            mask = magnitude > threshold

            if mask.sum() < 10:
                continue

            # Weighted average gradient direction
            weighted_gx = np.mean(gx[mask] * magnitude[mask])
            weighted_gy = np.mean(gy[mask] * magnitude[mask])

            angle = np.degrees(np.arctan2(weighted_gy, weighted_gx))
            region_directions.append({
                "region": f"({i},{j})",
                "angle": angle,
                "magnitude": float(np.mean(magnitude[mask])),
            })

    if len(region_directions) < 2:
        return {"score": 0.0, "consistent": True, "num_regions": len(region_directions)}

    angles = [r["angle"] for r in region_directions]

    # Compute circular variance (angles wrap around)
    angles_rad = np.radians(angles)
    mean_cos = np.mean(np.cos(angles_rad))
    mean_sin = np.mean(np.sin(angles_rad))
    circular_variance = 1 - np.sqrt(mean_cos ** 2 + mean_sin ** 2)
    mean_angle = float(np.degrees(np.arctan2(mean_sin, mean_cos)))

    # Convert to effective angle spread
    angle_spread = float(circular_variance * 180)  # 0-180 range

    consistent = angle_spread < consistency_threshold
    score = min(1.0, max(0.0, (angle_spread - consistency_threshold / 2) / consistency_threshold))

    return {
        "score": round(score, 4),
        "consistent": consistent,
        "mean_light_direction": round(mean_angle, 2),
        "angle_spread": round(angle_spread, 2),
        "circular_variance": round(float(circular_variance), 4),
        "num_regions": len(region_directions),
        "regions": region_directions,
    }
