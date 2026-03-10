"""Perspective vanishing point consistency analysis.

Uses Hough line detection and RANSAC to find vanishing points.
Multiple inconsistent vanishing points suggest compositing.
"""

import cv2
import numpy as np


def analyze_vanishing_points(
    image: np.ndarray,
    hough_threshold: int = 100,
    max_vanishing_points: int = 3,
    consistency_threshold: float = 50.0,
) -> dict:
    """Analyze perspective consistency via vanishing points.

    Args:
        image: RGB uint8 array.
        hough_threshold: Hough transform vote threshold.
        max_vanishing_points: Expected max vanishing points (1 for 1-point, 2 for 2-point).
        consistency_threshold: Max spread in pixels for consistent VP cluster.

    Returns:
        Dict with vanishing point analysis results.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    # Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Hough line detection
    lines = cv2.HoughLines(edges, 1, np.pi / 180, hough_threshold)

    if lines is None or len(lines) < 4:
        return {
            "score": 0.0,
            "num_lines": 0 if lines is None else len(lines),
            "consistent": True,
        }

    # Convert to line segments (rho, theta format)
    line_params = []
    for line in lines[:200]:  # Limit for performance
        rho, theta = line[0]
        # Skip near-horizontal and near-vertical lines (not useful for VP)
        angle_deg = np.degrees(theta)
        if 5 < angle_deg < 85 or 95 < angle_deg < 175:
            line_params.append((rho, theta))

    if len(line_params) < 4:
        return {
            "score": 0.0,
            "num_lines": len(line_params),
            "consistent": True,
        }

    # Find intersections (candidate vanishing points)
    intersections = []
    for i in range(min(len(line_params), 50)):
        for j in range(i + 1, min(len(line_params), 50)):
            pt = _line_intersection_polar(line_params[i], line_params[j])
            if pt is not None:
                x, y = pt
                # Keep intersections within reasonable range
                if -w < x < 2 * w and -h < y < 2 * h:
                    intersections.append(pt)

    if len(intersections) < 3:
        return {
            "score": 0.0,
            "num_lines": len(line_params),
            "num_intersections": len(intersections),
            "consistent": True,
        }

    # Cluster intersections to find vanishing points (simple approach)
    pts = np.array(intersections, dtype=np.float32)
    vanishing_points = _cluster_vanishing_points(pts, max_clusters=max_vanishing_points + 1)

    # Check consistency: a properly composed scene has at most 3 VPs
    # (1-point, 2-point, or 3-point perspective)
    num_vps = len(vanishing_points)

    # Score: too many VPs suggests inconsistent perspective
    if num_vps > max_vanishing_points:
        score = min(1.0, (num_vps - max_vanishing_points) / 3.0)
        consistent = False
    else:
        score = 0.0
        consistent = True

    return {
        "score": round(score, 4),
        "consistent": consistent,
        "num_vanishing_points": num_vps,
        "expected_max": max_vanishing_points,
        "num_lines": len(line_params),
        "vanishing_points": [
            {"x": round(vp[0], 1), "y": round(vp[1], 1), "support": int(vp[2])}
            for vp in vanishing_points
        ],
    }


def _line_intersection_polar(
    line1: tuple[float, float], line2: tuple[float, float]
) -> tuple[float, float] | None:
    """Find intersection of two lines in (rho, theta) polar form."""
    rho1, theta1 = line1
    rho2, theta2 = line2

    # Skip nearly parallel lines
    if abs(theta1 - theta2) < 0.1:
        return None

    a1, b1 = np.cos(theta1), np.sin(theta1)
    a2, b2 = np.cos(theta2), np.sin(theta2)

    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-10:
        return None

    x = (rho1 * b2 - rho2 * b1) / det
    y = (a1 * rho2 - a2 * rho1) / det

    return (float(x), float(y))


def _cluster_vanishing_points(
    points: np.ndarray, max_clusters: int = 4, min_support: int = 3
) -> list[tuple[float, float, int]]:
    """Simple distance-based clustering of candidate vanishing points.

    Returns list of (x, y, support_count) tuples.
    """
    if len(points) == 0:
        return []

    # Use iterative farthest-point clustering
    cluster_radius = 100.0  # pixels
    clusters: list[tuple[float, float, int]] = []
    remaining = points.copy()

    for _ in range(max_clusters + 2):
        if len(remaining) < min_support:
            break

        # Find densest point
        best_idx = 0
        best_count = 0
        for i in range(min(len(remaining), 100)):
            dists = np.sqrt(np.sum((remaining - remaining[i]) ** 2, axis=1))
            count = np.sum(dists < cluster_radius)
            if count > best_count:
                best_count = count
                best_idx = i

        if best_count < min_support:
            break

        # Form cluster
        center = remaining[best_idx]
        dists = np.sqrt(np.sum((remaining - center) ** 2, axis=1))
        mask = dists < cluster_radius
        cluster_pts = remaining[mask]
        centroid = np.mean(cluster_pts, axis=0)
        clusters.append((float(centroid[0]), float(centroid[1]), int(mask.sum())))

        # Remove clustered points
        remaining = remaining[~mask]

    return clusters
