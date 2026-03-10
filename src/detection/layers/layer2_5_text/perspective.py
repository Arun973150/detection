"""Text perspective alignment analysis.

Checks if text baselines converge to consistent vanishing points, as they
should on real signs, labels, and surfaces. AI generators often produce
text that doesn't follow scene perspective.
"""

import numpy as np

from detection.layers.layer2_5_text.ocr_engine import TextRegion


def analyze_text_perspective(
    regions: list[TextRegion],
    angle_threshold: float = 15.0,
) -> dict:
    """Analyze perspective consistency of text baselines.

    Args:
        regions: Detected text regions with bounding boxes.
        angle_threshold: Maximum acceptable baseline angle variance (degrees).

    Returns:
        Dict with perspective analysis results.
    """
    if len(regions) < 2:
        return {"score": 0.0, "num_baselines": len(regions), "consistent": True}

    # Extract baselines (bottom edge of each text region)
    baselines = []
    for region in regions:
        bbox = region.bbox
        if len(bbox) >= 4:
            # Bottom-left to bottom-right
            p1 = np.array(bbox[3], dtype=np.float64)
            p2 = np.array(bbox[2], dtype=np.float64)

            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            if abs(dx) > 5:  # Skip tiny regions
                angle = np.degrees(np.arctan2(dy, dx))
                baselines.append({
                    "p1": p1.tolist(),
                    "p2": p2.tolist(),
                    "angle": angle,
                    "text": region.text[:20],
                })

    if len(baselines) < 2:
        return {"score": 0.0, "num_baselines": len(baselines), "consistent": True}

    angles = [b["angle"] for b in baselines]

    # Group baselines by proximity (text on same surface should have similar angles)
    # Simple approach: check variance of all baseline angles
    angle_std = float(np.std(angles))
    angle_range = float(max(angles) - min(angles))

    # Check if baselines converge to a consistent vanishing point
    vp_consistency = _check_vanishing_point_consistency(baselines)

    # High angle variance suggests inconsistent perspective
    is_inconsistent = angle_std > angle_threshold or not vp_consistency["consistent"]

    if is_inconsistent:
        score = min(1.0, angle_std / (angle_threshold * 2))
    else:
        score = 0.0

    return {
        "score": round(score, 4),
        "num_baselines": len(baselines),
        "angle_std": round(angle_std, 2),
        "angle_range": round(angle_range, 2),
        "consistent": not is_inconsistent,
        "vanishing_point": vp_consistency,
    }


def _check_vanishing_point_consistency(baselines: list[dict]) -> dict:
    """Check if baselines converge to a consistent vanishing point."""
    if len(baselines) < 2:
        return {"consistent": True, "spread": 0.0}

    # Extend each baseline to find intersection points
    intersections = []

    for i in range(len(baselines)):
        for j in range(i + 1, len(baselines)):
            pt = _line_intersection(
                baselines[i]["p1"], baselines[i]["p2"],
                baselines[j]["p1"], baselines[j]["p2"],
            )
            if pt is not None:
                intersections.append(pt)

    if not intersections:
        return {"consistent": True, "spread": 0.0}

    pts = np.array(intersections)
    centroid = np.mean(pts, axis=0)
    distances = np.sqrt(np.sum((pts - centroid) ** 2, axis=1))
    spread = float(np.std(distances))

    # If intersections are tightly clustered, baselines converge consistently
    consistent = spread < 500  # pixels

    return {
        "consistent": consistent,
        "spread": round(spread, 2),
        "num_intersections": len(intersections),
    }


def _line_intersection(
    p1: list, p2: list, p3: list, p4: list
) -> list[float] | None:
    """Find intersection point of two lines defined by point pairs."""
    x1, y1 = p1[0], p1[1]
    x2, y2 = p2[0], p2[1]
    x3, y3 = p3[0], p3[1]
    x4, y4 = p4[0], p4[1]

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:
        return None  # Parallel lines

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom

    ix = x1 + t * (x2 - x1)
    iy = y1 + t * (y2 - y1)

    # Reject intersections too far from the image (likely parallel-ish lines)
    if abs(ix) > 10000 or abs(iy) > 10000:
        return None

    return [ix, iy]
