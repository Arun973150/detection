"""Font consistency analysis for AI-generated text detection.

Checks stroke width consistency and font metric uniformity within text regions.
AI generators often produce inconsistent letterforms within the same word.
"""

import cv2
import numpy as np

from detection.layers.layer2_5_text.ocr_engine import TextRegion


def analyze_font_consistency(
    image: np.ndarray,
    regions: list[TextRegion],
    stroke_variance_threshold: float = 0.4,
) -> dict:
    """Analyze font consistency across text regions.

    Args:
        image: RGB uint8 array.
        regions: Detected text regions.
        stroke_variance_threshold: Max acceptable stroke width variance ratio.

    Returns:
        Dict with consistency analysis results.
    """
    if not regions:
        return {"score": 0.0, "inconsistent_regions": 0, "total_regions": 0}

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    inconsistent_count = 0
    region_details = []

    for region in regions:
        if not region.text.strip():
            continue

        # Extract region of interest
        roi = _extract_roi(gray, region.bbox)
        if roi is None or roi.size < 100:
            continue

        # Analyze stroke width
        stroke_stats = _analyze_stroke_width(roi)

        # Analyze character height consistency
        height_stats = _analyze_char_heights(roi)

        # Determine if inconsistent
        is_inconsistent = False
        reasons = []

        if stroke_stats["cv"] > stroke_variance_threshold:
            is_inconsistent = True
            reasons.append("stroke_width_variance")

        if height_stats["cv"] > 0.3:
            is_inconsistent = True
            reasons.append("height_variance")

        if is_inconsistent:
            inconsistent_count += 1

        region_details.append({
            "text": region.text[:30],
            "stroke_cv": round(stroke_stats["cv"], 4),
            "height_cv": round(height_stats["cv"], 4),
            "inconsistent": is_inconsistent,
            "reasons": reasons,
        })

    total = len(region_details)
    if total == 0:
        return {"score": 0.0, "inconsistent_regions": 0, "total_regions": 0}

    score = min(1.0, inconsistent_count / total)

    return {
        "score": round(score, 4),
        "inconsistent_regions": inconsistent_count,
        "total_regions": total,
        "details": region_details,
    }


def _extract_roi(gray: np.ndarray, bbox: list[list[int]]) -> np.ndarray | None:
    """Extract and perspective-correct a text region."""
    try:
        pts = np.array(bbox, dtype=np.float32)
        width = int(max(
            np.linalg.norm(pts[0] - pts[1]),
            np.linalg.norm(pts[2] - pts[3]),
        ))
        height = int(max(
            np.linalg.norm(pts[0] - pts[3]),
            np.linalg.norm(pts[1] - pts[2]),
        ))
        if width < 10 or height < 10:
            return None

        dst = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(pts, dst)
        roi = cv2.warpPerspective(gray, M, (width, height))
        return roi
    except Exception:
        return None


def _analyze_stroke_width(roi: np.ndarray) -> dict:
    """Analyze stroke width consistency using distance transform."""
    # Binarize
    _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Distance transform gives distance to nearest background pixel
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 3)

    # Stroke width approximation: values at skeleton points
    skeleton_values = dist[dist > 0]
    if len(skeleton_values) < 5:
        return {"mean": 0.0, "std": 0.0, "cv": 0.0}

    mean_width = float(np.mean(skeleton_values))
    std_width = float(np.std(skeleton_values))
    cv = std_width / mean_width if mean_width > 0 else 0.0

    return {"mean": mean_width, "std": std_width, "cv": cv}


def _analyze_char_heights(roi: np.ndarray) -> dict:
    """Analyze character height consistency."""
    _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find connected components (approximate characters)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

    heights = []
    for i in range(1, num_labels):  # Skip background
        area = stats[i, cv2.CC_STAT_AREA]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        if area > 20:  # Filter noise
            heights.append(h)

    if len(heights) < 3:
        return {"mean": 0.0, "std": 0.0, "cv": 0.0}

    mean_h = float(np.mean(heights))
    std_h = float(np.std(heights))
    cv = std_h / mean_h if mean_h > 0 else 0.0

    return {"mean": mean_h, "std": std_h, "cv": cv}
