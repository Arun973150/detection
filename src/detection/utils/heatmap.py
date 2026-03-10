"""Heatmap creation and visualization utilities."""

import cv2
import numpy as np


def create_heatmap_overlay(
    image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5, colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """Overlay a heatmap on an image.

    Args:
        image: RGB uint8 array (H, W, 3)
        heatmap: Float32 array (H, W) with values in [0, 1]
        alpha: Blend factor (0 = only image, 1 = only heatmap)
        colormap: OpenCV colormap constant

    Returns:
        RGB uint8 blended image.
    """
    h, w = image.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
    heatmap_uint8 = (np.clip(heatmap_resized, 0, 1) * 255).astype(np.uint8)
    colored = cv2.applyColorMap(heatmap_uint8, colormap)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted(image, 1 - alpha, colored, alpha, 0)
    return blended


def merge_heatmaps(
    heatmaps: list[np.ndarray],
    target_size: tuple[int, int] = (512, 512),
    weights: list[float] | None = None,
    method: str = "weighted_avg",
) -> np.ndarray:
    """Merge multiple heatmaps into one.

    Args:
        heatmaps: List of (H, W) float32 arrays.
        target_size: (width, height) to resize all heatmaps to.
        weights: Optional weights for each heatmap.
        method: "weighted_avg" or "max"

    Returns:
        Merged heatmap as (target_h, target_w) float32 array.
    """
    if not heatmaps:
        return np.zeros((target_size[1], target_size[0]), dtype=np.float32)

    resized = []
    for hm in heatmaps:
        r = cv2.resize(hm.astype(np.float32), target_size, interpolation=cv2.INTER_LINEAR)
        resized.append(r)

    stacked = np.stack(resized, axis=0)

    if method == "max":
        return np.max(stacked, axis=0)

    # weighted average
    if weights is None:
        weights = [1.0 / len(heatmaps)] * len(heatmaps)
    w = np.array(weights).reshape(-1, 1, 1)
    w = w / w.sum()
    return np.sum(stacked * w, axis=0)


def normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    """Normalize heatmap to [0, 1] range."""
    mn, mx = heatmap.min(), heatmap.max()
    if mx - mn < 1e-8:
        return np.zeros_like(heatmap)
    return (heatmap - mn) / (mx - mn)
