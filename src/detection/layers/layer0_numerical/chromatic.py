"""Chromatic Aberration Analysis.

Real lenses produce color fringing at high-contrast edges. AI generators
often produce images with perfectly clean edges or physically inconsistent
chromatic aberration patterns.
"""

from typing import Optional

import cv2
import numpy as np

from detection.base.detector import BaseDetector
from detection.base.result import DetectorResult


class ChromaticAberrationDetector(BaseDetector):
    name = "chromatic_aberration"

    def analyze(self, image: np.ndarray, metadata: Optional[dict] = None) -> DetectorResult:
        edge_threshold = self.config.get("edge_threshold", 50)

        # Split into channels
        r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]

        # Find edges in each channel
        edges_r = cv2.Canny(r, edge_threshold, edge_threshold * 2)
        edges_g = cv2.Canny(g, edge_threshold, edge_threshold * 2)
        edges_b = cv2.Canny(b, edge_threshold, edge_threshold * 2)

        # Measure edge displacement between channels
        # Real CA: edges in R and B are slightly shifted from G
        rg_shift = self._measure_edge_shift(edges_r, edges_g)
        bg_shift = self._measure_edge_shift(edges_b, edges_g)

        # Measure spatial consistency of CA
        h, w = image.shape[:2]
        region_shifts = self._regional_shifts(r, g, b, edge_threshold, h, w)

        # Real CA increases from center to edges (radial pattern)
        ca_consistency = self._check_radial_consistency(region_shifts, h, w)

        # Score: no CA at all or inconsistent CA suggests synthetic
        has_ca = (rg_shift > 0.5 or bg_shift > 0.5)

        if not has_ca:
            # No chromatic aberration — suspicious for claimed camera photos
            score = 0.4
        elif ca_consistency < 0.3:
            # Inconsistent CA — very suspicious
            score = 0.7
        else:
            # Consistent radial CA — looks authentic
            score = 0.1

        return DetectorResult(
            detector_name=self.name,
            score=score,
            confidence=0.5,
            details={
                "rg_shift": round(rg_shift, 4),
                "bg_shift": round(bg_shift, 4),
                "has_ca": has_ca,
                "ca_consistency": round(ca_consistency, 4),
            },
        )

    def _measure_edge_shift(self, edges1: np.ndarray, edges2: np.ndarray) -> float:
        """Measure average spatial displacement between two edge maps."""
        if edges1.sum() == 0 or edges2.sum() == 0:
            return 0.0

        # Compute distance transform of edges2
        dist = cv2.distanceTransform(255 - edges2, cv2.DIST_L2, 3)
        # Average distance of edges1 from edges2
        edge_points = edges1 > 0
        if edge_points.sum() == 0:
            return 0.0
        return float(np.mean(dist[edge_points]))

    def _regional_shifts(
        self, r: np.ndarray, g: np.ndarray, b: np.ndarray,
        threshold: int, h: int, w: int,
    ) -> list[dict]:
        """Compute CA shifts in image quadrants."""
        regions = []
        half_h, half_w = h // 2, w // 2

        quadrants = [
            (0, 0, half_h, half_w),
            (0, half_w, half_h, w),
            (half_h, 0, h, half_w),
            (half_h, half_w, h, w),
        ]

        for y1, x1, y2, x2 in quadrants:
            er = cv2.Canny(r[y1:y2, x1:x2], threshold, threshold * 2)
            eg = cv2.Canny(g[y1:y2, x1:x2], threshold, threshold * 2)
            shift = self._measure_edge_shift(er, eg)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            dist_from_center = np.sqrt((cx - w / 2) ** 2 + (cy - h / 2) ** 2)
            regions.append({"shift": shift, "dist": dist_from_center})

        return regions

    def _check_radial_consistency(self, region_shifts: list[dict], h: int, w: int) -> float:
        """Check if CA increases radially from center (as real lenses produce)."""
        if not region_shifts:
            return 0.5

        # Sort by distance from center
        sorted_regions = sorted(region_shifts, key=lambda x: x["dist"])
        shifts = [r["shift"] for r in sorted_regions]

        if len(shifts) < 2:
            return 0.5

        # Check if shifts increase with distance (positive correlation)
        dists = [r["dist"] for r in sorted_regions]
        if np.std(dists) < 1e-8 or np.std(shifts) < 1e-8:
            return 0.5

        correlation = float(np.corrcoef(dists, shifts)[0, 1])
        # Positive correlation = radially increasing CA = authentic
        return max(0.0, min(1.0, (correlation + 1) / 2))
