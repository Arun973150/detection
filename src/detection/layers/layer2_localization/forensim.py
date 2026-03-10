"""Forensim — copy-move forgery detection.

Detects regions copied from one part of the image to another.
Identifies both source and target regions.
"""

import logging
from typing import Optional

import cv2
import numpy as np

from detection.base.detector import BaseDetector
from detection.base.result import DetectorResult

logger = logging.getLogger(__name__)


class ForensimDetector(BaseDetector):
    name = "forensim"

    def load_model(self) -> None:
        self._loaded = True

    def analyze(self, image: np.ndarray, metadata: Optional[dict] = None) -> DetectorResult:
        self._ensure_loaded()

        try:
            return self._copy_move_detection(image)
        except Exception as e:
            return DetectorResult(
                detector_name=self.name, score=0.0, confidence=0.0, error=str(e),
            )

    def _copy_move_detection(self, image: np.ndarray) -> DetectorResult:
        """Detect copy-move forgeries using block matching."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape

        # Use ORB feature matching for copy-move detection
        orb = cv2.ORB_create(nfeatures=1000)
        keypoints, descriptors = orb.detectAndCompute(gray, None)

        if descriptors is None or len(keypoints) < 10:
            return DetectorResult(
                detector_name=self.name,
                score=0.0,
                confidence=0.3,
                details={"method": "orb_block_matching", "matches": 0},
            )

        # Match features within the same image
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(descriptors, descriptors, k=5)

        # Filter: keep matches that are spatially separated (not self-matches)
        suspicious_pairs = []
        min_distance = min(h, w) * 0.05  # At least 5% of image size apart

        for match_group in matches:
            for m in match_group:
                if m.queryIdx == m.trainIdx:
                    continue
                pt1 = keypoints[m.queryIdx].pt
                pt2 = keypoints[m.trainIdx].pt
                spatial_dist = np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)
                if spatial_dist > min_distance and m.distance < 40:
                    suspicious_pairs.append((pt1, pt2, m.distance))

        # Build heatmap from suspicious matches
        heatmap = np.zeros((h, w), dtype=np.float32)
        radius = max(20, min(h, w) // 20)

        for pt1, pt2, dist in suspicious_pairs:
            x1, y1 = int(pt1[0]), int(pt1[1])
            x2, y2 = int(pt2[0]), int(pt2[1])
            cv2.circle(heatmap, (x1, y1), radius, 1.0, -1)
            cv2.circle(heatmap, (x2, y2), radius, 1.0, -1)

        # Normalize
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        # Score based on number of suspicious matches
        num_suspicious = len(suspicious_pairs)
        score = min(1.0, num_suspicious / 50.0)

        return DetectorResult(
            detector_name=self.name,
            score=score,
            confidence=0.6 if num_suspicious > 5 else 0.3,
            heatmap=heatmap if num_suspicious > 0 else None,
            details={
                "method": "orb_block_matching",
                "suspicious_matches": num_suspicious,
                "total_keypoints": len(keypoints),
            },
        )
