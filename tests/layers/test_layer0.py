"""Tests for Layer 0 numerical detectors."""

import numpy as np
import pytest

from detection.layers.layer0_numerical.ela import ELADetector
from detection.layers.layer0_numerical.dct import DCTDetector
from detection.layers.layer0_numerical.laplacian import LaplacianDetector
from detection.layers.layer0_numerical.srm import SRMDetector


class TestELA:
    def test_returns_valid_result(self, sample_image):
        detector = ELADetector({"quality_levels": [75], "threshold": 0.15}, "cpu")
        result = detector.analyze(sample_image)
        assert result.detector_name == "ela"
        assert 0 <= result.score <= 1
        assert result.heatmap is not None
        assert result.heatmap.shape == sample_image.shape[:2]

    def test_blank_image_low_score(self, blank_image):
        detector = ELADetector({"quality_levels": [75], "threshold": 0.15}, "cpu")
        result = detector.analyze(blank_image)
        assert result.score < 0.5


class TestDCT:
    def test_returns_valid_result(self, sample_image):
        detector = DCTDetector({"block_size": 8}, "cpu")
        result = detector.analyze(sample_image)
        assert result.detector_name == "dct"
        assert 0 <= result.score <= 1
        assert "mid_ratio" in result.details


class TestLaplacian:
    def test_returns_valid_result(self, sample_image):
        detector = LaplacianDetector({"patch_size": 32, "stride": 16}, "cpu")
        result = detector.analyze(sample_image)
        assert result.detector_name == "laplacian"
        assert 0 <= result.score <= 1
        assert result.heatmap is not None

    def test_uniform_image_low_score(self, blank_image):
        detector = LaplacianDetector({"patch_size": 32, "stride": 16}, "cpu")
        result = detector.analyze(blank_image)
        assert result.score < 0.3


class TestSRM:
    def test_returns_valid_result(self, sample_image):
        detector = SRMDetector({}, "cpu")
        result = detector.analyze(sample_image)
        assert result.detector_name == "srm"
        assert 0 <= result.score <= 1
        assert result.heatmap is not None
