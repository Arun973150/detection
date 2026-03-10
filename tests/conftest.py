"""Shared test fixtures."""

import numpy as np
import pytest


@pytest.fixture
def sample_image():
    """Generate a simple synthetic test image (256x256 RGB)."""
    rng = np.random.RandomState(42)
    return rng.randint(0, 256, (256, 256, 3), dtype=np.uint8)


@pytest.fixture
def gradient_image():
    """Generate a gradient image for testing edge/frequency detectors."""
    y = np.linspace(0, 255, 256).astype(np.uint8)
    x = np.linspace(0, 255, 256).astype(np.uint8)
    yy, xx = np.meshgrid(y, x)
    return np.stack([yy, xx, (yy + xx) // 2], axis=-1).astype(np.uint8)


@pytest.fixture
def blank_image():
    """Generate a uniform white image."""
    return np.full((256, 256, 3), 200, dtype=np.uint8)


@pytest.fixture
def default_config():
    """Return a minimal default config dict."""
    return {
        "enabled": True,
        "ela": {"quality_levels": [75], "threshold": 0.15},
        "dct": {"block_size": 8},
        "srm": {"num_filters": 12},
        "prnu": {"wavelet": "db4", "levels": 2},
        "laplacian": {"patch_size": 32, "stride": 16},
        "chromatic": {"edge_threshold": 50},
        "metadata": {"check_exif": True, "check_c2pa": False},
    }
