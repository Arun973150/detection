"""Layer 0.5: Adversarial Purification.

Generates purified image variants and stores them in context for downstream
layers. Does not produce a detection score — only flags and purified variants.
"""

import logging

import numpy as np

from detection.base.layer import BaseLayer
from detection.base.result import LayerResult
from detection.layers.layer0_5_purification.bitdepth import bitdepth_purify
from detection.layers.layer0_5_purification.crop import random_crop_purify
from detection.layers.layer0_5_purification.diffpure import DiffPurePurifier
from detection.layers.layer0_5_purification.jpeg_purify import jpeg_purify
from detection.layers.layer0_5_purification.spatial import gaussian_purify, median_purify

logger = logging.getLogger(__name__)


class AdversarialPurificationLayer(BaseLayer):
    name = "layer0_5_purification"
    order = 0.5

    def setup(self) -> None:
        self.diffpure = DiffPurePurifier(self.config.get("diffpure", {}), self.device)
        if self.config.get("diffpure", {}).get("enabled", False):
            self.diffpure.load()

    def analyze(self, image: np.ndarray, context: dict) -> LayerResult:
        purified_images: dict[str, np.ndarray] = {}

        # JPEG recompression variants
        jpeg_qualities = self.config.get("jpeg_qualities", [95, 85, 75, 65])
        purified_images.update(jpeg_purify(image, jpeg_qualities))

        # Gaussian blur variants
        sigmas = self.config.get("gaussian_sigmas", [1.0, 2.0])
        purified_images.update(gaussian_purify(image, sigmas))

        # Median filter variants
        median_ks = self.config.get("median_kernel_sizes", [3, 5])
        purified_images.update(median_purify(image, median_ks))

        # Bit-depth reduction variants
        bit_depths = self.config.get("bit_depths", [6, 5])
        purified_images.update(bitdepth_purify(image, bit_depths))

        # Random crop variants
        scales = self.config.get("random_crop", {}).get("scales", [0.8, 0.9, 0.95])
        purified_images.update(random_crop_purify(image, scales))

        # DiffPure (optional, heavy)
        if self.diffpure.enabled:
            diffpure_result = self.diffpure.purify(image)
            if diffpure_result is not None:
                purified_images["diffpure"] = diffpure_result

        logger.info(f"Generated {len(purified_images)} purified variants")

        return LayerResult(
            layer_name=self.name,
            order=self.order,
            aggregated_score=0.0,  # This layer doesn't produce a score
            flags=[],
            metadata={"purified_images": purified_images},
        )
