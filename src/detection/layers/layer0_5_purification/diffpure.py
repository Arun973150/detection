"""DiffPure-style diffusion-based purification (optional, GPU-heavy).

Forward-diffuses the image for t steps, then reverses, removing adversarial
perturbations while preserving semantic content. Disabled by default.
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class DiffPurePurifier:
    """Wraps a small diffusion model for adversarial purification."""

    def __init__(self, config: dict, device: str = "cpu"):
        self.config = config
        self.device = device
        self.pipeline = None
        self.enabled = config.get("enabled", False)

    def load(self) -> None:
        """Load the diffusion model."""
        if not self.enabled:
            return

        try:
            import torch
            from diffusers import DDPMPipeline

            model_id = self.config.get("model", "google/ddpm-cifar10-32")
            self.pipeline = DDPMPipeline.from_pretrained(model_id).to(self.device)
            logger.info(f"DiffPure model loaded: {model_id}")
        except Exception as e:
            logger.warning(f"Failed to load DiffPure model: {e}")
            self.enabled = False

    def purify(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Apply diffusion-based purification.

        Returns purified image or None if not available.
        """
        if not self.enabled or self.pipeline is None:
            return None

        try:
            import torch
            from PIL import Image

            timesteps = self.config.get("timesteps", 50)

            # Resize to model's expected size
            pil_img = Image.fromarray(image)
            orig_size = pil_img.size
            pil_img = pil_img.resize((256, 256))

            # Convert to tensor
            img_tensor = torch.from_numpy(np.array(pil_img)).float() / 127.5 - 1.0
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)

            scheduler = self.pipeline.scheduler

            # Forward diffusion
            noise = torch.randn_like(img_tensor)
            t = torch.tensor([timesteps], device=self.device)
            noisy = scheduler.add_noise(img_tensor, noise, t)

            # Reverse diffusion
            for step in range(timesteps, 0, -1):
                t_step = torch.tensor([step], device=self.device)
                with torch.no_grad():
                    pred_noise = self.pipeline.unet(noisy, t_step).sample
                noisy = scheduler.step(pred_noise, step, noisy).prev_sample

            # Convert back
            purified = ((noisy.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1) * 127.5)
            purified = np.clip(purified, 0, 255).astype(np.uint8)

            # Resize back to original
            purified_pil = Image.fromarray(purified).resize(orig_size)
            return np.array(purified_pil)

        except Exception as e:
            logger.warning(f"DiffPure purification failed: {e}")
            return None
