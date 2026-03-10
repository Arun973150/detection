"""Device selection utility for GPU/CPU inference."""

import torch


def get_device(prefer: str = "cuda") -> str:
    """Select the best available device based on preference.

    Args:
        prefer: Preferred device - "cuda", "mps", or "cpu"

    Returns:
        Device string compatible with PyTorch.
    """
    if prefer == "cuda" and torch.cuda.is_available():
        return "cuda"
    if prefer == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
