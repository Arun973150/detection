"""One-shot model weight download script.

Downloads all required pretrained weights from their respective sources.
Run this once before first use: python scripts/download_weights.py
"""

import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Model registry: name -> (url, filename, sha256)
# These are placeholder URLs — replace with actual weight download URLs
MODEL_REGISTRY = {
    "univfd": {
        "url": "https://huggingface.co/models/univfd/resolve/main/linear_probe.pth",
        "filename": "univfd_linear_probe.pth",
        "sha256": None,  # Add after first download
    },
    "npr": {
        "url": "https://huggingface.co/models/npr/resolve/main/resnet50.pth",
        "filename": "npr_resnet50.pth",
        "sha256": None,
    },
    "drct": {
        "url": "https://huggingface.co/models/drct/resolve/main/checkpoint.pth",
        "filename": "drct_checkpoint.pth",
        "sha256": None,
    },
    "fatformer": {
        "url": "https://huggingface.co/models/fatformer/resolve/main/checkpoint.pth",
        "filename": "fatformer_checkpoint.pth",
        "sha256": None,
    },
}


def main():
    cache_dir = Path("model_weights")
    cache_dir.mkdir(exist_ok=True)

    logger.info(f"Downloading model weights to {cache_dir.absolute()}")

    # Import after path setup
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from detection.utils.download import download_file

    for name, info in MODEL_REGISTRY.items():
        dest = cache_dir / info["filename"]
        if dest.exists():
            logger.info(f"[{name}] Already exists: {dest}")
            continue

        logger.info(f"[{name}] Downloading from {info['url']}")
        try:
            download_file(info["url"], dest, sha256=info.get("sha256"))
            logger.info(f"[{name}] Saved to {dest}")
        except Exception as e:
            logger.error(f"[{name}] Download failed: {e}")
            logger.error(f"  Please manually download from the model's repository")

    logger.info("Done. Update config/default.yaml with correct weight paths.")


if __name__ == "__main__":
    main()
