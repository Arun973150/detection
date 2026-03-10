"""Model weight downloading utilities."""

import hashlib
import logging
from pathlib import Path

import httpx
from tqdm import tqdm

logger = logging.getLogger(__name__)


def download_file(url: str, dest: Path, sha256: str | None = None, chunk_size: int = 8192) -> Path:
    """Download a file with progress bar and optional checksum verification.

    Args:
        url: URL to download from.
        dest: Local destination path.
        sha256: Expected SHA256 hex digest (optional).
        chunk_size: Download chunk size in bytes.

    Returns:
        Path to downloaded file.

    Raises:
        ValueError: If checksum doesn't match.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and sha256:
        if _compute_sha256(dest) == sha256:
            logger.info(f"File already exists with correct checksum: {dest}")
            return dest

    logger.info(f"Downloading {url} -> {dest}")
    with httpx.stream("GET", url, follow_redirects=True, timeout=300) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as pbar:
            for chunk in response.iter_bytes(chunk_size=chunk_size):
                f.write(chunk)
                pbar.update(len(chunk))

    if sha256:
        actual = _compute_sha256(dest)
        if actual != sha256:
            dest.unlink()
            raise ValueError(f"Checksum mismatch for {dest}: expected {sha256}, got {actual}")

    return dest


def _compute_sha256(path: Path) -> str:
    """Compute SHA256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
