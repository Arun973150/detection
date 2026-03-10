"""Performance timing utilities."""

import functools
import logging
import time

logger = logging.getLogger(__name__)


def timed(func):
    """Decorator that logs execution time of a function."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(f"{func.__qualname__} completed in {elapsed_ms:.1f}ms")
        return result

    return wrapper


class Timer:
    """Context manager for timing code blocks."""

    def __init__(self, label: str = ""):
        self.label = label
        self.elapsed_ms: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000
        if self.label:
            logger.info(f"{self.label}: {self.elapsed_ms:.1f}ms")
