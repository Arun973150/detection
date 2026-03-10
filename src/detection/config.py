"""Configuration loader with YAML parsing and environment variable interpolation."""

import os
import re
from pathlib import Path
from typing import Any

import yaml


_ENV_VAR_PATTERN = re.compile(r"\$\{(\w+)\}")


def _interpolate_env_vars(value: Any) -> Any:
    """Replace ${VAR_NAME} patterns with environment variable values."""
    if isinstance(value, str):
        def replacer(match):
            var_name = match.group(1)
            return os.environ.get(var_name, match.group(0))
        return _ENV_VAR_PATTERN.sub(replacer, value)
    elif isinstance(value, dict):
        return {k: _interpolate_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_interpolate_env_vars(item) for item in value]
    return value


def load_config(config_path: str | Path = "config/default.yaml") -> dict:
    """Load and parse YAML configuration with env var interpolation."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    return _interpolate_env_vars(raw)


def get_layer_config(config: dict, layer_name: str) -> dict:
    """Extract a specific layer's configuration."""
    return config.get("layers", {}).get(layer_name, {})
