"""
File: config_loader.py

Purpose
-------
Configuration loader for TruthLens AI.

This module reads YAML configuration files and provides
helper functions for retrieving nested values and resolving paths.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------
# Project Root
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


# ---------------------------------------------------------
# Path Resolver
# ---------------------------------------------------------

def _resolve_path(path_value: str | Path) -> Path:
    """
    Convert relative config paths to absolute paths.
    """

    path_obj = Path(path_value)

    if path_obj.is_absolute():

        return path_obj

    return (PROJECT_ROOT / path_obj).resolve()


# ---------------------------------------------------------
# Load Configuration
# ---------------------------------------------------------

@lru_cache(maxsize=1)
def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """
    Load configuration from YAML file.

    Uses caching so the file is only read once.
    """

    resolved_path = _resolve_path(config_path or DEFAULT_CONFIG_PATH)

    if not resolved_path.exists():

        raise FileNotFoundError(f"Config file not found: {resolved_path}")

    with resolved_path.open("r", encoding="utf-8") as config_file:

        config = yaml.safe_load(config_file) or {}

    return config


# ---------------------------------------------------------
# Retrieve Nested Config Values
# ---------------------------------------------------------

def get_config_value(
    config: dict[str, Any],
    *keys: str,
    default: Any = None,
) -> Any:
    """
    Retrieve nested configuration value safely.

    Example
    -------
    get_config_value(config, "model", "name")
    """

    current: Any = config

    for key in keys:

        if not isinstance(current, dict) or key not in current:

            return default

        current = current[key]

    return current


# ---------------------------------------------------------
# Retrieve Path Values
# ---------------------------------------------------------

def get_path(
    config: dict[str, Any],
    *keys: str,
    default: str | Path,
) -> Path:
    """
    Retrieve path value from config and resolve it.
    """

    value = get_config_value(config, *keys, default=default)

    return _resolve_path(value)