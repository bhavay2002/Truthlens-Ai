from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def _resolve_path(path_value: str | Path) -> Path:
    path_obj = Path(path_value)
    if path_obj.is_absolute():
        return path_obj
    return (PROJECT_ROOT / path_obj).resolve()


@lru_cache(maxsize=1)
def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    resolved_path = _resolve_path(config_path or DEFAULT_CONFIG_PATH)
    with resolved_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}
    return config


def get_config_value(
    config: dict[str, Any],
    *keys: str,
    default: Any = None,
) -> Any:
    current: Any = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def get_path(
    config: dict[str, Any],
    *keys: str,
    default: str | Path,
) -> Path:
    value = get_config_value(config, *keys, default=default)
    return _resolve_path(value)
