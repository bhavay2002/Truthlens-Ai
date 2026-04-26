from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any


# =========================================================
# CONFIG KEYS (STANDARDIZED)
# =========================================================

REQUIRED_SPLITS = ("train", "val", "test")


# =========================================================
# CORE RESOLVER
# =========================================================

def resolve_data_config(
    config: Dict[str, Dict[str, str]],
    *,
    env_var: str = "DATA_DIR",
    strict: bool = True,
) -> Dict[str, Dict[str, Path]]:
    """
    Resolve dataset paths for all tasks and splits.

    Args:
        config:
            {
                "bias": {
                    "train": "bias/train.csv",
                    "val": "bias/val.csv",
                    "test": "bias/test.csv"
                },
                ...
            }

        env_var:
            Environment variable for base directory.

        strict:
            If True, raise error if any file missing.

    Returns:
        {
            "bias": {
                "train": Path(...),
                "val": Path(...),
                "test": Path(...)
            },
            ...
        }
    """

    base_dir = os.environ.get(env_var, "")
    base_path = Path(base_dir) if base_dir else None

    resolved: Dict[str, Dict[str, Path]] = {}

    for task, split_map in config.items():

        if not isinstance(split_map, dict):
            raise ValueError(f"{task} config must be dict")

        resolved[task] = {}

        for split in REQUIRED_SPLITS:

            if split not in split_map:
                raise ValueError(f"{task} missing split: {split}")

            raw_path = Path(split_map[split])

            path = (
                (base_path / raw_path).resolve()
                if base_path
                else raw_path.resolve()
            )

            if strict and not path.exists():
                raise FileNotFoundError(
                    f"[{task}][{split}] File not found: {path}"
                )

            resolved[task][split] = path

    return resolved


# =========================================================
# SINGLE PATH RESOLVER (UTILITY)
# =========================================================

def resolve_path(
    path: str | Path,
    *,
    env_var: str = "DATA_DIR",
    strict: bool = True,
) -> Path:
    """
    Resolve a single path with optional environment base.
    """

    base_dir = os.environ.get(env_var, "")
    base_path = Path(base_dir) if base_dir else None

    p = Path(path)

    resolved = (base_path / p).resolve() if base_path else p.resolve()

    if strict and not resolved.exists():
        raise FileNotFoundError(f"File not found: {resolved}")

    return resolved


# =========================================================
# DEBUG / LOGGING
# =========================================================

def pretty_print_config(resolved: Dict[str, Dict[str, Path]]) -> None:
    """
    Print resolved dataset structure (for debugging).
    """

    print("\n📦 DATA CONFIG:")
    for task, splits in resolved.items():
        print(f"\n🔹 {task}")
        for split, path in splits.items():
            print(f"   {split}: {path}")