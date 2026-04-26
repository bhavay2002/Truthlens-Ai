"""
Dataset cache.

Improvements vs the original:
- Module no longer calls ``load_settings()`` at import time (so importing
  this module does not require the data CSVs to exist).
- File fingerprint uses ``(size, sha256(first 1MB) + sha256(last 1MB))``
  instead of ``mtime``, so ``cp -p`` / ``git checkout`` does not spuriously
  invalidate the cache.
- ``get_cache_key`` accepts arbitrary extra inputs (tokenizer name,
  max_length, cleaning/augmentation config) so changing any of them
  invalidates the cache.
- Cache load/save logs the failed file when corruption is detected.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Any

import pandas as pd

logger = logging.getLogger(__name__)

# bump this if pipeline logic changes
CACHE_VERSION = "v3"

_CACHE_DIR: Optional[Path] = None


# =========================================================
# LAZY SETTINGS
# =========================================================

def _get_cache_dir() -> Path:
    global _CACHE_DIR
    if _CACHE_DIR is None:
        # imported lazily so importing data_cache does not trigger
        # filesystem validation in settings_loader
        from src.config.settings_loader import load_settings
        _CACHE_DIR = load_settings().paths.cache_dir / "data"
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR


# =========================================================
# HASHING
# =========================================================

def _hash_dict(obj: Dict) -> str:
    raw = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


def _file_fingerprint(path: Path) -> Dict[str, Any]:
    """Stable, mtime-free fingerprint: (size, sha256(head+tail))."""
    if not path.exists():
        return {"missing": True}

    size = path.stat().st_size
    h = hashlib.sha256()
    with open(path, "rb") as f:
        head = f.read(1 << 20)  # 1 MB
        h.update(head)
        if size > (2 << 20):
            f.seek(-(1 << 20), 2)  # last 1 MB
            tail = f.read()
            h.update(tail)
    return {"size": size, "sha": h.hexdigest()}


def _hash_files(file_paths: Dict[str, Dict[str, Path]]) -> str:
    fingerprint: Dict[str, Dict[str, Any]] = {}
    for task, splits in file_paths.items():
        fingerprint[task] = {
            split: _file_fingerprint(Path(p)) for split, p in splits.items()
        }
    return _hash_dict(fingerprint)


# =========================================================
# CACHE KEY
# =========================================================

def get_cache_key(
    data_config: Dict,
    file_paths: Dict[str, Dict[str, Path]],
    *,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Cache key derived from:
        - data_config dict
        - file fingerprints (size + content sha)
        - cache version
        - any extra dict (tokenizer, max_length, cleaning, augmentation, …)
    """
    return _hash_dict({
        "config": data_config,
        "files": _hash_files(file_paths),
        "version": CACHE_VERSION,
        "extra": extra or {},
    })


# =========================================================
# SAVE
# =========================================================

def save_cached_datasets(
    datasets: Dict[str, Dict[str, pd.DataFrame]],
    cache_key: str,
) -> None:
    base = _get_cache_dir() / cache_key
    base.mkdir(parents=True, exist_ok=True)
    logger.info("Saving dataset cache → %s", base)

    meta: Dict[str, int] = {}
    for task, splits in datasets.items():
        for split, df in splits.items():
            file = base / f"{task}__{split}.parquet"
            df.to_parquet(file, index=False)
            meta[f"{task}__{split}"] = len(df)

    with open(base / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("Cache saved (%d frames)", len(meta))


# =========================================================
# LOAD
# =========================================================

def load_cached_datasets(cache_key: str) -> Optional[Dict[str, Dict[str, pd.DataFrame]]]:
    base = _get_cache_dir() / cache_key
    if not base.exists():
        logger.info("Cache miss: %s", cache_key[:12])
        return None

    files = list(base.glob("*.parquet"))
    if not files:
        logger.warning("Empty cache directory, ignoring: %s", base)
        return None

    logger.info("Loading cached dataset → %s", base)
    datasets: Dict[str, Dict[str, pd.DataFrame]] = {}

    for file in files:
        name = file.stem
        if "__" not in name:
            continue
        task, split = name.rsplit("__", 1)

        try:
            df = pd.read_parquet(file)
        except Exception as e:
            logger.warning("Cache corruption at %s: %s — invalidating", file, e)
            return None

        datasets.setdefault(task, {})[split] = df

    logger.info("Cache loaded (%d frames)", sum(len(v) for v in datasets.values()))
    return datasets
