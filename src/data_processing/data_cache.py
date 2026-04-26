from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from src.config.settings_loader import load_settings

logger = logging.getLogger(__name__)

# =========================================================
# SETTINGS
# =========================================================

_settings = load_settings()
CACHE_DIR = _settings.paths.cache_dir / "data"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# bump this if pipeline logic changes
CACHE_VERSION = "v2"


# =========================================================
# HASHING
# =========================================================

def _hash_dict(obj: Dict) -> str:
    raw = json.dumps(obj, sort_keys=True)
    return hashlib.md5(raw.encode()).hexdigest()


def _hash_files(file_paths: Dict[str, Dict[str, Path]]) -> str:
    """
    Hash file metadata (fast, no full file read).
    """
    fingerprint = {}

    for task, splits in file_paths.items():
        fingerprint[task] = {}
        for split, path in splits.items():
            if path.exists():
                stat = path.stat()
                fingerprint[task][split] = {
                    "size": stat.st_size,
                    "mtime": stat.st_mtime,
                }
            else:
                fingerprint[task][split] = "missing"

    return _hash_dict(fingerprint)


# =========================================================
# CACHE KEY
# =========================================================

def get_cache_key(data_config: Dict, file_paths: Dict[str, Dict[str, Path]]) -> str:
    """
    Combines:
    - config
    - file metadata
    - version
    """
    return _hash_dict({
        "config": data_config,
        "files": _hash_files(file_paths),
        "version": CACHE_VERSION,
    })


# =========================================================
# SAVE
# =========================================================

def save_cached_datasets(
    datasets: Dict[str, Dict[str, pd.DataFrame]],
    cache_key: str,
) -> None:

    path = CACHE_DIR / cache_key
    path.mkdir(parents=True, exist_ok=True)

    logger.info("Saving dataset cache → %s", path)

    meta = {}

    for task, splits in datasets.items():
        for split, df in splits.items():

            file = path / f"{task}_{split}.parquet"
            df.to_parquet(file, index=False)

            meta[f"{task}_{split}"] = len(df)

    # save metadata (optional but useful)
    with open(path / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(" Cache saved")


# =========================================================
# LOAD
# =========================================================

def load_cached_datasets(cache_key: str) -> Optional[Dict[str, Dict[str, pd.DataFrame]]]:

    path = CACHE_DIR / cache_key

    if not path.exists():
        logger.info("Cache not found")
        return None

    logger.info("⚡ Loading cached dataset → %s", path)

    datasets: Dict[str, Dict[str, pd.DataFrame]] = {}

    files = list(path.glob("*.parquet"))

    if not files:
        logger.warning(" Empty cache directory, ignoring")
        return None

    for file in files:

        name = file.stem  # bias_train
        if "_" not in name:
            continue

        task, split = name.rsplit("_", 1)

        if task not in datasets:
            datasets[task] = {}

        try:
            datasets[task][split] = pd.read_parquet(file)
        except Exception as e:
            logger.warning("Failed loading %s: %s", file, e)
            return None  # safety fallback

    logger.info("✅ Cache loaded successfully")

    return datasets