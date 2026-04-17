"""
File Name: feature_cache.py
Module: features.cache
Description:
    Feature caching utilities for the TruthLens AI system.

    This module provides a lightweight caching layer for expensive feature
    computation pipelines used during training, evaluation, and inference.
    Cached objects are serialized using joblib and stored in a dedicated
    cache directory.

    The cache prevents redundant feature computation for identical inputs,
    significantly improving performance during iterative experimentation
    and large-scale batch processing.

Author: ML Engineering System
Date: 2026-04-03
Dependencies:
    logging
    pickle
    re
    pathlib
    typing
Inputs:
    key : str
        Unique identifier for cached object

    data : Any
        Serializable Python object
Outputs:
    Cached serialized objects stored on disk
"""

from __future__ import annotations

import hashlib
import logging
import pickle
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class FeatureCache:
    """
    Disk-based cache for feature computation outputs.
    """

    def __init__(self, cache_dir: str | Path = "cache") -> None:
        """
        Initialize cache directory.

        Parameters
        ----------
        cache_dir : str | Path
            Directory where cache files are stored.
        """

        self.cache_dir = Path(cache_dir)
        self._key_cache: Dict[str, str] = {}
        self._path_cache: Dict[str, Path] = {}
        self._lock = threading.Lock()

        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            logger.exception("Failed to create cache directory")
            raise RuntimeError("Unable to initialize cache directory") from exc

    # -------------------------------------------------
    # Internal Utilities
    # -------------------------------------------------

    @staticmethod
    def _normalize_key(key: str) -> str:
        """
        Normalize cache key into a filesystem-safe identifier.
        """

        if not isinstance(key, str):
            raise TypeError("Cache key must be a string")

        normalized = re.sub(r"[^a-zA-Z0-9_.-]+", "_", key.strip())
        normalized = normalized.strip("._")

        if not normalized:
            raise ValueError("Cache key must contain at least one valid character")

        return normalized

    def _get_cache_path(self, key: str) -> Path:
        """
        Construct cache file path for a given key.
        """
        if not isinstance(key, str):
            raise TypeError("Cache key must be a string")

        with self._lock:
            cached_path = self._path_cache.get(key)
            if cached_path is not None:
                return cached_path

            digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
            safe_prefix = self._normalize_key(key)[:40] if key.strip() else "empty"
            filename = f"{safe_prefix}_{digest}.pkl"
            path = self.cache_dir / filename
            self._path_cache[key] = path
            return path

    # -------------------------------------------------
    # Save Cache
    # -------------------------------------------------

    def save(self, key: str, data: Any) -> Path:
        """
        Save object to cache.

        Parameters
        ----------
        key : str
        data : Any

        Returns
        -------
        Path
            Path to cached file.
        """

        path = self._get_cache_path(key)
        path_open = path.open
        pickle_dump = pickle.dump

        try:
            with path_open("wb", buffering=1024 * 1024) as file_obj:
                pickle_dump(data, file_obj, protocol=pickle.HIGHEST_PROTOCOL)
            return path
        except Exception as exc:
            logger.exception("Failed to save cache")
            raise RuntimeError("Cache save failed") from exc

    # -------------------------------------------------
    # Load Cache
    # -------------------------------------------------

    def load(self, key: str) -> Optional[Any]:
        """
        Load cached object if it exists.

        Parameters
        ----------
        key : str

        Returns
        -------
        Optional[Any]
        """

        path = self._get_cache_path(key)
        path_open = path.open
        pickle_load = pickle.load

        try:
            with path_open("rb", buffering=1024 * 1024) as file_obj:
                return pickle_load(file_obj)
        except FileNotFoundError:
            return None
        except Exception as exc:
            logger.exception("Failed to load cache")
            raise RuntimeError("Cache load failed") from exc

    def load_many(self, keys: List[str]) -> List[Optional[Any]]:
        """
        Load multiple cache entries.
        """

        if len(keys) > 50:
            with ThreadPoolExecutor(max_workers=4) as executor:
                return list(executor.map(self.load, keys))

        results: List[Optional[Any]] = []
        append = results.append
        load = self.load

        for key in keys:
            append(load(key))

        return results

    # -------------------------------------------------
    # Cache Exists
    # -------------------------------------------------

    def exists(self, key: str) -> bool:
        """
        Check whether a cached object exists.
        """

        return self._get_cache_path(key).exists()

    # -------------------------------------------------
    # Delete Cache
    # -------------------------------------------------

    def delete(self, key: str) -> None:
        """
        Remove cached object.
        """

        path = self._get_cache_path(key)

        if path.exists():
            try:
                path.unlink()
                logger.info("Deleted cache: %s", path)
            except Exception as exc:
                logger.exception("Failed to delete cache")
                raise RuntimeError("Cache deletion failed") from exc

    # -------------------------------------------------
    # Clear Cache
    # -------------------------------------------------

    def clear(self) -> None:
        """
        Remove all cached feature files.
        """

        try:
            for file in self.cache_dir.glob("*.pkl"):
                file.unlink()

            # Backward-compatible cleanup for older cache files.
            for file in self.cache_dir.glob("*.joblib"):
                file.unlink()

            self._path_cache.clear()

            if __debug__:
                logger.debug("Cache directory cleared: %s", self.cache_dir)

        except Exception as exc:
            logger.exception("Failed to clear cache directory")
            raise RuntimeError("Cache clear operation failed") from exc