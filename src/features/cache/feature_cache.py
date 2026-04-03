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
    re
    pathlib
    typing
    joblib
Inputs:
    key : str
        Unique identifier for cached object

    data : Any
        Serializable Python object
Outputs:
    Cached serialized objects stored on disk
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Optional

import joblib

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

        safe_key = self._normalize_key(key)
        filename = f"{safe_key}.joblib"

        return self.cache_dir / filename

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

        try:
            joblib.dump(data, path)
            logger.info("Saved cache: %s", path)
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

        if not path.exists():
            return None

        try:
            logger.info("Loading cache: %s", path)
            return joblib.load(path)
        except Exception as exc:
            logger.exception("Failed to load cache")
            raise RuntimeError("Cache load failed") from exc

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
            for file in self.cache_dir.glob("*.joblib"):
                file.unlink()

            logger.info("Cache directory cleared: %s", self.cache_dir)

        except Exception as exc:
            logger.exception("Failed to clear cache directory")
            raise RuntimeError("Cache clear operation failed") from exc