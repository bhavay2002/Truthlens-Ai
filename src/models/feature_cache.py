"""
File: feature_cache.py

Purpose
-------
Feature caching utilities for TruthLens AI.

This module stores and retrieves expensive feature computation
results to avoid redundant processing during training and evaluation.
"""

import logging
from pathlib import Path
from typing import Any, Optional

import joblib


# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Feature Cache
# ---------------------------------------------------------

class FeatureCache:
    """
    Utility class for caching feature computation results.
    """

    def __init__(self, cache_dir: str | Path = "cache"):

        self.cache_dir = Path(cache_dir)

        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------
    # Cache Path
    # -------------------------------------------------

    def _get_cache_path(self, key: str) -> Path:

        filename = f"{key}.joblib"

        return self.cache_dir / filename

    # -------------------------------------------------
    # Save Cache
    # -------------------------------------------------

    def save(self, key: str, data: Any) -> Path:
        """
        Save cached data.
        """

        path = self._get_cache_path(key)

        try:

            joblib.dump(data, path)

            logger.info("Saved cache: %s", path)

            return path

        except Exception:

            logger.exception("Failed to save cache")

            raise

    # -------------------------------------------------
    # Load Cache
    # -------------------------------------------------

    def load(self, key: str) -> Optional[Any]:
        """
        Load cached data if available.
        """

        path = self._get_cache_path(key)

        if not path.exists():

            return None

        try:

            logger.info("Loading cache: %s", path)

            return joblib.load(path)

        except Exception:

            logger.exception("Failed to load cache")

            raise

    # -------------------------------------------------
    # Cache Exists
    # -------------------------------------------------

    def exists(self, key: str) -> bool:
        """
        Check if cached object exists.
        """

        return self._get_cache_path(key).exists()

    # -------------------------------------------------
    # Delete Cache
    # -------------------------------------------------

    def delete(self, key: str):
        """
        Remove cached object.
        """

        path = self._get_cache_path(key)

        if path.exists():

            path.unlink()

            logger.info("Deleted cache: %s", path)

    # -------------------------------------------------
    # Clear Cache Directory
    # -------------------------------------------------

    def clear(self):
        """
        Remove all cached features.
        """

        for file in self.cache_dir.glob("*.joblib"):

            file.unlink()

        logger.info("Cache directory cleared")