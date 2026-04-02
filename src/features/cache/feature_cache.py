"""
File Name: feature_cache.py
Module: Feature Engineering - Feature Cache
Description:
    Provides a caching layer for feature extraction in the TruthLens feature
    engineering pipeline. The cache stores computed feature vectors keyed by
    deterministic hashes of the input text and configuration metadata. This
    prevents redundant computation during repeated experiments, evaluation,
    or large-scale batch preprocessing.

    The cache supports:
        • in-memory caching
        • optional disk-backed persistence
        • deterministic hashing
        • safe serialization
        • configurable cache eviction

Dependencies:
    dataclasses
    typing
    logging
    hashlib
    json
    pathlib
    pickle

Inputs:
    FeatureContext / raw text

Outputs:
    Dict[str, float] feature vector (cached or computed)
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from src.features.base.base_feature import FeatureContext

logger = logging.getLogger(__name__)

FeatureVector = Dict[str, float]


def _hash_text(text: str) -> str:
    """
    Compute deterministic hash for text.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class FeatureCache:
    """
    Feature caching system supporting memory and disk storage.
    """

    cache_dir: Optional[Path] = None
    use_memory_cache: bool = True
    max_memory_items: int = 10000

    _memory_cache: Dict[str, FeatureVector] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _disk_path(self, key: str) -> Path:
        """
        Compute disk cache file path.
        """
        if not self.cache_dir:
            raise RuntimeError("Disk cache directory not configured")

        return self.cache_dir / f"{key}.pkl"

    def compute_key(self, context: FeatureContext) -> str:
        """
        Generate cache key for input context.
        """
        if not context.text:
            raise ValueError("FeatureContext.text cannot be empty")

        payload = {
            "text": context.text,
            "tokens": context.tokens,
        }

        serialized = json.dumps(payload, sort_keys=True)

        return _hash_text(serialized)

    def get(self, key: str) -> Optional[FeatureVector]:
        """
        Retrieve cached features.
        """

        if self.use_memory_cache and key in self._memory_cache:
            logger.debug("Feature cache hit (memory)")
            return self._memory_cache[key]

        if self.cache_dir:
            path = self._disk_path(key)

            if path.exists():
                try:
                    with path.open("rb") as f:
                        data = pickle.load(f)

                    if self.use_memory_cache:
                        self._memory_cache[key] = data

                    logger.debug("Feature cache hit (disk)")
                    return data

                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to load cache file: %s", exc)

        return None

    def set(self, key: str, features: FeatureVector) -> None:
        """
        Store features in cache.
        """

        if self.use_memory_cache:
            if len(self._memory_cache) >= self.max_memory_items:
                self._memory_cache.pop(next(iter(self._memory_cache)))

            self._memory_cache[key] = features

        if self.cache_dir:
            path = self._disk_path(key)

            try:
                with path.open("wb") as f:
                    pickle.dump(features, f)

            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to write cache file: %s", exc)

    def get_or_compute(
        self,
        context: FeatureContext,
        compute_fn,
    ) -> FeatureVector:
        """
        Retrieve cached features or compute them.
        """

        key = self.compute_key(context)

        cached = self.get(key)

        if cached is not None:
            return cached

        features = compute_fn(context)

        self.set(key, features)

        return features

    def clear_memory(self) -> None:
        """
        Clear in-memory cache.
        """
        self._memory_cache.clear()

        logger.info("Memory feature cache cleared")

    def clear_disk(self) -> None:
        """
        Remove all disk cache files.
        """

        if not self.cache_dir:
            return

        for file in self.cache_dir.glob("*.pkl"):
            try:
                file.unlink()
            except Exception:  # noqa: BLE001
                pass

        logger.info("Disk feature cache cleared")