"""
File Name: inference_cache.py
Module: Inference Caching System
Description:
    Provides a production-grade caching layer for inference results in the
    TruthLens system. The cache prevents redundant computations when the same
    article or feature set is analyzed repeatedly.

    Typical benefits:
        • Reduced latency for repeated requests
        • Lower GPU/CPU utilization
        • Faster batch evaluation
        • Efficient API responses

    The cache uses deterministic hashing of inputs (e.g., article text or
    feature dictionary) and supports optional disk persistence for long-lived
    caching across sessions.

Dependencies:
    logging
    typing
    dataclasses
    hashlib
    json
    pathlib
    time

Inputs:
    Article text or feature dictionary.

Outputs:
    Cached prediction results retrieved by unique hash keys.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class InferenceCacheConfig:
    """
    Configuration for inference caching behavior.
    """
    cache_dir: str = "cache"
    enable_disk_cache: bool = True
    ttl_seconds: Optional[int] = None
    enable_memory_cache: bool = True


class InferenceCache:
    """
    Cache layer used to store and retrieve inference outputs.

    Supports:
        • in-memory caching
        • disk-based caching
        • TTL-based expiration
    """

    def __init__(self, config: InferenceCacheConfig) -> None:
        self.config = config
        self.memory_cache: Dict[str, Dict[str, Any]] = {}

        self.cache_dir = Path(config.cache_dir)

        if self.config.enable_disk_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info("InferenceCache initialized")

    def _hash_input(self, data: Any) -> str:
        """
        Generate deterministic hash key for cache.
        """

        try:
            if isinstance(data, str):
                payload = data
            else:
                payload = json.dumps(data, sort_keys=True)

            hash_key = hashlib.sha256(payload.encode("utf-8")).hexdigest()

            return hash_key

        except Exception as exc:
            logger.exception("Failed to hash input")
            raise RuntimeError("Cache key generation failed") from exc

    def _cache_path(self, key: str) -> Path:
        """
        Generate disk cache path.
        """
        return self.cache_dir / f"{key}.json"

    def _safe_write(self, path: Path, data: str) -> None:
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(data)
        os.replace(tmp, path)

    def _is_expired(self, timestamp: float) -> bool:
        """
        Check TTL expiration.
        """

        if self.config.ttl_seconds is None:
            return False

        return (time.time() - timestamp) > self.config.ttl_seconds

    def get(self, data: Any) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached result if available.
        """

        key = self._hash_input(data)

        if self.config.enable_memory_cache:

            entry = self.memory_cache.get(key)

            if entry:

                if not self._is_expired(entry["timestamp"]):
                    logger.debug("Memory cache hit")
                    return entry["value"]

                logger.debug("Memory cache expired")
                del self.memory_cache[key]

        if self.config.enable_disk_cache:

            path = self._cache_path(key)

            if path.exists():

                try:
                    with open(path, "r", encoding="utf-8") as f:
                        entry = json.load(f)

                    if not self._is_expired(entry["timestamp"]):
                        logger.debug("Disk cache hit")

                        if self.config.enable_memory_cache:
                            self.memory_cache[key] = entry

                        return entry["value"]

                    logger.debug("Disk cache expired")
                    path.unlink(missing_ok=True)

                except Exception as exc:
                    logger.warning("Failed to read disk cache: %s", exc)

        return None

    def set(self, data: Any, value: Dict[str, Any]) -> None:
        """
        Store inference result in cache.
        """

        key = self._hash_input(data)

        entry = {
            "timestamp": time.time(),
            "value": value,
        }

        if self.config.enable_memory_cache:
            self.memory_cache[key] = entry

        if self.config.enable_disk_cache:

            path = self._cache_path(key)

            try:
                payload = json.dumps(entry)
                self._safe_write(path, payload)

            except Exception as exc:
                logger.warning("Failed to write disk cache: %s", exc)

    def invalidate(self, data: Any) -> None:
        """
        Remove cached entry.
        """

        key = self._hash_input(data)

        if key in self.memory_cache:
            del self.memory_cache[key]

        if self.config.enable_disk_cache:

            path = self._cache_path(key)

            if path.exists():
                path.unlink(missing_ok=True)

        logger.debug("Cache invalidated")

    def clear(self) -> None:
        """
        Clear entire cache.
        """

        self.memory_cache.clear()

        if self.config.enable_disk_cache:

            for file in self.cache_dir.glob("*.json"):
                file.unlink(missing_ok=True)

        logger.info("Cache cleared")