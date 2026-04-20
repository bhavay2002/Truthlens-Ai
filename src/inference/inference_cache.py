from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
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
    max_memory_items: int = 1024
    max_disk_items: Optional[int] = None


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
        self.memory_cache: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._lock = Lock()
        self._inflight: Dict[str, Lock] = {}

        self.cache_dir = Path(config.cache_dir)

        if self.config.enable_disk_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info("InferenceCache initialized")

    def _hash_input(self, data: Any) -> str:
        """
        Generate deterministic hash key for cache.
        """

        try:
            def default(obj: Any) -> Any:
                if hasattr(obj, "tolist"):
                    return obj.tolist()
                if isinstance(obj, set):
                    return sorted(obj)
                return repr(obj)

            payload = (
                data
                if isinstance(data, str)
                else json.dumps(data, sort_keys=True, default=default)
            )

            return hashlib.sha256(payload.encode("utf-8")).hexdigest()

        except Exception as exc:
            logger.exception("Failed to hash input")
            raise RuntimeError("Cache key generation failed") from exc

    def _cache_path(self, key: str) -> Path:
        """
        Generate disk cache path.
        """
        return self.cache_dir / f"{key}.json"

    def _safe_write(self, path: Path, data: str) -> None:
        tmp_path = path.with_suffix(".tmp")

        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp_path, path)

    def _is_expired(self, timestamp: float) -> bool:
        """
        Check TTL expiration.
        """

        if self.config.ttl_seconds is None:
            return False

        return (time.monotonic() - timestamp) > self.config.ttl_seconds

    def _update_memory_cache(self, key: str, entry: Dict[str, Any]) -> None:
        self.memory_cache[key] = entry
        self.memory_cache.move_to_end(key)

        if len(self.memory_cache) > self.config.max_memory_items:
            self.memory_cache.popitem(last=False)

    def _get_inflight_lock(self, key: str) -> Lock:
        with self._lock:
            if key not in self._inflight:
                self._inflight[key] = Lock()
            return self._inflight[key]

    def get(self, data: Any) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached result if available.
        """

        key = self._hash_input(data)

        with self._lock:

            if self.config.enable_memory_cache:
                entry = self.memory_cache.get(key)

                if entry:
                    if not self._is_expired(entry["timestamp"]):
                        self.memory_cache.move_to_end(key)
                        logger.debug("Memory cache hit")
                        return entry["value"]

                    del self.memory_cache[key]

            if self.config.enable_disk_cache:
                path = self._cache_path(key)

                if path.exists():
                    try:
                        with open(path, "r", encoding="utf-8") as f:
                            entry = json.load(f)

                        if "timestamp" not in entry or "value" not in entry:
                            raise ValueError("Invalid cache entry")

                        if not self._is_expired(entry["timestamp"]):
                            entry["timestamp"] = time.monotonic()
                            logger.debug("Disk cache hit")

                            if self.config.enable_memory_cache:
                                self._update_memory_cache(key, entry)

                            return entry["value"]

                        path.unlink(missing_ok=True)

                    except Exception as exc:
                        logger.warning("Corrupt cache removed: %s", exc)
                        path.unlink(missing_ok=True)

        lock = self._get_inflight_lock(key)
        with lock:
            with self._lock:
                if self.config.enable_memory_cache:
                    entry = self.memory_cache.get(key)
                    if entry and not self._is_expired(entry["timestamp"]):
                        self.memory_cache.move_to_end(key)
                        return entry["value"]

        return None

    def set(self, data: Any, value: Dict[str, Any]) -> None:
        """
        Store inference result in cache.
        """

        key = self._hash_input(data)

        entry = {
            "timestamp": time.monotonic(),
            "value": value,
        }

        with self._lock:

            if self.config.enable_memory_cache:
                self._update_memory_cache(key, entry)

            if self.config.enable_disk_cache:
                path = self._cache_path(key)

                try:
                    payload = json.dumps(entry, sort_keys=True, separators=(",", ":"))
                    self._safe_write(path, payload)

                    if self.config.max_disk_items:
                        files = sorted(
                            self.cache_dir.glob("*.json"),
                            key=os.path.getmtime,
                        )
                        while len(files) > self.config.max_disk_items:
                            files.pop(0).unlink(missing_ok=True)
                except Exception as exc:
                    logger.warning("Failed to write disk cache: %s", exc)

    def invalidate(self, data: Any) -> None:
        """
        Remove cached entry.
        """

        key = self._hash_input(data)

        with self._lock:
            self.memory_cache.pop(key, None)

            if self.config.enable_disk_cache:
                path = self._cache_path(key)
                if path.exists():
                    path.unlink(missing_ok=True)

        logger.debug("Cache invalidated")

    def clear(self) -> None:
        """
        Clear entire cache.
        """

        with self._lock:
            self.memory_cache.clear()

            if self.config.enable_disk_cache:
                for file in self.cache_dir.glob("*.json"):
                    file.unlink(missing_ok=True)

        logger.info("Cache cleared")