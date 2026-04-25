from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import gzip
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


# =========================================================
# CONFIG
# =========================================================

@dataclass
class InferenceCacheConfig:
    cache_dir: str = "cache"
    enable_disk_cache: bool = True
    ttl_seconds: Optional[int] = None
    enable_memory_cache: bool = True
    max_memory_items: int = 1024
    max_disk_items: Optional[int] = None

    # 🔥 NEW
    cache_version: str = "v1"
    enable_compression: bool = True


# =========================================================
# SERIALIZATION (CRITICAL)
# =========================================================

def _serialize(obj: Any):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "item"):
        return obj.item()
    return obj


def _deserialize(obj: Any):
    return obj


# =========================================================
# MAIN CACHE
# =========================================================

class InferenceCache:

    def __init__(self, config: InferenceCacheConfig):

        self.config = config
        self.memory_cache: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._lock = Lock()
        self._inflight: Dict[str, Lock] = {}

        self.cache_dir = Path(config.cache_dir)

        if config.enable_disk_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"InferenceCache initialized (version={config.cache_version})")

    # =====================================================
    # HASH (UPGRADED 🔥)
    # =====================================================

    def _hash_input(self, data: Any) -> str:

        try:
            def default(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, set):
                    return sorted(obj)
                return repr(obj)

            payload = (
                data
                if isinstance(data, str)
                else json.dumps(data, sort_keys=True, default=default)
            )

            # 🔥 VERSION AWARE HASH
            payload = f"{self.config.cache_version}:{payload}"

            return hashlib.sha256(payload.encode("utf-8")).hexdigest()

        except Exception as exc:
            raise RuntimeError("Cache key generation failed") from exc

    # =====================================================
    # PATH
    # =====================================================

    def _cache_path(self, key: str) -> Path:
        suffix = ".json.gz" if self.config.enable_compression else ".json"
        return self.cache_dir / f"{key}{suffix}"

    # =====================================================
    # IO
    # =====================================================

    def _safe_write(self, path: Path, payload: str):

        tmp = path.with_suffix(".tmp")

        if self.config.enable_compression:
            with gzip.open(tmp, "wt", encoding="utf-8") as f:
                f.write(payload)
        else:
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())

        os.replace(tmp, path)

    def _safe_read(self, path: Path):

        if self.config.enable_compression:
            with gzip.open(path, "rt", encoding="utf-8") as f:
                return json.load(f)
        else:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)

    # =====================================================
    # TTL
    # =====================================================

    def _is_expired(self, ts: float):
        if self.config.ttl_seconds is None:
            return False
        return (time.monotonic() - ts) > self.config.ttl_seconds

    # =====================================================
    # MEMORY CACHE
    # =====================================================

    def _update_memory(self, key, entry):
        self.memory_cache[key] = entry
        self.memory_cache.move_to_end(key)

        if len(self.memory_cache) > self.config.max_memory_items:
            self.memory_cache.popitem(last=False)

    # =====================================================
    # GET
    # =====================================================

    def get(self, data: Any) -> Optional[Dict[str, Any]]:

        key = self._hash_input(data)

        with self._lock:

            # MEMORY
            if self.config.enable_memory_cache:
                entry = self.memory_cache.get(key)
                if entry and not self._is_expired(entry["ts"]):
                    return entry["value"]

            # DISK
            if self.config.enable_disk_cache:
                path = self._cache_path(key)

                if path.exists():
                    try:
                        entry = self._safe_read(path)

                        if not self._is_expired(entry["ts"]):
                            if self.config.enable_memory_cache:
                                self._update_memory(key, entry)
                            return entry["value"]

                        path.unlink(missing_ok=True)

                    except Exception:
                        path.unlink(missing_ok=True)

        return None

    # =====================================================
    # SET
    # =====================================================

    def set(self, data: Any, value: Dict[str, Any]):

        key = self._hash_input(data)

        entry = {
            "ts": time.monotonic(),
            "value": json.loads(json.dumps(value, default=_serialize)),
        }

        with self._lock:

            if self.config.enable_memory_cache:
                self._update_memory(key, entry)

            if self.config.enable_disk_cache:
                path = self._cache_path(key)

                try:
                    payload = json.dumps(entry, separators=(",", ":"))
                    self._safe_write(path, payload)

                except Exception as exc:
                    logger.warning(f"Cache write failed: {exc}")

    # =====================================================
    # INVALIDATE
    # =====================================================

    def invalidate(self, data: Any):

        key = self._hash_input(data)

        with self._lock:
            self.memory_cache.pop(key, None)

            path = self._cache_path(key)
            path.unlink(missing_ok=True)

    # =====================================================
    # CLEAR
    # =====================================================

    def clear(self):

        with self._lock:
            self.memory_cache.clear()

            if self.config.enable_disk_cache:
                for f in self.cache_dir.glob("*"):
                    f.unlink(missing_ok=True)

        logger.info("Cache cleared")