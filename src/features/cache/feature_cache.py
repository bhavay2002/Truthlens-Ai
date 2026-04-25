# src/features/cache/feature_cache.py

from __future__ import annotations

import hashlib
import json
import logging
import threading
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import gzip

logger = logging.getLogger(__name__)


# =========================================================
# CONFIG
# =========================================================

CACHE_VERSION = "v2"
USE_COMPRESSION = True


# =========================================================
# CACHE
# =========================================================

class FeatureCache:

    def __init__(self, cache_dir: str | Path = "cache") -> None:
        self.cache_dir = Path(cache_dir)
        self._path_cache: Dict[str, Path] = {}
        self._lock = threading.Lock()

        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------

    def _get_path(self, key: str) -> Path:

        with self._lock:
            if key in self._path_cache:
                return self._path_cache[key]

            digest = hashlib.sha256(key.encode()).hexdigest()
            filename = f"{digest}.json.gz" if USE_COMPRESSION else f"{digest}.json"

            path = self.cache_dir / filename
            self._path_cache[key] = path
            return path

    # -----------------------------------------------------
    # SAFE SERIALIZATION
    # -----------------------------------------------------

    def _serialize(self, data: Any) -> bytes:

        payload = {
            "version": CACHE_VERSION,
            "data": data,
        }

        raw = json.dumps(payload, separators=(",", ":"), default=str).encode()

        if USE_COMPRESSION:
            return gzip.compress(raw)

        return raw

    def _deserialize(self, raw: bytes) -> Any:

        if USE_COMPRESSION:
            raw = gzip.decompress(raw)

        payload = json.loads(raw.decode())

        if payload.get("version") != CACHE_VERSION:
            logger.warning("Cache version mismatch")
            return None

        return payload.get("data")

    # -----------------------------------------------------
    # ATOMIC WRITE (CRITICAL)
    # -----------------------------------------------------

    def save(self, key: str, data: Any) -> Path:

        path = self._get_path(key)

        try:
            serialized = self._serialize(data)

            with tempfile.NamedTemporaryFile(
                delete=False,
                dir=self.cache_dir,
            ) as tmp:

                tmp.write(serialized)
                tmp.flush()

                temp_path = Path(tmp.name)

            temp_path.replace(path)

            return path

        except Exception:
            logger.exception("Cache save failed")
            raise

    # -----------------------------------------------------
    # LOAD
    # -----------------------------------------------------

    def load(self, key: str) -> Optional[Any]:

        path = self._get_path(key)

        if not path.exists():
            return None

        try:
            raw = path.read_bytes()
            return self._deserialize(raw)

        except Exception:
            logger.exception("Cache load failed → deleting corrupted file")
            path.unlink(missing_ok=True)
            return None

    # -----------------------------------------------------
    # BATCH LOAD
    # -----------------------------------------------------

    def load_many(self, keys: List[str]) -> List[Optional[Any]]:
        return [self.load(k) for k in keys]

    # -----------------------------------------------------

    def exists(self, key: str) -> bool:
        return self._get_path(key).exists()

    # -----------------------------------------------------

    def clear(self) -> None:
        for file in self.cache_dir.glob("*"):
            file.unlink(missing_ok=True)

        self._path_cache.clear()
        logger.info("Cache cleared: %s", self.cache_dir)