# src/features/cache/cache_manager.py

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from src.features.base.base_feature import FeatureContext
from src.features.cache.feature_cache import FeatureCache

logger = logging.getLogger(__name__)

FeatureVector = Dict[str, float]
CACHE_VERSION = "v2"  # 🔥 CRITICAL


# =========================================================
# LRU MEMORY CACHE
# =========================================================

class LRUCache:
    def __init__(self, max_items: int):
        self.max_items = max_items
        self.store: OrderedDict[str, FeatureVector] = OrderedDict()

    def get(self, key: str) -> Optional[FeatureVector]:
        if key not in self.store:
            return None
        self.store.move_to_end(key)
        return self.store[key]

    def set(self, key: str, value: FeatureVector) -> None:
        self.store[key] = value
        self.store.move_to_end(key)

        if len(self.store) > self.max_items:
            self.store.popitem(last=False)


# =========================================================
# CACHE MANAGER
# =========================================================

@dataclass
class CacheManager:

    base_cache_dir: Optional[Path] = None
    max_memory_items: int = 10000

    namespaces: Dict[str, FeatureCache] = field(default_factory=dict)
    _memory_cache: LRUCache = field(init=False)

    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    # -----------------------------------------------------

    def __post_init__(self):
        self._memory_cache = LRUCache(self.max_memory_items)

    # -----------------------------------------------------

    def _namespace_path(self, namespace: str) -> Path:
        base = self.base_cache_dir or Path("cache")
        return base / namespace

    # -----------------------------------------------------

    def get_cache(self, namespace: str) -> FeatureCache:

        if namespace not in self.namespaces:
            with self._lock:
                if namespace not in self.namespaces:
                    cache = FeatureCache(self._namespace_path(namespace))
                    self.namespaces[namespace] = cache
                    logger.info("Registered cache namespace: %s", namespace)

        return self.namespaces[namespace]

    # -----------------------------------------------------
    # VERSIONED KEY (CRITICAL)
    # -----------------------------------------------------

    def _context_key(self, context: FeatureContext) -> str:

        payload = {
            "version": CACHE_VERSION,
            "text": context.text,
            "tokens": context.tokens,
            "metadata": context.metadata,
        }

        raw = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()

    # -----------------------------------------------------

    def get_or_compute(
        self,
        namespace: str,
        context: FeatureContext,
        compute_fn: Callable[[FeatureContext], FeatureVector],
    ) -> FeatureVector:

        cache = self.get_cache(namespace)
        key = self._context_key(context)

        # -------------------------
        # MEMORY CACHE
        # -------------------------

        cached = self._memory_cache.get(key)
        if cached is not None:
            return cached

        # -------------------------
        # DISK CACHE
        # -------------------------

        cached = cache.load(key)

        if cached is not None:
            self._memory_cache.set(key, cached)
            return cached

        # -------------------------
        # COMPUTE
        # -------------------------

        result = compute_fn(context)

        # -------------------------
        # SAVE
        # -------------------------

        try:
            cache.save(key, result)
        except Exception:
            logger.warning("Disk cache write failed")

        self._memory_cache.set(key, result)

        return result

    # -----------------------------------------------------
    # BATCH (OPTIMIZED)
    # -----------------------------------------------------

    def get_or_compute_batch(
        self,
        namespace: str,
        contexts: List[FeatureContext],
        compute_batch_fn: Callable[[List[FeatureContext]], List[FeatureVector]],
    ) -> List[FeatureVector]:

        if not contexts:
            return []

        cache = self.get_cache(namespace)

        keys = [self._context_key(c) for c in contexts]

        results: List[Optional[FeatureVector]] = [None] * len(contexts)
        missing: List[FeatureContext] = []
        missing_idx: List[int] = []

        # -------------------------
        # LOOKUP
        # -------------------------

        for i, key in enumerate(keys):

            cached = self._memory_cache.get(key)

            if cached is not None:
                results[i] = cached
                continue

            cached = cache.load(key)

            if cached is not None:
                results[i] = cached
                self._memory_cache.set(key, cached)
            else:
                missing.append(contexts[i])
                missing_idx.append(i)

        # -------------------------
        # COMPUTE MISSING
        # -------------------------

        if missing:

            computed = compute_batch_fn(missing)

            for i, key, val in zip(missing_idx, [keys[j] for j in missing_idx], computed):

                results[i] = val

                try:
                    cache.save(key, val)
                except Exception:
                    logger.warning("Disk write failed")

                self._memory_cache.set(key, val)

        return results  # type: ignore