"""
File Name: cache_manager.py
Module: Feature Engineering - Cache Manager
Description:
    Centralized cache management layer used by the TruthLens feature
    engineering system. This module coordinates multiple cache backends,
    provides cache invalidation policies, and exposes a unified API for
    feature pipelines and preprocessing jobs.

    The CacheManager wraps FeatureCache instances and provides:

        • unified caching interface
        • namespace-aware cache separation
        • cache versioning
        • memory + disk cache coordination
        • safe concurrent access patterns
        • configurable eviction policies

    Designed for large-scale ML preprocessing pipelines where repeated
    feature computation must be avoided.

Dependencies:
    dataclasses
    typing
    logging
    pathlib
    threading

Inputs:
    FeatureContext
    Feature computation functions

Outputs:
    Cached or computed feature vectors
"""

from __future__ import annotations

import logging
import threading
from multiprocessing import Manager
from multiprocessing.managers import SyncManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from src.features.base.base_feature import FeatureContext
from src.features.cache.feature_cache import FeatureCache

logger = logging.getLogger(__name__)

FeatureVector = Dict[str, float]


@dataclass
class CacheNamespace:
    """
    Logical namespace for grouping cache entries.
    """

    name: str
    cache: FeatureCache


@dataclass
class CacheManager:
    """
    Manages multiple feature cache namespaces and coordinates
    cache access across the feature pipeline.
    """

    base_cache_dir: Optional[Path] = None
    enable_memory_cache: bool = True
    max_memory_items: int = 10000
    use_multiprocess_memory_cache: bool = False

    namespaces: Dict[str, CacheNamespace] = field(default_factory=dict)
    _memory_cache: Dict[str, FeatureVector] = field(default_factory=dict, init=False)
    _memory_size: int = field(default=0, init=False)
    _memory_manager: Optional[SyncManager] = field(default=None, init=False, repr=False)

    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def __post_init__(self) -> None:
        if self.enable_memory_cache and self.use_multiprocess_memory_cache:
            self._memory_manager = Manager()
            self._memory_cache = self._memory_manager.dict()  # type: ignore[assignment]

    def _namespace_path(self, namespace: str) -> Path:
        base = self.base_cache_dir if self.base_cache_dir else Path("cache")
        return base / namespace

    def register_namespace(self, namespace: str) -> None:
        """
        Register a new cache namespace.
        """

        with self._lock:

            if namespace in self.namespaces:
                logger.debug("Cache namespace already exists: %s", namespace)
                return

            cache_dir = self._namespace_path(namespace)

            cache = FeatureCache(cache_dir=cache_dir)

            self.namespaces[namespace] = CacheNamespace(
                name=namespace,
                cache=cache,
            )

            logger.info("Cache namespace registered: %s", namespace)

    def get_cache(self, namespace: str) -> FeatureCache:
        """
        Retrieve cache instance for namespace.
        """

        if namespace not in self.namespaces:
            with self._lock:
                if namespace not in self.namespaces:
                    cache_dir = self._namespace_path(namespace)
                    cache = FeatureCache(cache_dir=cache_dir)
                    self.namespaces[namespace] = CacheNamespace(
                        name=namespace,
                        cache=cache,
                    )
                    logger.info("Cache namespace registered: %s", namespace)

        return self.namespaces[namespace].cache

    def _context_key(self, context: FeatureContext) -> str:
        """
        Derive a stable cache key from a FeatureContext.
        """
        return context.text or ""

    def _memory_key(self, namespace: str, key: str) -> str:
        return f"{namespace}:{key}"

    def get_or_compute(
        self,
        namespace: str,
        context: FeatureContext,
        compute_fn: Callable[[FeatureContext], FeatureVector],
    ) -> FeatureVector:
        """
        Retrieve cached features or compute them using provided function.
        """

        cache = self.get_cache(namespace)
        cache_load = cache.load
        cache_save = cache.save
        key = self._context_key(context)
        memory_key = self._memory_key(namespace, key)
        memory_cache = self._memory_cache
        enable_mem = self.enable_memory_cache
        can_store = enable_mem and self._memory_size < self.max_memory_items

        try:
            if enable_mem:
                cached = memory_cache.get(memory_key)
                if cached is not None:
                    return cached

            cached = cache_load(key)
            if cached is not None:
                logger.debug("Cache hit | namespace=%s", namespace)
                if can_store:
                    if memory_key not in memory_cache:
                        memory_cache[memory_key] = cached
                        self._memory_size += 1
                return cached

            result = compute_fn(context)
            cache_save(key, result)

            if can_store:
                if memory_key not in memory_cache:
                    memory_cache[memory_key] = result
                    self._memory_size += 1

            return result

        except Exception as exc:  # noqa: BLE001
            logger.exception("Cache retrieval failed in namespace '%s'", namespace)
            raise RuntimeError("CacheManager get_or_compute failure") from exc

    def get_or_compute_batch(
        self,
        namespace: str,
        contexts: List[FeatureContext],
        compute_batch_fn: Callable[[List[FeatureContext]], List[FeatureVector]],
    ) -> List[FeatureVector]:
        """
        Retrieve cached features or compute missing features in batch.
        """

        if not contexts:
            return []

        cache = self.get_cache(namespace)
        cache_load = cache.load
        cache_save = cache.save
        memory_cache = self._memory_cache
        enable_mem = self.enable_memory_cache
        can_store = enable_mem and self._memory_size < self.max_memory_items
        remaining_slots = self.max_memory_items - self._memory_size if can_store else 0

        keys = [self._context_key(ctx) for ctx in contexts]
        memory_keys = [f"{namespace}:{key}" for key in keys]

        results: List[Optional[FeatureVector]] = [None] * len(contexts)
        missing_contexts: List[FeatureContext] = []
        missing_indices: List[int] = []

        for index, key in enumerate(keys):
            memory_key = memory_keys[index]

            if enable_mem:
                cached = memory_cache.get(memory_key)
                if cached is not None:
                    results[index] = cached
                    continue

            cached = cache_load(key)
            if cached is not None:
                results[index] = cached
                if remaining_slots > 0:
                    if memory_key not in memory_cache:
                        memory_cache[memory_key] = cached
                        self._memory_size += 1
                        remaining_slots -= 1
            else:
                missing_contexts.append(contexts[index])
                missing_indices.append(index)

        if missing_contexts:
            new_results = compute_batch_fn(missing_contexts)

            if len(new_results) != len(missing_contexts):
                raise RuntimeError(
                    "compute_batch_fn must return one result per missing context"
                )

            for index, key, feature_vector in zip(
                missing_indices,
                [keys[i] for i in missing_indices],
                new_results,
            ):
                memory_key = memory_keys[index]

                results[index] = feature_vector
                cache_save(key, feature_vector)

                if remaining_slots > 0:
                    if memory_key not in memory_cache:
                        memory_cache[memory_key] = feature_vector
                        self._memory_size += 1
                        remaining_slots -= 1

        return results

    def clear_namespace(self, namespace: str) -> None:
        """
        Clear the disk cache for a namespace.
        """

        if namespace not in self.namespaces:
            return

        cache = self.namespaces[namespace].cache

        cache.clear()

        prefix = f"{namespace}:"
        keys_to_remove = [
            key for key in self._memory_cache.keys() if key.startswith(prefix)
        ]
        for key in keys_to_remove:
            del self._memory_cache[key]

        if keys_to_remove:
            self._memory_size = max(0, self._memory_size - len(keys_to_remove))

        logger.info("Cache namespace cleared: %s", namespace)

    def clear_all(self) -> None:
        """
        Clear all namespaces.
        """

        for namespace in list(self.namespaces.keys()):
            self.clear_namespace(namespace)

        self._memory_cache.clear()
        self._memory_size = 0

        logger.info("All cache namespaces cleared")

    def list_namespaces(self) -> list[str]:
        """
        List registered cache namespaces.
        """

        return list(self.namespaces.keys())