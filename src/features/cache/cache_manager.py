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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional

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

    namespaces: Dict[str, CacheNamespace] = field(default_factory=dict)

    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def _namespace_path(self, namespace: str) -> Optional[Path]:
        if not self.base_cache_dir:
            return None
        return self.base_cache_dir / namespace

    def register_namespace(self, namespace: str) -> None:
        """
        Register a new cache namespace.
        """

        with self._lock:

            if namespace in self.namespaces:
                logger.debug("Cache namespace already exists: %s", namespace)
                return

            cache_dir = self._namespace_path(namespace)

            cache = FeatureCache(
                cache_dir=cache_dir,
                use_memory_cache=self.enable_memory_cache,
                max_memory_items=self.max_memory_items,
            )

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
            self.register_namespace(namespace)

        return self.namespaces[namespace].cache

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

        try:
            result = cache.get_or_compute(context, compute_fn)
            return result

        except Exception as exc:  # noqa: BLE001
            logger.exception("Cache retrieval failed in namespace '%s'", namespace)
            raise RuntimeError("CacheManager get_or_compute failure") from exc

    def clear_namespace(self, namespace: str) -> None:
        """
        Clear both memory and disk cache for a namespace.
        """

        if namespace not in self.namespaces:
            return

        cache = self.namespaces[namespace].cache

        cache.clear_memory()
        cache.clear_disk()

        logger.info("Cache namespace cleared: %s", namespace)

    def clear_all(self) -> None:
        """
        Clear all namespaces.
        """

        for namespace in list(self.namespaces.keys()):
            self.clear_namespace(namespace)

        logger.info("All cache namespaces cleared")

    def list_namespaces(self) -> list[str]:
        """
        List registered cache namespaces.
        """

        return list(self.namespaces.keys())