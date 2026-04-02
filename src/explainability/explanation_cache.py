"""
File Name: explanation_cache.py
Module: Explainability - Caching
Description:
    Provides a caching layer for explainability outputs in the TruthLens AI
    system. SHAP and LIME explanations are computationally expensive; this
    module stores previously computed explanations using a deterministic text
    hash and returns cached results when available.

    Supports:
        • In-memory LRU caching
        • Optional disk persistence
        • Deterministic hashing of text inputs
        • Safe serialization of explanation outputs

Dependencies:
    logging
    hashlib
    json
    pathlib
    typing
    collections

Inputs:
    text
    explanation output

Outputs:
    cached explanation result
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ExplanationCache:
    """
    Cache system for expensive explanation computations.
    """

    def __init__(
        self,
        max_size: int = 128,
        cache_dir: Optional[str | Path] = None,
    ) -> None:
        if max_size <= 0:
            raise ValueError("max_size must be greater than 0")

        self.max_size = max_size
        self.memory_cache: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()

        self.cache_dir: Optional[Path] = None
        if cache_dir is not None:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info("ExplanationCache initialized (max_size=%s)", max_size)

    @staticmethod
    def _hash_text(text: str) -> str:
        """
        Generate deterministic SHA256 hash for text.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _evict_if_needed(self) -> None:
        """
        Enforce LRU eviction policy.
        """
        while len(self.memory_cache) > self.max_size:
            evicted_key, _ = self.memory_cache.popitem(last=False)
            logger.debug("Evicted explanation cache key: %s", evicted_key)

    def _disk_path(self, key: str) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{key}.json"

    def get(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached explanation if available.
        """
        key = self._hash_text(text)

        if key in self.memory_cache:
            logger.debug("Explanation cache hit (memory)")
            self.memory_cache.move_to_end(key)
            return self.memory_cache[key]

        disk_path = self._disk_path(key)

        if disk_path and disk_path.exists():
            try:
                with disk_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)

                self.memory_cache[key] = data
                self.memory_cache.move_to_end(key)
                self._evict_if_needed()

                logger.debug("Explanation cache hit (disk)")
                return data

            except Exception as exc:  # pragma: no cover
                logger.warning("Failed to read cached explanation: %s", exc)

        logger.debug("Explanation cache miss")
        return None

    def set(self, text: str, explanation: Dict[str, Any]) -> None:
        """
        Store explanation in cache.
        """
        if not isinstance(explanation, dict):
            raise TypeError("explanation must be a dictionary")

        key = self._hash_text(text)

        self.memory_cache[key] = explanation
        self.memory_cache.move_to_end(key)

        self._evict_if_needed()

        disk_path = self._disk_path(key)

        if disk_path:
            try:
                with disk_path.open("w", encoding="utf-8") as f:
                    json.dump(explanation, f, ensure_ascii=False, indent=2)

                logger.debug("Explanation stored on disk cache")

            except Exception as exc:  # pragma: no cover
                logger.warning("Failed to write explanation cache: %s", exc)

    def clear_memory(self) -> None:
        """
        Clear in-memory cache.
        """
        self.memory_cache.clear()
        logger.info("Explanation memory cache cleared")

    def clear_disk(self) -> None:
        """
        Remove disk cache files.
        """
        if self.cache_dir is None:
            return

        for file in self.cache_dir.glob("*.json"):
            try:
                file.unlink()
            except Exception as exc:  # pragma: no cover
                logger.warning("Failed to delete cache file %s: %s", file, exc)

        logger.info("Explanation disk cache cleared")