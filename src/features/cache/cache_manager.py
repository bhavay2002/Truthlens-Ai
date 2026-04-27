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
from src.features.cache.feature_cache import FeatureCache, CACHE_VERSION

logger = logging.getLogger(__name__)

FeatureVector = Dict[str, float]
# CACHE_VERSION is re-exported from feature_cache to avoid the previous
# split-brain (two constants that had to be edited in lock-step).


# =========================================================
# LRU MEMORY CACHE
# =========================================================

class LRUCache:
    """Bounded in-process feature-vector LRU.

    Audit fix §7.6 — the previous implementation only enforced an
    item-count budget (``max_items``). For wide feature vectors (~6k
    keys at ~80 bytes each ≈ 500KB per entry) ``max_items=10_000``
    silently consumed several GB of RSS before the count cap kicked in.
    We now also track an estimated *byte* budget per entry and evict
    LRU until the global byte total is under ``max_bytes``.
    """

    # Approximate per-entry overhead: dict header + each (key, float)
    # pair. Numbers are conservative for CPython 3.11+; the goal is a
    # stable order-of-magnitude estimate, not a precise heap accountant.
    _DICT_OVERHEAD_BYTES = 232
    _PER_ENTRY_BYTES = 80  # str interned ptr + float64 value + slot

    def __init__(
        self,
        max_items: int,
        max_bytes: Optional[int] = None,
    ):
        self.max_items = max_items
        # ``None`` disables the byte budget and preserves the previous
        # count-only semantics for callers that opt out explicitly.
        self.max_bytes = max_bytes

        self.store: OrderedDict[str, FeatureVector] = OrderedDict()
        self._sizes: Dict[str, int] = {}
        self._total_bytes: int = 0

    @classmethod
    def _estimate_bytes(cls, value: FeatureVector) -> int:
        if not isinstance(value, dict):
            return cls._DICT_OVERHEAD_BYTES
        return cls._DICT_OVERHEAD_BYTES + len(value) * cls._PER_ENTRY_BYTES

    def get(self, key: str) -> Optional[FeatureVector]:
        if key not in self.store:
            return None
        self.store.move_to_end(key)
        # Return a shallow copy so downstream pruning / scaling cannot
        # corrupt the cached object (FeatureVector values are floats so
        # a shallow copy is sufficient).
        return dict(self.store[key])

    def set(self, key: str, value: FeatureVector) -> None:
        # Store a copy so subsequent caller mutation does not propagate
        # back into the cache.
        if key in self.store:
            self._total_bytes -= self._sizes.pop(key, 0)
            del self.store[key]

        copied = dict(value)
        size = self._estimate_bytes(copied)
        self.store[key] = copied
        self._sizes[key] = size
        self._total_bytes += size
        self.store.move_to_end(key)

        # Evict by item count first (cheap), then by byte budget. The
        # two caps interact safely because byte eviction also drops
        # items so the count cap is implicitly respected.
        while len(self.store) > self.max_items:
            self._evict_oldest()
        if self.max_bytes is not None:
            while self._total_bytes > self.max_bytes and self.store:
                self._evict_oldest()

    def _evict_oldest(self) -> None:
        old_key, _ = self.store.popitem(last=False)
        self._total_bytes -= self._sizes.pop(old_key, 0)

    @property
    def total_bytes(self) -> int:
        return self._total_bytes


# =========================================================
# CACHE MANAGER
# =========================================================

@dataclass
class CacheManager:

    base_cache_dir: Optional[Path] = None
    max_memory_items: int = 10000
    # Audit fix §7.6 — global byte budget for the in-process LRU.
    # ``None`` preserves the previous count-only behaviour. Default
    # 512MB matches the host RAM headroom we leave for the rest of the
    # inference pipeline; callers running on smaller workers should
    # set this explicitly.
    max_memory_bytes: Optional[int] = 512 * 1024 * 1024

    namespaces: Dict[str, FeatureCache] = field(default_factory=dict)
    _memory_cache: LRUCache = field(init=False)

    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    # Lazily-computed fingerprint of the registered feature set; included
    # in cache keys so enable/disable of features auto-invalidates.
    feature_set_fingerprint: Optional[str] = field(default=None, init=False)

    # -----------------------------------------------------

    def __post_init__(self):
        self._memory_cache = LRUCache(
            self.max_memory_items, max_bytes=self.max_memory_bytes
        )

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
    # PRUNE  (audit fix #1.5)
    #
    # Sweep every namespace under base_cache_dir, plus any namespaces
    # registered in this process, applying the same TTL + byte-budget
    # eviction.  Safe to invoke at process start: missing dirs and
    # transient OS errors are logged and skipped, never raised.
    # -----------------------------------------------------

    def prune_all(
        self,
        *,
        max_bytes_per_namespace: Optional[int] = None,
        max_age_days: Optional[float] = None,
    ) -> Dict[str, Dict[str, int]]:

        results: Dict[str, Dict[str, int]] = {}
        seen: set[Path] = set()

        # In-process namespaces first (they own the live LRU memo we
        # need to invalidate on file deletion).
        for ns, cache in list(self.namespaces.items()):
            try:
                results[ns] = cache.prune(
                    max_bytes=max_bytes_per_namespace,
                    max_age_days=max_age_days,
                )
                seen.add(cache.cache_dir.resolve())
            except Exception as exc:
                logger.warning("Prune failed for namespace %s: %s", ns, exc)

        # Plus any namespaces persisted on disk from a previous run that
        # have not yet been registered this process.
        base = self.base_cache_dir or Path("cache")
        if base.exists():
            for child in base.iterdir():
                if not child.is_dir():
                    continue
                if child.resolve() in seen:
                    continue
                try:
                    cache = FeatureCache(child)
                    results[child.name] = cache.prune(
                        max_bytes=max_bytes_per_namespace,
                        max_age_days=max_age_days,
                    )
                except Exception as exc:
                    logger.warning("Prune failed for dir %s: %s", child, exc)

        return results

    # -----------------------------------------------------
    # VERSIONED KEY (CRITICAL)
    #
    # The key includes a *feature-set fingerprint* derived from the
    # currently-registered feature names.  This means: enabling /
    # disabling features automatically invalidates stale cache entries
    # without requiring CACHE_VERSION to be bumped manually.
    # -----------------------------------------------------

    @staticmethod
    def _compute_feature_set_fingerprint() -> str:
        try:
            from src.features.base.feature_registry import FeatureRegistry
            names = sorted(FeatureRegistry.list_features())
        except Exception:
            names = []
        if not names:
            return "no-registry"
        return hashlib.sha256("|".join(names).encode()).hexdigest()[:16]

    def _get_feature_fingerprint(self) -> str:
        if self.feature_set_fingerprint is None:
            self.feature_set_fingerprint = self._compute_feature_set_fingerprint()
        return self.feature_set_fingerprint

    # -----------------------------------------------------
    # LEXICON FINGERPRINT
    #
    # SHA over the *contents* of every loaded lexicon source file. If
    # any lexicon (bias, emotion, propaganda, …) changes on disk, the
    # fingerprint changes and stale cache entries are auto-invalidated
    # without requiring CACHE_VERSION to be bumped manually.
    # -----------------------------------------------------

    # Source files holding the in-process lexicons. Resolved once at
    # class-load time and hashed lazily on first cache key computation.
    _LEXICON_SOURCES: tuple = (
        "src/features/bias/bias_lexicon.py",
        "src/features/bias/bias_lexicon_features.py",
        "src/features/bias/bias_features.py",
        "src/features/emotion/emotion_lexicon.py",
        "src/features/emotion/emotion_lexicon_features.py",
        "src/features/emotion/emotion_features.py",
        "src/features/propaganda/propaganda_lexicon_features.py",
        "src/features/propaganda/propaganda_features.py",
    )

    lexicon_fingerprint: Optional[str] = field(default=None, init=False)

    @classmethod
    def _compute_lexicon_fingerprint(cls) -> str:
        # Resolve relative paths against the project root (two levels up
        # from this file: src/features/cache/cache_manager.py).
        project_root = Path(__file__).resolve().parents[3]
        h = hashlib.sha256()
        h.update(b"lexicons-v1\n")
        for rel in cls._LEXICON_SOURCES:
            p = project_root / rel
            if not p.is_file():
                # Missing file is itself a meaningful signal — record
                # the path so adding/removing a lexicon invalidates.
                h.update(f"missing:{rel}\n".encode())
                continue
            try:
                h.update(rel.encode())
                h.update(b"\0")
                h.update(p.read_bytes())
                h.update(b"\n")
            except OSError as exc:
                logger.warning("Lexicon fingerprint read failed (%s): %s", rel, exc)
                h.update(f"unreadable:{rel}\n".encode())
        return h.hexdigest()[:16]

    def _get_lexicon_fingerprint(self) -> str:
        if self.lexicon_fingerprint is None:
            self.lexicon_fingerprint = self._compute_lexicon_fingerprint()
        return self.lexicon_fingerprint

    def _context_key(self, context: FeatureContext) -> str:

        # Pull tokenizer_id out of metadata (if any) so switching
        # roberta-base ↔ xlm-roberta-base or upgrading the tokenizer
        # auto-invalidates without leaking BPE-aligned features into a
        # different model head.  Audit fix #1.6.
        meta = dict(context.metadata or {})
        tokenizer_id = meta.pop("tokenizer_id", None)

        payload = {
            "version": CACHE_VERSION,
            "feature_set": self._get_feature_fingerprint(),
            "lexicons": self._get_lexicon_fingerprint(),
            "tokenizer_id": tokenizer_id,
            "text": context.text,
            "tokens": context.tokens,
            "metadata": meta,
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