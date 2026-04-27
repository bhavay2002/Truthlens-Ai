from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List

import numpy as np

logger = logging.getLogger(__name__)

EPS = 1e-8


# =========================================================
# CONTEXT
# =========================================================

@dataclass
class FeatureContext:
    """
    Context object passed to all feature extractors.

    Supports:
    - raw text
    - optional tokens
    - embeddings
    - per-sample cache
    - shared batch cache (NEW)
    """

    text: str
    metadata: Optional[Dict[str, Any]] = None

    # -----------------------------------
    # NLP / Preprocessing
    # -----------------------------------
    tokens: Optional[List[str]] = None
    embeddings: Optional[Any] = None  # backward compatibility

    # -----------------------------------
    # CACHES
    # -----------------------------------
    cache: Dict[str, Any] = field(default_factory=dict)

    # 🔥 NEW: shared cache across batch (critical for performance)
    shared: Optional[Dict[str, Any]] = None

    # -----------------------------------------------------
    # CACHE HELPERS
    # -----------------------------------------------------

    def get_cache(self, key: str) -> Any:
        return self.cache.get(key)

    def set_cache(self, key: str, value: Any) -> None:
        self.cache[key] = value

    # -----------------------------------------------------
    # SHARED CACHE HELPERS (NEW)
    # -----------------------------------------------------

    def get_shared(self, key: str) -> Any:
        if self.shared is None:
            return None
        return self.shared.get(key)

    def set_shared(self, key: str, value: Any) -> None:
        if self.shared is None:
            self.shared = {}
        self.shared[key] = value


# =========================================================
# BASE FEATURE
# =========================================================

@dataclass
class BaseFeature:
    """
    Abstract base class for all feature extractors.
    """

    name: str
    group: str = "general"

    version: str = "1.0"
    description: Optional[str] = None

    enabled: bool = True
    fail_silent: bool = True

    _initialized: bool = field(default=False, init=False)

    # -----------------------------------------------------

    def __post_init__(self):

        if not self.name:
            raise ValueError("Feature must have a name")

        logger.debug(
            "Initialized feature | %s (%s)",
            self.name,
            self.group,
        )

    # =====================================================
    # CORE
    # =====================================================

    def extract(self, context: FeatureContext) -> Dict[str, Any]:
        """
        Override in subclasses.
        """
        raise NotImplementedError

    # -----------------------------------------------------
    # BATCH SUPPORT
    # -----------------------------------------------------

    def extract_batch(
        self,
        contexts: List[FeatureContext],
    ) -> List[Dict[str, Any]]:
        """
        Default batch implementation. Lexicon and other CPU-bound extractors
        should override this to amortize per-batch setup (precompiled
        regexes, shared lookup tables, etc.).
        """
        return [self.extract(ctx) for ctx in contexts]

    # =====================================================
    # SAFE EXECUTION
    # =====================================================

    def safe_extract(self, context: FeatureContext) -> Dict[str, Any]:

        if not self.enabled:
            return {}

        if not isinstance(context.text, str):
            raise TypeError("context.text must be string")

        # Initialize once
        if not self._initialized:
            self.initialize()
            self._initialized = True

        start = time.time()

        try:
            features = self.extract(context)

            self._validate_output(features)

            duration = time.time() - start

            logger.debug(
                "Feature '%s' extracted %d values in %.4fs",
                self.name,
                len(features),
                duration,
            )

            return features

        except Exception as e:

            logger.exception("Feature failed: %s", self.name)

            if self.fail_silent:
                return self._fallback()

            raise RuntimeError(f"Feature failed: {self.name}") from e

    # -----------------------------------------------------

    def safe_extract_batch(
        self,
        contexts: List[FeatureContext],
    ) -> List[Dict[str, Any]]:
        """
        Batch counterpart of `safe_extract`.

        Calls the extractor's `extract_batch` (which subclasses may override
        for vectorized work), then validates each output and applies the
        same `fail_silent` policy on a per-sample basis.
        """

        n = len(contexts)

        if not self.enabled or n == 0:
            return [{} for _ in range(n)]

        for ctx in contexts:
            if not isinstance(ctx.text, str):
                raise TypeError("context.text must be string")

        if not self._initialized:
            self.initialize()
            self._initialized = True

        start = time.time()

        try:
            results = self.extract_batch(contexts)
        except Exception as e:
            logger.exception("Feature batch failed: %s", self.name)
            if self.fail_silent:
                return [self._fallback() for _ in range(n)]
            raise RuntimeError(f"Feature batch failed: {self.name}") from e

        # Per-sample validation; on failure, fall back rather than killing the batch.
        validated: List[Dict[str, Any]] = []
        for idx, features in enumerate(results):
            try:
                if not isinstance(features, dict):
                    raise ValueError(f"{self.name} batch[{idx}] must return dict")
                self._validate_output(features)
                validated.append(features)
            except Exception:
                logger.exception(
                    "Feature batch validation failed: %s [idx=%d]", self.name, idx
                )
                if self.fail_silent:
                    validated.append(self._fallback())
                else:
                    raise

        # Length contract — fusion relies on alignment with `contexts`.
        if len(validated) < n:
            validated.extend(self._fallback() for _ in range(n - len(validated)))
        elif len(validated) > n:
            validated = validated[:n]

        logger.debug(
            "Feature '%s' batch=%d in %.4fs",
            self.name, n, time.time() - start,
        )
        return validated

    # =====================================================
    # VALIDATION
    # =====================================================

    def _validate_output(self, features: Dict[str, Any]) -> None:

        if not isinstance(features, dict):
            raise ValueError(f"{self.name} must return dict")

        for k, v in features.items():

            if not isinstance(k, str):
                raise ValueError("Feature keys must be strings")

            if isinstance(v, (int, float)):

                if not np.isfinite(v):
                    features[k] = 0.0

            else:
                raise ValueError(f"Feature '{k}' must be numeric")

    # =====================================================
    # FALLBACK
    # =====================================================

    def _fallback(self) -> Dict[str, float]:
        return {}

    # =====================================================
    # METADATA
    # =====================================================

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "group": self.group,
            "version": self.version,
            "enabled": self.enabled,
            "class": self.__class__.__name__,
        }

    # =====================================================
    # LIFECYCLE
    # =====================================================

    def initialize(self) -> None:
        logger.debug("Initializing %s", self.name)

    def teardown(self) -> None:
        logger.debug("Tearing down %s", self.name)