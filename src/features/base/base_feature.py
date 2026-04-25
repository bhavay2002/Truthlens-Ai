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
        Default batch implementation (can be overridden).
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