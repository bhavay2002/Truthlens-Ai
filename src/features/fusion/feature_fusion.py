from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from src.features.base.base_feature import BaseFeature, FeatureContext
from src.features.fusion.feature_merger import merge_features  # 🔥 NEW

logger = logging.getLogger(__name__)

EPS = 1e-8


# =========================================================
# FEATURE FUSION
# =========================================================

@dataclass
class FeatureFusion:

    features: List[BaseFeature] = field(default_factory=list)
    enforce_unique_names: bool = True

    # Per-sample z-score across feature TYPES is statistically invalid:
    # it mixes units (ratios, counts, densities, embeddings) within one
    # row.  Population-level scaling MUST be applied via FeatureScaling
    # using a scaler fitted on the training set.  Default: OFF.
    normalize: bool = False
    return_vector: bool = False

    _feature_order: List[str] = field(default_factory=list, init=False)

    # -----------------------------------------------------

    def _validate_feature_names(self) -> None:
        names = [f.name for f in self.features]

        if len(names) != len(set(names)):
            counts = Counter(names)
            duplicates = {name for name, cnt in counts.items() if cnt > 1}
            raise ValueError(f"Duplicate feature extractors detected: {duplicates}")

    # -----------------------------------------------------

    def _ensure_initialized(self) -> None:
        if not hasattr(self, "_initialized"):
            for feature in self.features:
                feature.initialize()
            self._initialized = True

    # -----------------------------------------------------

    def _ensure_validated(self) -> None:
        if self.enforce_unique_names and not hasattr(self, "_validated"):
            self._validate_feature_names()
            self._validated = True

    # -----------------------------------------------------
    # NORMALIZATION
    # -----------------------------------------------------

    def _normalize(self, features: Dict[str, float]) -> Dict[str, float]:

        if not features:
            return features

        values = np.array(list(features.values()), dtype=np.float32)

        mean = values.mean()
        std = values.std()

        if std < EPS:
            return features

        norm_values = (values - mean) / (std + EPS)

        return dict(zip(features.keys(), norm_values.astype(float)))

    # -----------------------------------------------------
    # CORE EXTRACTION
    # -----------------------------------------------------

    def extract(self, context: FeatureContext):

        self._ensure_validated()
        self._ensure_initialized()

        # -------------------------------------------------
        #  COLLECT ALL FEATURE OUTPUTS
        # -------------------------------------------------

        outputs: List[Dict[str, float]] = []

        for feature in self.features:

            try:
                output = feature.safe_extract(context)
            except Exception:
                logger.exception("Feature failed: %s", feature.name)
                continue

            if isinstance(output, dict) and output:
                outputs.append(output)

        # -------------------------------------------------
        #  MERGE USING CENTRAL LOGIC
        # -------------------------------------------------

        fused: Dict[str, float] = merge_features(outputs)

        # -------------------------------------------------
        # NORMALIZATION
        # -------------------------------------------------

        if self.normalize:
            fused = self._normalize(fused)

        # -------------------------------------------------
        # FREEZE ORDER (CRITICAL)
        # -------------------------------------------------

        if not self._feature_order:
            self._feature_order = sorted(fused.keys())

        # -------------------------------------------------
        # VECTOR OUTPUT
        # -------------------------------------------------

        if self.return_vector:
            vector = np.array(
                [fused.get(k, 0.0) for k in self._feature_order],
                dtype=np.float32,
            )
            return vector

        return fused

    # -----------------------------------------------------
    # BATCH
    # -----------------------------------------------------

    def extract_batch(self, contexts: List[FeatureContext]):
        return [self.extract(c) for c in contexts]

    # -----------------------------------------------------

    def get_feature_order(self) -> List[str]:
        return self._feature_order

    # -----------------------------------------------------

    def get_feature_dim(self) -> int:
        return len(self._feature_order)