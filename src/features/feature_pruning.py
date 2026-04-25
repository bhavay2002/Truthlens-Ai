from __future__ import annotations

"""
Feature Pruning Module

Performs automatic feature reduction:
- constant feature removal
- low variance filtering
- high correlation filtering

Designed for:
- ML preprocessing
- feature optimization
- training stability
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set

import numpy as np

logger = logging.getLogger(__name__)

FeatureVector = Dict[str, float]


# =========================================================
# UTILS
# =========================================================

def _dict_to_matrix(features: List[FeatureVector]) -> Tuple[np.ndarray, List[str]]:
    if not features:
        raise ValueError("Feature list cannot be empty")

    keys = sorted(features[0].keys())

    X = np.array(
        [[f.get(k, 0.0) for k in keys] for f in features],
        dtype=np.float32,
    )

    return X, keys


def _matrix_to_dicts(X: np.ndarray, keys: List[str]) -> List[FeatureVector]:
    return [
        {k: float(v) for k, v in zip(keys, row)}
        for row in X
    ]


# =========================================================
# PRUNER
# =========================================================

@dataclass
class FeaturePruner:
    """
    Automated feature pruning pipeline.
    """

    variance_threshold: float = 1e-6
    correlation_threshold: float = 0.95

    removed_features_: Set[str] = field(default_factory=set, init=False)
    kept_features_: List[str] = field(default_factory=list, init=False)

    # -----------------------------------------------------

    def fit(self, features: List[FeatureVector]) -> None:
        """
        Learn which features to keep.
        """

        X, keys = _dict_to_matrix(features)

        logger.info("Starting feature pruning | features=%d", len(keys))

        # -------------------------------------------------
        # 1. REMOVE CONSTANT / LOW VARIANCE
        # -------------------------------------------------

        variances = np.var(X, axis=0)

        keep_mask = variances > self.variance_threshold

        removed_low_variance = [
            keys[i] for i, keep in enumerate(keep_mask) if not keep
        ]

        X = X[:, keep_mask]
        keys = [k for i, k in enumerate(keys) if keep_mask[i]]

        logger.info("Removed low variance features: %d", len(removed_low_variance))

        # -------------------------------------------------
        # 2. REMOVE HIGHLY CORRELATED FEATURES
        # -------------------------------------------------

        if X.shape[1] > 1:

            corr_matrix = np.corrcoef(X, rowvar=False)

            to_remove = set()

            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):

                    if abs(corr_matrix[i, j]) > self.correlation_threshold:
                        # remove second feature
                        to_remove.add(keys[j])

            keep_indices = [i for i, k in enumerate(keys) if k not in to_remove]

            removed_corr = list(to_remove)

            X = X[:, keep_indices]
            keys = [keys[i] for i in keep_indices]

            logger.info("Removed correlated features: %d", len(removed_corr))

        else:
            removed_corr = []

        # -------------------------------------------------
        # FINAL
        # -------------------------------------------------

        self.kept_features_ = keys
        self.removed_features_ = set(removed_low_variance + removed_corr)

        logger.info(
            "Feature pruning complete | kept=%d removed=%d",
            len(self.kept_features_),
            len(self.removed_features_),
        )

    # -----------------------------------------------------

    def transform(self, features: List[FeatureVector]) -> List[FeatureVector]:
        """
        Apply pruning.
        """

        if not self.kept_features_:
            raise RuntimeError("FeaturePruner must be fitted first")

        pruned = []

        for f in features:
            pruned.append({
                k: float(f.get(k, 0.0))
                for k in self.kept_features_
            })

        return pruned

    # -----------------------------------------------------

    def fit_transform(
        self,
        features: List[FeatureVector],
    ) -> List[FeatureVector]:
        self.fit(features)
        return self.transform(features)

    # -----------------------------------------------------

    def get_removed_features(self) -> List[str]:
        return sorted(self.removed_features_)

    def get_kept_features(self) -> List[str]:
        return self.kept_features_