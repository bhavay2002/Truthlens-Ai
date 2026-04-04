"""
File Name: feature_selection.py
Module: Feature Engineering - Feature Selection
Description:
    Implements feature selection utilities for the TruthLens feature
    engineering pipeline. The module provides multiple strategies for
    selecting informative features and reducing dimensionality before
    model training.

    Supported methods include:

        • Variance threshold filtering
        • Correlation-based feature removal
        • Mutual information ranking
        • Top-K feature selection

    The implementation works with dictionary-based feature vectors used
    throughout the TruthLens system while internally converting them to
    matrix form for efficient computation.

Dependencies:
    dataclasses
    typing
    logging
    numpy
    sklearn (optional for mutual information)

Inputs:
    List[Dict[str, float]] feature vectors
    Optional labels for supervised selection

Outputs:
    Reduced List[Dict[str, float]] feature vectors
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    from sklearn.feature_selection import mutual_info_classif

    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available. Mutual information disabled.")

FeatureVector = Dict[str, float]


def _dict_to_matrix(features: List[FeatureVector]) -> Tuple[np.ndarray, List[str]]:
    """Convert feature dictionaries to matrix representation."""
    if not features:
        raise ValueError("Feature list cannot be empty")

    keys = sorted(features[0].keys())

    matrix = np.array(
        [[f.get(k, 0.0) for k in keys] for f in features],
        dtype=np.float32,
    )

    return matrix, keys


def _matrix_to_dict(matrix: np.ndarray, keys: List[str]) -> List[FeatureVector]:
    """Convert matrix back to dictionary feature format."""
    output = []

    for row in matrix:
        output.append({k: float(v) for k, v in zip(keys, row)})

    return output


@dataclass
class VarianceThresholdSelector:
    """
    Removes features with variance below a threshold.
    """

    threshold: float = 0.0
    selected_indices: List[int] = field(default_factory=list)

    def fit(self, X: np.ndarray) -> None:
        variances = np.var(X, axis=0)
        self.selected_indices = [i for i, v in enumerate(variances) if v > self.threshold]

        logger.debug(
            "VarianceThresholdSelector fitted | kept=%d removed=%d",
            len(self.selected_indices),
            X.shape[1] - len(self.selected_indices),
        )

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.selected_indices:
            raise RuntimeError("Selector must be fitted before transform")

        return X[:, self.selected_indices]


@dataclass
class CorrelationSelector:
    """
    Removes highly correlated features.
    """

    threshold: float = 0.95
    selected_indices: List[int] = field(default_factory=list)

    def fit(self, X: np.ndarray) -> None:
        corr = np.corrcoef(X, rowvar=False)

        keep = set(range(X.shape[1]))

        for i in range(X.shape[1]):
            for j in range(i + 1, X.shape[1]):
                if abs(corr[i, j]) > self.threshold and j in keep:
                    keep.remove(j)

        self.selected_indices = sorted(list(keep))

        logger.debug(
            "CorrelationSelector fitted | kept=%d removed=%d",
            len(self.selected_indices),
            X.shape[1] - len(self.selected_indices),
        )

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.selected_indices:
            raise RuntimeError("Selector must be fitted before transform")

        return X[:, self.selected_indices]


@dataclass
class TopKSelector:
    """
    Select top-K features based on variance or mutual information.
    """

    k: int = 50
    method: str = "variance"
    selected_indices: List[int] = field(default_factory=list)

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        if self.method == "variance":
            scores = np.var(X, axis=0)

        elif self.method == "mutual_info":
            if not SKLEARN_AVAILABLE:
                raise RuntimeError("scikit-learn required for mutual_info method")

            if y is None:
                raise ValueError("Labels required for mutual information selection")

            scores = mutual_info_classif(X, y)

        else:
            raise ValueError(f"Unsupported selection method: {self.method}")

        ranked = np.argsort(scores)[::-1]

        self.selected_indices = ranked[: self.k].tolist()

        logger.debug(
            "TopKSelector fitted | k=%d method=%s",
            self.k,
            self.method,
        )

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.selected_indices:
            raise RuntimeError("Selector must be fitted before transform")

        return X[:, self.selected_indices]


@dataclass
class FeatureSelectionPipeline:
    """
    High-level pipeline for dictionary-based feature selection.
    """

    selector: object
    feature_order: List[str] = field(default_factory=list)

    def fit(
        self,
        features: List[FeatureVector],
        labels: Optional[List[int]] = None,
    ) -> None:

        matrix, keys = _dict_to_matrix(features)

        self.feature_order = keys

        y = np.array(labels) if labels is not None else None

        if hasattr(self.selector, "fit"):
            if y is not None:
                self.selector.fit(matrix, y)
            else:
                self.selector.fit(matrix)

        logger.info(
            "FeatureSelectionPipeline fitted | samples=%d features=%d",
            matrix.shape[0],
            matrix.shape[1],
        )

    def transform(self, features: List[FeatureVector]) -> List[FeatureVector]:

        if not self.feature_order:
            raise RuntimeError("Pipeline must be fitted before transform")

        matrix = np.array(
            [[f.get(k, 0.0) for k in self.feature_order] for f in features],
            dtype=np.float32,
        )

        reduced = self.selector.transform(matrix)

        selected_indices = getattr(self.selector, "selected_indices", None)
        if selected_indices is None:
            raise AttributeError(
                f"Selector {type(self.selector).__name__} must have a "
                "'selected_indices' attribute after fitting"
            )
        selected_keys = [self.feature_order[i] for i in selected_indices]

        return _matrix_to_dict(reduced, selected_keys)

    def fit_transform(
        self,
        features: List[FeatureVector],
        labels: Optional[List[int]] = None,
    ) -> List[FeatureVector]:

        matrix, keys = _dict_to_matrix(features)

        self.feature_order = keys

        y = np.array(labels) if labels is not None else None

        if hasattr(self.selector, "fit"):
            if y is not None:
                self.selector.fit(matrix, y)
            else:
                self.selector.fit(matrix)

        reduced = self.selector.transform(matrix)

        selected_indices = getattr(self.selector, "selected_indices", None)
        if selected_indices is None:
            raise AttributeError(
                f"Selector {type(self.selector).__name__} must have a "
                "'selected_indices' attribute after fitting"
            )
        selected_keys = [keys[i] for i in selected_indices]

        return _matrix_to_dict(reduced, selected_keys)