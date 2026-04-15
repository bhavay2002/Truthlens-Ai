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

    all_keys = set()
    for feature_vector in features:
        all_keys.update(feature_vector.keys())

    keys = sorted(all_keys)
    name_to_idx = {k: i for i, k in enumerate(keys)}

    matrix = np.zeros((len(features), len(keys)), dtype=np.float32)

    for i, feature_vector in enumerate(features):
        row = matrix[i]
        for key, value in feature_vector.items():
            j = name_to_idx.get(key)
            if j is not None:
                row[j] = value

    return matrix, keys


def _matrix_to_dict(matrix: np.ndarray, keys: List[str]) -> List[FeatureVector]:
    """Convert matrix back to dictionary feature format."""
    return [
        {key: float(value) for key, value in zip(keys, row) if value != 0.0}
        for row in matrix
    ]


@dataclass
class VarianceThresholdSelector:
    """
    Removes features with variance below a threshold.
    """

    threshold: float = 0.0
    selected_indices: List[int] = field(default_factory=list)
    fitted: bool = False

    def fit(self, X: np.ndarray) -> None:
        variances = np.var(X, axis=0)
        self.selected_indices = [i for i, v in enumerate(variances) if v > self.threshold]
        self.fitted = True

        logger.debug(
            "VarianceThresholdSelector fitted | kept=%d removed=%d",
            len(self.selected_indices),
            X.shape[1] - len(self.selected_indices),
        )

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Selector must be fitted before transform")

        if not self.selected_indices:
            return np.empty((X.shape[0], 0), dtype=X.dtype)

        return X[:, self.selected_indices]


@dataclass
class CorrelationSelector:
    """
    Removes highly correlated features.
    """

    threshold: float = 0.95
    selected_indices: List[int] = field(default_factory=list)
    fitted: bool = False

    def fit(self, X: np.ndarray) -> None:
        corr = np.corrcoef(X, rowvar=False)

        keep = set(range(X.shape[1]))

        for i in range(X.shape[1]):
            for j in range(i + 1, X.shape[1]):
                if abs(corr[i, j]) > self.threshold and j in keep:
                    keep.remove(j)

        self.selected_indices = sorted(list(keep))
        self.fitted = True

        logger.debug(
            "CorrelationSelector fitted | kept=%d removed=%d",
            len(self.selected_indices),
            X.shape[1] - len(self.selected_indices),
        )

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Selector must be fitted before transform")

        if not self.selected_indices:
            return np.empty((X.shape[0], 0), dtype=X.dtype)

        return X[:, self.selected_indices]


@dataclass
class TopKSelector:
    """
    Select top-K features based on variance or mutual information.
    """

    k: int = 50
    method: str = "variance"
    selected_indices: List[int] = field(default_factory=list)
    fitted: bool = False

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
        self.fitted = True

        logger.debug(
            "TopKSelector fitted | k=%d method=%s",
            self.k,
            self.method,
        )

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Selector must be fitted before transform")

        if not self.selected_indices:
            return np.empty((X.shape[0], 0), dtype=X.dtype)

        return X[:, self.selected_indices]


@dataclass
class FeatureSelectionPipeline:
    """
    High-level pipeline for dictionary-based feature selection.
    """

    selector: object
    feature_order: List[str] = field(default_factory=list)
    _name_to_idx: Dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _buffer: np.ndarray | None = field(default=None, init=False, repr=False)
    use_fp16_output: bool = False

    def fit(
        self,
        features: List[FeatureVector],
        labels: Optional[List[int]] = None,
    ) -> None:

        matrix, keys = _dict_to_matrix(features)

        self.feature_order = keys
        self._name_to_idx = {k: i for i, k in enumerate(keys)}

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

    def transform(
        self,
        features: List[FeatureVector],
        *,
        return_array: bool = True,
    ) -> List[FeatureVector] | np.ndarray:

        if not self.feature_order:
            raise RuntimeError("Pipeline must be fitted before transform")

        expected_shape = (len(features), len(self.feature_order))
        if self._buffer is None or self._buffer.shape != expected_shape:
            self._buffer = np.zeros(expected_shape, dtype=np.float32)

        matrix = self._buffer
        matrix.fill(0.0)
        name_to_idx = self._name_to_idx

        for i, feature_vector in enumerate(features):
            row = matrix[i]
            for key, value in feature_vector.items():
                j = name_to_idx.get(key)
                if j is not None:
                    row[j] = value

        reduced = self.selector.transform(matrix)

        if self.use_fp16_output:
            reduced = reduced.astype(np.float16)

        if return_array:
            return reduced.copy()

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
        *,
        return_array: bool = True,
    ) -> List[FeatureVector] | np.ndarray:

        matrix, keys = _dict_to_matrix(features)

        self.feature_order = keys
        self._name_to_idx = {k: i for i, k in enumerate(keys)}

        y = np.array(labels) if labels is not None else None

        if hasattr(self.selector, "fit"):
            if y is not None:
                self.selector.fit(matrix, y)
            else:
                self.selector.fit(matrix)

        reduced = self.selector.transform(matrix)

        if self.use_fp16_output:
            reduced = reduced.astype(np.float16)

        if return_array:
            return reduced

        selected_indices = getattr(self.selector, "selected_indices", None)
        if selected_indices is None:
            raise AttributeError(
                f"Selector {type(self.selector).__name__} must have a "
                "'selected_indices' attribute after fitting"
            )
        selected_keys = [keys[i] for i in selected_indices]

        return _matrix_to_dict(reduced, selected_keys)