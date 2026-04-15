"""
File Name: feature_scaling.py
Module: Feature Engineering - Feature Scaling
Description:
    Provides feature scaling utilities used within the TruthLens feature
    engineering pipeline. The module implements deterministic scaling
    strategies commonly used in ML systems, including:

        • Standardization (z-score scaling)
        • Min-Max normalization
        • Robust scaling (median/IQR)
        • Log scaling for skewed distributions

    The implementation supports both dictionary-based feature vectors
    (used throughout the TruthLens feature system) and NumPy arrays for
    compatibility with downstream ML frameworks.

    The scaler objects follow a fit/transform paradigm similar to
    scikit-learn to support reproducibility and pipeline integration.

Dependencies:
    dataclasses
    typing
    logging
    numpy

Inputs:
    Dict[str, float] or numpy.ndarray feature vectors

Outputs:
    Scaled feature vectors
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)

FeatureVector = Dict[str, float]


def _dict_to_matrix(features: List[FeatureVector]) -> tuple[np.ndarray, List[str]]:
    """
    Convert list of feature dictionaries to matrix representation.
    """
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
    """
    Convert matrix back to dictionary representation.
    """
    return [
        {keys[index]: float(value) for index, value in enumerate(row) if value != 0.0}
        for row in matrix
    ]


@dataclass
class BaseScaler:
    """
    Base class for feature scalers.
    """

    fitted: bool = False

    def fit(self, X: np.ndarray) -> None:
        raise NotImplementedError

    def transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)


@dataclass
class StandardScaler(BaseScaler):
    """
    Standard z-score scaler.
    """

    mean_: np.ndarray | None = None
    std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> None:
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)

        self.std_[self.std_ == 0] = 1.0

        self.fitted = True

        logger.debug("StandardScaler fitted | features=%d", X.shape[1])

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("StandardScaler must be fitted before transform")

        X -= self.mean_
        X /= self.std_
        return X


@dataclass
class MinMaxScaler(BaseScaler):
    """
    Min-Max normalization scaler.
    """

    min_: np.ndarray | None = None
    max_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> None:
        self.min_ = np.min(X, axis=0)
        self.max_ = np.max(X, axis=0)

        self.fitted = True

        logger.debug("MinMaxScaler fitted | features=%d", X.shape[1])

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("MinMaxScaler must be fitted before transform")

        X -= self.min_
        denom = self.max_ - self.min_
        denom[denom == 0] = 1.0
        X /= denom

        return X


@dataclass
class RobustScaler(BaseScaler):
    """
    Robust scaling using median and interquartile range.
    """

    median_: np.ndarray | None = None
    iqr_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> None:
        self.median_ = np.median(X, axis=0)

        q75 = np.percentile(X, 75, axis=0)
        q25 = np.percentile(X, 25, axis=0)

        self.iqr_ = q75 - q25
        self.iqr_[self.iqr_ == 0] = 1.0

        self.fitted = True

        logger.debug("RobustScaler fitted | features=%d", X.shape[1])

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("RobustScaler must be fitted before transform")

        X -= self.median_
        X /= self.iqr_

        return X


@dataclass
class LogScaler(BaseScaler):
    """
    Log transformation scaler for skewed distributions.
    """

    epsilon: float = 1e-9

    def fit(self, X: np.ndarray) -> None:
        self.fitted = True

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("LogScaler must be fitted before transform")

        return np.log1p(np.maximum(X, 0) + self.epsilon)


@dataclass
class FeatureScalingPipeline:
    """
    High-level scaling pipeline for dictionary-based feature vectors.
    """

    scaler: BaseScaler
    feature_order: List[str] = field(default_factory=list)
    _name_to_idx: Dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _buffer: np.ndarray | None = field(default=None, init=False, repr=False)
    use_fp16_output: bool = False

    def fit(self, features: List[FeatureVector]) -> None:
        matrix, keys = _dict_to_matrix(features)

        self.feature_order = keys
        self._name_to_idx = {k: i for i, k in enumerate(keys)}

        self.scaler.fit(matrix)

        logger.info(
            "FeatureScalingPipeline fitted | samples=%d features=%d",
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
            raise RuntimeError("FeatureScalingPipeline must be fitted before transform")

        expected_shape = (len(features), len(self.feature_order))
        if self._buffer is None or self._buffer.shape != expected_shape:
            self._buffer = np.zeros(expected_shape, dtype=np.float32)

        matrix = self._buffer
        matrix.fill(0.0)
        name_to_idx = self._name_to_idx

        for i, feature_vector in enumerate(features):
            row = matrix[i]
            f_local = feature_vector
            row_local = row

            for key, value in f_local.items():
                j = name_to_idx.get(key)
                if j is not None:
                    row_local[j] = value

        scaled = self.scaler.transform(matrix)

        if self.use_fp16_output:
            scaled = scaled.astype(np.float16)

        if return_array:
            return scaled.copy()

        return _matrix_to_dict(scaled, self.feature_order)

    def fit_transform(
        self,
        features: List[FeatureVector],
        *,
        return_array: bool = True,
    ) -> List[FeatureVector] | np.ndarray:

        matrix, keys = _dict_to_matrix(features)

        self.feature_order = keys
        self._name_to_idx = {k: i for i, k in enumerate(keys)}

        scaled = self.scaler.fit_transform(matrix)

        if self.use_fp16_output:
            scaled = scaled.astype(np.float16)

        if return_array:
            return scaled

        return _matrix_to_dict(scaled, keys)
