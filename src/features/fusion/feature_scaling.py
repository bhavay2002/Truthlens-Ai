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
from typing import Dict, List, Union

import numpy as np

logger = logging.getLogger(__name__)

FeatureVector = Dict[str, float]


def _dict_to_matrix(features: List[FeatureVector]) -> tuple[np.ndarray, List[str]]:
    """
    Convert list of feature dictionaries to matrix representation.
    """
    if not features:
        raise ValueError("Feature list cannot be empty")

    keys = sorted(features[0].keys())

    matrix = np.array([[f.get(k, 0.0) for k in keys] for f in features], dtype=np.float32)

    return matrix, keys


def _matrix_to_dict(matrix: np.ndarray, keys: List[str]) -> List[FeatureVector]:
    """
    Convert matrix back to dictionary representation.
    """
    output = []

    for row in matrix:
        output.append({k: float(v) for k, v in zip(keys, row)})

    return output


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

        return (X - self.mean_) / self.std_


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

        denom = self.max_ - self.min_
        denom[denom == 0] = 1.0

        return (X - self.min_) / denom


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

        return (X - self.median_) / self.iqr_


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

    def fit(self, features: List[FeatureVector]) -> None:
        matrix, keys = _dict_to_matrix(features)

        self.feature_order = keys

        self.scaler.fit(matrix)

        logger.info(
            "FeatureScalingPipeline fitted | samples=%d features=%d",
            matrix.shape[0],
            matrix.shape[1],
        )

    def transform(self, features: List[FeatureVector]) -> List[FeatureVector]:

        if not self.feature_order:
            raise RuntimeError("FeatureScalingPipeline must be fitted before transform")

        matrix = np.array(
            [[f.get(k, 0.0) for k in self.feature_order] for f in features],
            dtype=np.float32,
        )

        scaled = self.scaler.transform(matrix)

        return _matrix_to_dict(scaled, self.feature_order)

    def fit_transform(self, features: List[FeatureVector]) -> List[FeatureVector]:

        matrix, keys = _dict_to_matrix(features)

        self.feature_order = keys

        scaled = self.scaler.fit_transform(matrix)

        return _matrix_to_dict(scaled, keys)