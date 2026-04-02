"""
File Name: feature_statistics.py
Module: Feature Engineering - Feature Statistics
Description:
    Computes descriptive statistics and diagnostics for feature vectors
    produced by the TruthLens feature pipeline. The module provides tools
    for inspecting feature distributions, detecting skewness, identifying
    constant features, and analyzing feature correlations.

    These utilities are useful for:
        • dataset diagnostics
        • feature quality analysis
        • feature drift monitoring
        • experiment reporting

    The module operates on dictionary-based feature vectors and internally
    converts them to NumPy matrices for statistical computation.

Dependencies:
    dataclasses
    typing
    logging
    numpy

Inputs:
    List[Dict[str, float]] feature vectors

Outputs:
    Feature statistics dictionary and optional reports
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

FeatureVector = Dict[str, float]


def _dict_to_matrix(features: List[FeatureVector]) -> Tuple[np.ndarray, List[str]]:
    """
    Convert list of feature dictionaries to matrix representation.
    """

    if not features:
        raise ValueError("Feature list cannot be empty")

    keys = sorted(features[0].keys())

    matrix = np.array(
        [[f.get(k, 0.0) for k in keys] for f in features],
        dtype=np.float32,
    )

    return matrix, keys


@dataclass
class FeatureStatistics:
    """
    Computes descriptive statistics for feature datasets.
    """

    def compute_basic_statistics(
        self,
        features: List[FeatureVector],
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute mean, std, min, max for each feature.
        """

        X, keys = _dict_to_matrix(features)

        stats: Dict[str, Dict[str, float]] = {}

        for idx, name in enumerate(keys):
            column = X[:, idx]

            stats[name] = {
                "mean": float(np.mean(column)),
                "std": float(np.std(column)),
                "min": float(np.min(column)),
                "max": float(np.max(column)),
                "median": float(np.median(column)),
            }

        logger.info(
            "Computed basic statistics | features=%d samples=%d",
            X.shape[1],
            X.shape[0],
        )

        return stats

    def compute_variance(self, features: List[FeatureVector]) -> Dict[str, float]:
        """
        Compute variance for each feature.
        """

        X, keys = _dict_to_matrix(features)

        variances = np.var(X, axis=0)

        return {k: float(v) for k, v in zip(keys, variances)}

    def detect_constant_features(
        self,
        features: List[FeatureVector],
        tolerance: float = 1e-12,
    ) -> List[str]:
        """
        Identify features with near-zero variance.
        """

        X, keys = _dict_to_matrix(features)

        variances = np.var(X, axis=0)

        constant = [keys[i] for i, v in enumerate(variances) if v < tolerance]

        logger.warning("Detected %d constant features", len(constant))

        return constant

    def compute_skewness(self, features: List[FeatureVector]) -> Dict[str, float]:
        """
        Estimate skewness for each feature distribution.
        """

        X, keys = _dict_to_matrix(features)

        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)

        std[std == 0] = 1.0

        skew = np.mean(((X - mean) / std) ** 3, axis=0)

        return {k: float(v) for k, v in zip(keys, skew)}

    def compute_correlation_matrix(
        self,
        features: List[FeatureVector],
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Compute feature correlation matrix.
        """

        X, keys = _dict_to_matrix(features)

        corr = np.corrcoef(X, rowvar=False)

        logger.info("Computed feature correlation matrix")

        return corr, keys

    def dataset_summary(
        self,
        features: List[FeatureVector],
    ) -> Dict[str, float]:
        """
        Compute dataset-level feature diagnostics.
        """

        X, keys = _dict_to_matrix(features)

        variances = np.var(X, axis=0)

        summary = {
            "num_samples": float(X.shape[0]),
            "num_features": float(X.shape[1]),
            "mean_variance": float(np.mean(variances)),
            "max_variance": float(np.max(variances)),
            "min_variance": float(np.min(variances)),
        }

        logger.info(
            "Feature dataset summary | samples=%d features=%d",
            X.shape[0],
            X.shape[1],
        )

        return summary