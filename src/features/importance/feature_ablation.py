"""
File Name: feature_ablation.py
Module: Feature Engineering - Ablation Analysis
Description:
    Implements feature ablation analysis utilities for the TruthLens ML
    pipeline. Feature ablation evaluates the contribution of individual
    features or feature groups by systematically removing them and
    measuring the resulting impact on model performance.

    This module supports:
        • single-feature ablation
        • group-feature ablation
        • performance delta analysis
        • ranked importance estimation

    The implementation is framework-agnostic and works with any model
    exposing a `predict()` interface. Custom evaluation metrics can be
    provided to measure model performance.

Dependencies:
    dataclasses
    typing
    logging
    numpy

Inputs:
    model
    feature matrix (numpy.ndarray)
    labels
    feature names
    optional feature groups

Outputs:
    Ablation performance impact metrics
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

MetricFn = Callable[[np.ndarray, np.ndarray], float]


def accuracy_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Default evaluation metric.
    """
    return float(np.mean(y_true == y_pred))


@dataclass
class FeatureAblation:
    """
    Performs feature ablation experiments to measure feature importance.
    """

    model: object
    metric: MetricFn = accuracy_metric

    def _predict(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self.model, "predict"):
            return self.model.predict(X)

        raise RuntimeError("Model must implement predict()")

    def _baseline_score(self, X: np.ndarray, y: np.ndarray) -> float:
        pred = self._predict(X)
        score = self.metric(y, pred)

        logger.info("Baseline model score: %.6f", score)

        return score

    def single_feature_ablation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
    ) -> Dict[str, float]:
        """
        Perform ablation by removing one feature at a time.
        """

        if X.ndim != 2:
            raise ValueError("X must be a 2D matrix")
        if y.ndim != 1 or len(y) != X.shape[0]:
            raise ValueError("y must be 1D and match X rows")
        if X.shape[1] != len(feature_names):
            raise ValueError("Feature names must match feature dimension")

        baseline = self._baseline_score(X, y)

        results: Dict[str, float] = {}

        for i, name in enumerate(feature_names):

            X_ablate = X.copy()
            X_ablate[:, i] = 0.0

            pred = self._predict(X_ablate)

            score = self.metric(y, pred)

            impact = baseline - score

            results[name] = float(impact)

            logger.debug(
                "Ablation | feature=%s impact=%.6f",
                name,
                impact,
            )

        return results

    def group_ablation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        groups: Dict[str, List[str]],
    ) -> Dict[str, float]:
        """
        Perform ablation on feature groups.
        """

        if X.ndim != 2:
            raise ValueError("X must be a 2D matrix")
        if y.ndim != 1 or len(y) != X.shape[0]:
            raise ValueError("y must be 1D and match X rows")

        baseline = self._baseline_score(X, y)

        name_to_idx = {name: i for i, name in enumerate(feature_names)}

        results: Dict[str, float] = {}

        for group_name, group_features in groups.items():

            indices = [name_to_idx[f] for f in group_features if f in name_to_idx]
            if not indices:
                results[group_name] = 0.0
                continue

            X_ablate = X.copy()
            X_ablate[:, indices] = 0.0

            pred = self._predict(X_ablate)

            score = self.metric(y, pred)

            impact = baseline - score

            results[group_name] = float(impact)

            logger.debug(
                "Group ablation | group=%s impact=%.6f",
                group_name,
                impact,
            )

        return results

    def rank_features(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
    ) -> List[Tuple[str, float]]:
        """
        Rank features by ablation impact.
        """

        scores = self.single_feature_ablation(X, y, feature_names)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        logger.info("Computed feature ranking via ablation")

        return ranked

    def top_k(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        k: int = 20,
    ) -> List[Tuple[str, float]]:
        """
        Return top-k important features.
        """

        ranked = self.rank_features(X, y, feature_names)

        return ranked[:k]
