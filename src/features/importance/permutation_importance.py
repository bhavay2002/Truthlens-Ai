"""
File Name: permutation_importance.py
Module: Feature Engineering - Feature Importance
Description:
    Implements permutation feature importance used for model interpretability
    and feature analysis. The algorithm measures the importance of each
    feature by randomly shuffling its values and observing the resulting
    drop in model performance.

    This implementation is framework-agnostic and works with any model that
    exposes a predict or predict_proba interface. It supports custom metric
    functions and deterministic reproducibility through controlled random
    seeds.

    The module is useful for:
        • model interpretability
        • feature ranking
        • feature pruning
        • experiment diagnostics

Dependencies:
    dataclasses
    typing
    logging
    numpy

Inputs:
    model
    feature matrix (numpy.ndarray)
    labels
    evaluation metric

Outputs:
    Feature importance scores
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
    Default accuracy metric.
    """
    return float(np.mean(y_true == y_pred))


@dataclass
class PermutationImportance:
    """
    Permutation feature importance calculator.
    """

    model: object
    metric: MetricFn = accuracy_metric
    n_repeats: int = 5
    random_seed: int = 42

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Run model prediction using predict or predict_proba.
        """

        if hasattr(self.model, "predict"):
            return self.model.predict(X)

        raise RuntimeError("Model must implement predict()")

    def compute(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
    ) -> Dict[str, float]:
        """
        Compute permutation importance scores.

        Parameters
        ----------
        X : np.ndarray
        y : np.ndarray
        feature_names : List[str]

        Returns
        -------
        Dict[str, float]
        """

        if self.n_repeats <= 0:
            raise ValueError("n_repeats must be > 0")
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        if y.ndim != 1 or len(y) != X.shape[0]:
            raise ValueError("y must be 1D and match X rows")
        if X.shape[1] != len(feature_names):
            raise ValueError("Feature name count must match feature dimension")

        rng = np.random.default_rng(self.random_seed)

        baseline_pred = self._predict(X)
        baseline_score = self.metric(y, baseline_pred)

        if not np.isfinite(baseline_score):
            raise ValueError("Baseline metric is not finite")

        logger.info("Baseline model score: %.6f", baseline_score)

        importances: Dict[str, float] = {}

        for feature_idx, name in enumerate(feature_names):

            scores = []

            for _ in range(self.n_repeats):

                X_permuted = X.copy()

                shuffled = rng.permutation(X_permuted[:, feature_idx])

                X_permuted[:, feature_idx] = shuffled

                pred = self._predict(X_permuted)

                score = self.metric(y, pred)

                if not np.isfinite(score):
                    raise ValueError(
                        f"Metric returned non-finite score for feature '{name}'"
                    )

                scores.append(baseline_score - score)

            importance = float(np.mean(scores))

            importances[name] = importance

            logger.debug(
                "Permutation importance | feature=%s score=%.6f",
                name,
                importance,
            )

        return importances

    def rank_features(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
    ) -> List[Tuple[str, float]]:
        """
        Compute and rank features by importance.
        """

        scores = self.compute(X, y, feature_names)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        logger.info("Computed permutation feature ranking")

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