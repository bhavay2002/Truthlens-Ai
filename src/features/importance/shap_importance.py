"""
File Name: shap_importance.py
Module: Feature Engineering - Feature Importance
Description:
    Computes SHAP (SHapley Additive exPlanations) based feature importance
    scores for machine learning models. SHAP values quantify the contribution
    of each feature to model predictions based on cooperative game theory.

    This module supports models compatible with the SHAP framework and
    produces global feature importance metrics by aggregating absolute
    SHAP values across samples.

    The implementation includes:
        • automatic explainer selection
        • SHAP value computation
        • global feature importance ranking
        • top-k feature extraction
        • optional sampling for large datasets

Dependencies:
    dataclasses
    typing
    logging
    numpy
    shap

Inputs:
    model
    feature matrix (numpy.ndarray)
    feature names

Outputs:
    Feature importance scores based on SHAP values
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import shap

    SHAP_AVAILABLE = True
except Exception:  # noqa: BLE001
    SHAP_AVAILABLE = False
    shap = None
    logger.warning("SHAP library not available. SHAP importance disabled.")


@dataclass
class ShapImportance:
    """
    SHAP-based feature importance calculator.
    """

    model: object
    max_samples: Optional[int] = 1000
    random_seed: int = 42

    def _create_explainer(self, X: np.ndarray):
        """
        Automatically select SHAP explainer.
        """

        if not SHAP_AVAILABLE:
            raise RuntimeError("SHAP library is required for SHAP importance")

        try:
            return shap.Explainer(self.model, X)
        except Exception:  # noqa: BLE001
            background = X[: min(len(X), 100)]
            return shap.KernelExplainer(self.model.predict, background)

    def _sample_data(self, X: np.ndarray) -> np.ndarray:
        """
        Sample dataset for SHAP computation if dataset is large.
        """

        if self.max_samples is None or X.shape[0] <= self.max_samples:
            return X

        rng = np.random.default_rng(self.random_seed)
        indices = rng.choice(X.shape[0], size=self.max_samples, replace=False)

        logger.info("Sampling %d rows for SHAP computation", self.max_samples)

        return X[indices]

    def compute(
        self,
        X: np.ndarray,
        feature_names: List[str],
    ) -> Dict[str, float]:
        """
        Compute global SHAP importance.

        Parameters
        ----------
        X : np.ndarray
        feature_names : List[str]

        Returns
        -------
        Dict[str, float]
        """

        if X.shape[1] != len(feature_names):
            raise ValueError("Feature names must match feature dimension")
        if X.ndim != 2:
            raise ValueError("X must be 2D")

        X_sample = self._sample_data(X)

        explainer = self._create_explainer(X_sample)

        shap_values = explainer(X_sample)

        values = getattr(shap_values, "values", shap_values)

        if isinstance(values, list):
            values = np.stack(values, axis=0).mean(axis=0)
        values = np.asarray(values)

        if values.ndim == 3:
            values = np.mean(values, axis=1)
        elif values.ndim != 2:
            raise ValueError(f"Unsupported SHAP values shape: {values.shape}")

        importance_scores = np.mean(np.abs(values), axis=0)

        results = {
            name: float(score)
            for name, score in zip(feature_names, importance_scores)
        }

        logger.info("Computed SHAP importance for %d features", len(results))

        return results

    def rank_features(
        self,
        X: np.ndarray,
        feature_names: List[str],
    ) -> List[Tuple[str, float]]:
        """
        Rank features by SHAP importance.
        """

        scores = self.compute(X, feature_names)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        logger.info("Ranked features by SHAP importance")

        return ranked

    def top_k(
        self,
        X: np.ndarray,
        feature_names: List[str],
        k: int = 20,
    ) -> List[Tuple[str, float]]:
        """
        Return top-k important features.
        """

        ranked = self.rank_features(X, feature_names)

        return ranked[:k]