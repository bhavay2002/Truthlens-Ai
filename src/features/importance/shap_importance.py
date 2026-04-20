from __future__ import annotations

import logging 
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False
    shap = None
    logger.warning("SHAP library not available. SHAP importance disabled.")


@dataclass
class ShapImportance:
    model: object
    max_samples: Optional[int] = 1000
    random_seed: int = 42

    # =========================================================
    # EXPLAINER
    # =========================================================

    def _create_explainer(self, X: np.ndarray):

        if not SHAP_AVAILABLE:
            raise RuntimeError("SHAP library is required")

        # Try optimized explainers first
        try:
            return shap.Explainer(self.model)
        except Exception:
            pass

        # Fallback: KernelExplainer with safe prediction fn
        background = X[: min(len(X), 100)]

        if hasattr(self.model, "predict_proba"):
            predict_fn = self.model.predict_proba
        else:
            predict_fn = self.model.predict

        return shap.KernelExplainer(predict_fn, background)

    # =========================================================
    # SAMPLING
    # =========================================================

    def _sample_data(self, X: np.ndarray) -> np.ndarray:

        if self.max_samples is None or X.shape[0] <= self.max_samples:
            return np.asarray(X, dtype=np.float32)

        rng = np.random.default_rng(self.random_seed)
        indices = rng.choice(X.shape[0], size=self.max_samples, replace=False)

        logger.info("Sampling %d rows for SHAP computation", self.max_samples)

        return np.asarray(X[indices], dtype=np.float32)

    # =========================================================
    # SHAP PROCESSING
    # =========================================================

    def _process_shap_values(self, shap_values):

        values = getattr(shap_values, "values", shap_values)

        # list → stack
        if isinstance(values, list):
            values = np.stack(values, axis=0)

        values = np.asarray(values, dtype=np.float32)

        # Handle multi-class
        if values.ndim == 3:
            # shape: (samples, features, classes)
            values = values[:, :, -1]  # positive class

        if values.ndim != 2:
            raise ValueError(f"Unsupported SHAP shape: {values.shape}")

        # sanitize
        values = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=-1.0)

        return values

    # =========================================================
    # MAIN
    # =========================================================

    def compute(
        self,
        X: np.ndarray,
        feature_names: List[str],
        shap_values=None,
    ) -> Dict[str, float]:

        if not isinstance(X, np.ndarray) or X.ndim != 2:
            raise ValueError("X must be 2D numpy array")

        if X.shape[1] != len(feature_names):
            raise ValueError("Feature names must match feature dimension")

        X_sample = self._sample_data(X)

        if shap_values is None:
            try:
                explainer = self._create_explainer(X_sample)
                shap_values = explainer(X_sample)
            except Exception as e:
                logger.warning("SHAP failed, returning zero importance: %s", e)
                return {name: 0.0 for name in feature_names}

        values = self._process_shap_values(shap_values)

        # global importance
        importance_scores = np.mean(np.abs(values), axis=0)

        # normalize to [0,1]
        max_val = np.max(importance_scores) or 1.0
        importance_scores = importance_scores / max_val

        results = {
            name: float(score)
            for name, score in zip(feature_names, importance_scores)
        }

        logger.info("Computed SHAP importance for %d features", len(results))

        return results

    # =========================================================
    # RANKING
    # =========================================================

    def rank_features(
        self,
        X: np.ndarray,
        feature_names: List[str],
    ) -> List[Tuple[str, float]]:

        scores = self.compute(X, feature_names)

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # =========================================================
    # TOP-K
    # =========================================================

    def top_k(
        self,
        X: np.ndarray,
        feature_names: List[str],
        k: int = 20,
    ) -> List[Tuple[str, float]]:

        if k <= 0:
            return []

        ranked = self.rank_features(X, feature_names)

        return ranked[:k]