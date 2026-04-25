# src/features/feature_selection.py

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
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
EPS = 1e-8


# =========================================================
# UTILITIES
# =========================================================

def _dict_to_matrix(features: List[FeatureVector]) -> Tuple[np.ndarray, List[str]]:
    if not features:
        raise ValueError("Feature list cannot be empty")

    keys = sorted({k for f in features for k in f.keys()})
    name_to_idx = {k: i for i, k in enumerate(keys)}

    matrix = np.zeros((len(features), len(keys)), dtype=np.float32)

    for i, f in enumerate(features):
        row = matrix[i]
        for k, v in f.items():
            j = name_to_idx.get(k)
            if j is not None:
                if not np.isfinite(v):
                    v = 0.0
                row[j] = float(v)

    return matrix, keys


def _matrix_to_dict(matrix: np.ndarray, keys: List[str]) -> List[FeatureVector]:
    return [
        {k: float(v) for k, v in zip(keys, row) if v != 0.0}
        for row in matrix
    ]


# =========================================================
# SELECTORS
# =========================================================

@dataclass
class VarianceThresholdSelector:
    threshold: float = 0.0
    selected_indices: List[int] = field(default_factory=list)
    scores_: Optional[np.ndarray] = None
    fitted: bool = False

    def fit(self, X: np.ndarray, y=None) -> None:
        var = np.var(X, axis=0)
        self.scores_ = var
        self.selected_indices = [i for i, v in enumerate(var) if v > self.threshold]
        self.fitted = True

    def transform(self, X: np.ndarray) -> np.ndarray:
        self._check()
        return X[:, self.selected_indices]

    def _check(self):
        if not self.fitted:
            raise RuntimeError("Selector not fitted")


# ---------------------------------------------------------

@dataclass
class CorrelationSelector:
    threshold: float = 0.95
    selected_indices: List[int] = field(default_factory=list)
    fitted: bool = False

    def fit(self, X: np.ndarray, y=None) -> None:

        corr = np.corrcoef(X, rowvar=False)
        corr = np.nan_to_num(corr)

        upper = np.triu(np.abs(corr), k=1)

        to_drop = set()

        for i in range(upper.shape[0]):
            if i in to_drop:
                continue

            for j in np.where(upper[i] > self.threshold)[0]:
                to_drop.add(j)

        keep = sorted(set(range(X.shape[1])) - to_drop)

        self.selected_indices = keep
        self.fitted = True

    def transform(self, X: np.ndarray) -> np.ndarray:
        self._check()
        return X[:, self.selected_indices]

    def _check(self):
        if not self.fitted:
            raise RuntimeError("Selector not fitted")


# ---------------------------------------------------------

@dataclass
class TopKSelector:
    k: int = 50
    method: str = "variance"

    selected_indices: List[int] = field(default_factory=list)
    scores_: Optional[np.ndarray] = None
    fitted: bool = False

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:

        if self.method == "variance":
            scores = np.var(X, axis=0)

        elif self.method == "mutual_info":
            if not SKLEARN_AVAILABLE:
                raise RuntimeError("sklearn required for mutual_info")

            if y is None:
                raise ValueError("Labels required")

            scores = mutual_info_classif(X, y)

        else:
            raise ValueError("Invalid method")

        self.scores_ = scores

        ranked = np.argsort(scores)[::-1]
        self.selected_indices = ranked[: self.k].tolist()
        self.fitted = True

    def transform(self, X: np.ndarray) -> np.ndarray:
        self._check()
        return X[:, self.selected_indices]

    def _check(self):
        if not self.fitted:
            raise RuntimeError("Selector not fitted")


# =========================================================
# COMPOSITE SELECTOR
# =========================================================

@dataclass
class CompositeSelector:

    selectors: List[object]
    fitted: bool = False

    def fit(self, X: np.ndarray, y=None) -> None:

        for sel in self.selectors:
            if hasattr(sel, "fit"):
                if y is not None:
                    sel.fit(X, y)
                else:
                    sel.fit(X)

                X = sel.transform(X)

        self.fitted = True

    def transform(self, X: np.ndarray) -> np.ndarray:

        if not self.fitted:
            raise RuntimeError("CompositeSelector not fitted")

        for sel in self.selectors:
            X = sel.transform(X)

        return X


# =========================================================
# PIPELINE
# =========================================================

@dataclass
class FeatureSelectionPipeline:

    selector: object

    feature_order: List[str] = field(default_factory=list)
    selected_keys: List[str] = field(default_factory=list)

    _name_to_idx: Dict[str, int] = field(default_factory=dict, init=False)
    fitted: bool = False

    # -----------------------------------------------------

    def fit(
        self,
        features: List[FeatureVector],
        labels: Optional[List[int]] = None,
    ) -> None:

        X, keys = _dict_to_matrix(features)

        self.feature_order = keys
        self._name_to_idx = {k: i for i, k in enumerate(keys)}

        y = np.array(labels) if labels is not None else None

        if hasattr(self.selector, "fit"):
            self.selector.fit(X, y)

        selected_idx = getattr(self.selector, "selected_indices", None)

        if selected_idx is None:
            raise ValueError("Selector must expose selected_indices")

        self.selected_keys = [keys[i] for i in selected_idx]

        self.fitted = True

        logger.info(
            "FeatureSelection fitted | original=%d selected=%d",
            len(keys),
            len(self.selected_keys),
        )

    # -----------------------------------------------------

    def transform(
        self,
        features: List[FeatureVector],
        *,
        return_array: bool = True,
    ):

        if not self.fitted:
            raise RuntimeError("Pipeline not fitted")

        X = np.zeros((len(features), len(self.feature_order)), dtype=np.float32)

        for i, f in enumerate(features):
            for k, v in f.items():
                j = self._name_to_idx.get(k)
                if j is not None:
                    if not np.isfinite(v):
                        v = 0.0
                    X[i, j] = float(v)

        X = self.selector.transform(X)

        if return_array:
            return X

        return _matrix_to_dict(X, self.selected_keys)

    # -----------------------------------------------------

    def fit_transform(
        self,
        features: List[FeatureVector],
        labels: Optional[List[int]] = None,
        *,
        return_array: bool = True,
    ):
        self.fit(features, labels)
        return self.transform(features, return_array=return_array)

    # -----------------------------------------------------
    # PERSISTENCE (CRITICAL)
    # -----------------------------------------------------

    def save(self, path: str | Path) -> None:

        data = {
            "feature_order": self.feature_order,
            "selected_keys": self.selected_keys,
        }

        Path(path).write_text(json.dumps(data))
        logger.info("FeatureSelection saved → %s", path)

    def load(self, path: str | Path) -> None:

        data = json.loads(Path(path).read_text())

        self.feature_order = data["feature_order"]
        self.selected_keys = data["selected_keys"]

        self._name_to_idx = {k: i for i, k in enumerate(self.feature_order)}

        self.fitted = True

        logger.info("FeatureSelection loaded ← %s", path)

    # -----------------------------------------------------

    def get_selected_features(self) -> List[str]:
        return self.selected_keys