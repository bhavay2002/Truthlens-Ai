from __future__ import annotations

import logging
from typing import Iterable, Optional, Dict, Any, Union

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


logger = logging.getLogger(__name__)

EPS = 1e-12


ArrayLike = Union[np.ndarray, "torch.Tensor"]


# =========================
# BASE NORMALIZER
# =========================
class ScoreNormalizer:
    """
    Production-grade normalizer supporting:
    - fit / transform paradigm
    - numpy + torch
    - strict validation
    """

    def __init__(
        self,
        method: str = "minmax",
        *,
        strict: bool = False,
        feature_range: tuple[float, float] = (0.0, 1.0),
    ):
        self.method = method.lower()
        self.strict = strict
        self.feature_range = feature_range

        self.fitted = False
        self.stats: Dict[str, Any] = {}

        logger.info("[Normalizer] Initialized | method=%s", self.method)

    # =========================
    # UTIL
    # =========================
    def _to_array(self, values: Iterable[float]) -> np.ndarray:
        try:
            arr = np.asarray(list(values), dtype=np.float32)
        except Exception as exc:
            raise TypeError("values must be numeric iterable") from exc

        if arr.size == 0:
            raise ValueError("values cannot be empty")

        if not np.isfinite(arr).all():
            msg = "Non-finite values detected"
            if self.strict:
                raise ValueError(msg)
            logger.warning("[Normalizer] %s → applying safe fallback", msg)
            arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)

        return arr

    def _to_tensor(self, arr: np.ndarray, like: ArrayLike):
        if TORCH_AVAILABLE and isinstance(like, torch.Tensor):
            return torch.from_numpy(arr).to(like.device)
        return arr

    # =========================
    # FIT
    # =========================
    def fit(self, values: Iterable[float]) -> "ScoreNormalizer":
        arr = self._to_array(values)

        if self.method == "minmax":
            self.stats["min"] = float(np.min(arr))
            self.stats["max"] = float(np.max(arr))

        elif self.method == "zscore":
            self.stats["mean"] = float(np.mean(arr))
            self.stats["std"] = float(np.std(arr))

        elif self.method == "robust":
            self.stats["median"] = float(np.median(arr))
            self.stats["q1"] = float(np.percentile(arr, 25))
            self.stats["q3"] = float(np.percentile(arr, 75))

        else:
            raise ValueError(f"Unsupported method: {self.method}")

        self.fitted = True
        logger.info("[Normalizer] Fitted | stats=%s", self.stats)

        return self

    # =========================
    # TRANSFORM
    # =========================
    def transform(self, values: ArrayLike) -> ArrayLike:
        if not self.fitted:
            raise RuntimeError("Normalizer must be fitted before transform")

        is_tensor = TORCH_AVAILABLE and isinstance(values, torch.Tensor)

        arr = values.detach().cpu().numpy() if is_tensor else self._to_array(values)

        if self.method == "minmax":
            vmin = self.stats["min"]
            vmax = self.stats["max"]

            if abs(vmax - vmin) < EPS:
                result = np.zeros_like(arr)
            else:
                a, b = self.feature_range
                norm = (arr - vmin) / (vmax - vmin)
                result = norm * (b - a) + a

        elif self.method == "zscore":
            mean = self.stats["mean"]
            std = self.stats["std"]

            if std < EPS:
                result = np.zeros_like(arr)
            else:
                result = (arr - mean) / std

        elif self.method == "robust":
            median = self.stats["median"]
            iqr = self.stats["q3"] - self.stats["q1"]

            if abs(iqr) < EPS:
                result = np.zeros_like(arr)
            else:
                result = (arr - median) / iqr

        else:
            raise ValueError(f"Unsupported method: {self.method}")

        return self._to_tensor(result.astype(np.float32), values)

    # =========================
    # FIT + TRANSFORM
    # =========================
    def fit_transform(self, values: Iterable[float]) -> np.ndarray:
        return self.fit(values).transform(values)

    # =========================
    # SERIALIZATION
    # =========================
    def state_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "stats": self.stats,
            "feature_range": self.feature_range,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.method = state["method"]
        self.stats = state["stats"]
        self.feature_range = tuple(state["feature_range"])
        self.fitted = True


# =========================
# ADVANCED UTILITIES
# =========================
def log_scale(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    return np.log1p(arr)


def percentile_clip(values: Iterable[float], low=1, high=99) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    lo = np.percentile(arr, low)
    hi = np.percentile(arr, high)
    return np.clip(arr, lo, hi)


def sigmoid_calibration(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    return 1 / (1 + np.exp(-arr))