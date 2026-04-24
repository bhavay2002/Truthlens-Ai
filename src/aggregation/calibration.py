from __future__ import annotations

import logging
from typing import Optional, Dict, Any, Union

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from sklearn.isotonic import IsotonicRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


logger = logging.getLogger(__name__)

EPS = 1e-12
ArrayLike = Union[np.ndarray, "torch.Tensor"]


# =========================================================
# BASE CALIBRATOR
# =========================================================

class BaseCalibrator:
    def __init__(self) -> None:
        self.fitted = False

    def fit(self, logits: np.ndarray, labels: np.ndarray):
        raise NotImplementedError

    def transform(self, logits: ArrayLike) -> ArrayLike:
        raise NotImplementedError

    def state_dict(self) -> Dict[str, Any]:
        raise NotImplementedError

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        raise NotImplementedError


# =========================================================
# TEMPERATURE SCALING (MULTICLASS)
# =========================================================

class TemperatureScaler(BaseCalibrator):
    def __init__(self, init_temp: float = 1.0) -> None:
        super().__init__()
        self.temperature = float(init_temp)

    def fit(self, logits: np.ndarray, labels: np.ndarray):
        logits = np.asarray(logits, dtype=np.float64)
        labels = np.asarray(labels)

        if logits.ndim != 2:
            raise ValueError("Temperature scaling requires 2D logits")

        T = self.temperature

        # simple optimization (gradient-free)
        for _ in range(50):
            probs = self._softmax(logits / T)
            loss = -np.mean(np.log(probs[np.arange(len(labels)), labels] + EPS))

            # finite difference gradient
            T_eps = T + 1e-3
            probs_eps = self._softmax(logits / T_eps)
            loss_eps = -np.mean(np.log(probs_eps[np.arange(len(labels)), labels] + EPS))

            grad = (loss_eps - loss) / 1e-3
            T -= 0.01 * grad
            T = max(T, 1e-3)

        self.temperature = float(T)
        self.fitted = True

        logger.info("[Calibration] Temperature fitted: %.4f", self.temperature)

    def transform(self, logits: ArrayLike) -> ArrayLike:
        arr = self._to_numpy(logits)
        scaled = arr / self.temperature
        probs = self._softmax(scaled)
        return self._to_output(probs, logits)

    def _softmax(self, x):
        x = x - np.max(x, axis=-1, keepdims=True)
        e = np.exp(x)
        return e / (np.sum(e, axis=-1, keepdims=True) + EPS)

    def _to_numpy(self, x):
        if TORCH_AVAILABLE and isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def _to_output(self, arr, like):
        if TORCH_AVAILABLE and isinstance(like, torch.Tensor):
            return torch.from_numpy(arr).to(like.device)
        return arr

    def state_dict(self):
        return {"temperature": self.temperature}

    def load_state_dict(self, state):
        self.temperature = state["temperature"]
        self.fitted = True


# =========================================================
# SIGMOID CALIBRATION (PLATT SCALING)
# =========================================================

class SigmoidCalibrator(BaseCalibrator):
    def __init__(self):
        super().__init__()
        self.a = 1.0
        self.b = 0.0

    def fit(self, logits: np.ndarray, labels: np.ndarray):
        logits = logits.reshape(-1)
        labels = labels.reshape(-1)

        a, b = self.a, self.b

        for _ in range(100):
            probs = 1 / (1 + np.exp(-(a * logits + b)))

            error = probs - labels
            grad_a = np.mean(error * logits)
            grad_b = np.mean(error)

            a -= 0.1 * grad_a
            b -= 0.1 * grad_b

        self.a = float(a)
        self.b = float(b)
        self.fitted = True

        logger.info("[Calibration] Sigmoid fitted: a=%.4f b=%.4f", self.a, self.b)

    def transform(self, logits: ArrayLike) -> ArrayLike:
        arr = self._to_numpy(logits)
        probs = 1 / (1 + np.exp(-(self.a * arr + self.b)))
        return self._to_output(probs, logits)

    def _to_numpy(self, x):
        if TORCH_AVAILABLE and isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def _to_output(self, arr, like):
        if TORCH_AVAILABLE and isinstance(like, torch.Tensor):
            return torch.from_numpy(arr).to(like.device)
        return arr

    def state_dict(self):
        return {"a": self.a, "b": self.b}

    def load_state_dict(self, state):
        self.a = state["a"]
        self.b = state["b"]
        self.fitted = True


# =========================================================
# ISOTONIC CALIBRATION (NON-PARAMETRIC)
# =========================================================

class IsotonicCalibrator(BaseCalibrator):
    def __init__(self):
        super().__init__()
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for isotonic calibration")
        self.model = IsotonicRegression(out_of_bounds="clip")

    def fit(self, logits: np.ndarray, labels: np.ndarray):
        logits = logits.reshape(-1)
        labels = labels.reshape(-1)

        self.model.fit(logits, labels)
        self.fitted = True

        logger.info("[Calibration] Isotonic regression fitted")

    def transform(self, logits: ArrayLike) -> ArrayLike:
        arr = logits.reshape(-1)
        calibrated = self.model.transform(arr)
        return calibrated.reshape(logits.shape)

    def state_dict(self):
        return {"model": self.model}

    def load_state_dict(self, state):
        self.model = state["model"]
        self.fitted = True


# =========================================================
# FACTORY
# =========================================================

def get_calibrator(method: str) -> BaseCalibrator:
    method = method.lower()

    if method == "temperature":
        return TemperatureScaler()

    if method == "sigmoid":
        return SigmoidCalibrator()

    if method == "isotonic":
        return IsotonicCalibrator()

    raise ValueError(f"Unknown calibration method: {method}")