from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from threading import RLock

import numpy as np

logger = logging.getLogger(__name__)

EPS = 1e-12


# =========================================================
# DEFAULTS
# =========================================================

DEFAULT_WEIGHTS: Dict[str, float] = {
    "bias": 0.40,
    "emotion": 0.30,
    "narrative": 0.20,
    "analysis_influence_manipulation": 0.10,

    "discourse": 0.55,
    "graph": 0.35,
    "analysis_influence_credibility": 0.10,

    "credibility_bias_penalty": 0.20,

    "final_credibility": 0.5,
    "final_manipulation": 0.3,
    "final_ideology": 0.2,
}


WEIGHT_GROUPS = {
    "manipulation": ("bias", "emotion", "narrative", "analysis_influence_manipulation"),
    "credibility": ("discourse", "graph", "analysis_influence_credibility"),
    "final": ("final_credibility", "final_manipulation", "final_ideology"),
}


# =========================================================
# MANAGER
# =========================================================

class WeightManager:

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        *,
        version: str = "v2",
        frozen: bool = False,
        smoothing: float = 0.1,
    ) -> None:

        self._lock = RLock()
        self.version = version
        self.frozen = frozen
        self.smoothing = smoothing

        self.weights = (weights or DEFAULT_WEIGHTS).copy()
        self._validate_weights(self.weights)
        self._normalize_weights()

        logger.info(
            "[WeightManager] Initialized | version=%s frozen=%s",
            self.version,
            self.frozen,
        )

    # =====================================================
    # LOAD
    # =====================================================

    def load_weights_from_config(self, config_path: str | Path) -> Dict[str, float]:

        with self._lock:

            if self.frozen:
                raise RuntimeError("Weights are frozen")

            config_path = Path(config_path)

            with config_path.open("r", encoding="utf-8") as f:
                loaded = json.load(f)

            if not isinstance(loaded, dict):
                raise ValueError("Weight config must be dict")

            merged = self.weights.copy()
            merged.update(loaded)

            self._validate_weights(merged)
            self.weights = merged
            self._normalize_weights()

            return self.get_weights()

    # =====================================================
    # VALIDATION
    # =====================================================

    def _validate_weights(self, weights: Dict[str, Any]):

        for k, v in weights.items():

            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise TypeError(f"{k} must be numeric")

            if not np.isfinite(v) or v < 0:
                raise ValueError(f"{k} invalid: {v}")

    # =====================================================
    # NORMALIZATION
    # =====================================================

    def _normalize_group(self, keys):

        values = np.array([self.weights[k] for k in keys], dtype=np.float64)

        total = np.sum(values)

        if total <= 0:
            raise ValueError(f"Invalid group: {keys}")

        values = values / total

        for k, v in zip(keys, values):
            self.weights[k] = float(v)

    def _normalize_weights(self):
        for group in WEIGHT_GROUPS.values():
            self._normalize_group(group)

    # =====================================================
    # ADAPTIVE WEIGHTING (🔥 FIXED)
    # =====================================================

    def get_adaptive_weights(
        self,
        *,
        confidence: Optional[Dict[str, float]] = None,
        entropy: Optional[Dict[str, float]] = None,
        explanation_scores: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:

        with self._lock:

            weights = self.weights.copy()

            for key in weights:

                scale = 1.0

                # -------------------------
                # confidence scaling
                # -------------------------
                if confidence and key in confidence:
                    scale *= np.clip(confidence[key], 0.0, 1.0)

                # -------------------------
                # entropy penalty
                # -------------------------
                if entropy and key in entropy:
                    scale *= (1.0 - np.clip(entropy[key], 0.0, 1.0))

                # -------------------------
                # explainability boost
                # -------------------------
                if explanation_scores and key in explanation_scores:
                    scale *= (1.0 + explanation_scores[key])

                # =====================================================
                # 🔥 CRITICAL FIX: SCALE CLIPPING
                # =====================================================
                scale = np.clip(scale, 0.1, 2.0)

                weights[key] *= scale

            # -------------------------
            # smoothing (stability)
            # -------------------------
            for k in weights:
                weights[k] = (
                    (1 - self.smoothing) * self.weights[k]
                    + self.smoothing * weights[k]
                )

            # -------------------------
            # renormalize
            # -------------------------
            for group in WEIGHT_GROUPS.values():
                total = sum(weights[k] for k in group) + EPS
                for k in group:
                    weights[k] /= total

            return weights

    # =====================================================
    # SIMPLE CONFIDENCE MODE (BACKWARD COMPAT)
    # =====================================================

    def get_weighted(
        self,
        confidence: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:

        return self.get_adaptive_weights(confidence=confidence)

    # =====================================================
    # UPDATE
    # =====================================================

    def adjust_weight(self, key: str, value: float):

        with self._lock:

            if self.frozen:
                raise RuntimeError("Weights frozen")

            self.weights[key] = float(value)

            self._validate_weights(self.weights)
            self._normalize_weights()

            return self.get_weights()

    # =====================================================
    # ACCESS
    # =====================================================

    def get_weights(self) -> Dict[str, float]:
        with self._lock:
            return self.weights.copy()