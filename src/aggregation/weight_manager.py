from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from threading import RLock

import numpy as np


logger = logging.getLogger(__name__)


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


class WeightManager:

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        *,
        version: str = "v1",
        frozen: bool = False,
    ) -> None:

        self._lock = RLock()
        self.version = version
        self.frozen = frozen

        self.weights = (weights or DEFAULT_WEIGHTS).copy()
        self._validate_weights(self.weights)
        self._normalize_weights()

        logger.info(
            "[WeightManager] Initialized | version=%s frozen=%s",
            self.version,
            self.frozen,
        )

    # =========================
    # LOAD
    # =========================
    def load_weights_from_config(self, config_path: str | Path) -> Dict[str, float]:
        with self._lock:

            if self.frozen:
                raise RuntimeError("Weights are frozen and cannot be modified")

            config_path = Path(config_path)

            with config_path.open("r", encoding="utf-8") as f:
                loaded = json.load(f)

            if not isinstance(loaded, dict):
                raise ValueError("Weight configuration must be a dictionary")

            merged = self.weights.copy()
            merged.update(loaded)

            self._validate_weights(merged)
            self.weights = merged
            self._normalize_weights()

            logger.info(
                "[WeightManager] Loaded config | path=%s version=%s",
                config_path,
                self.version,
            )

            return self.get_weights()

    # =========================
    # VALIDATION
    # =========================
    def _validate_weights(self, weights: Dict[str, Any]) -> None:
        for key, value in weights.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"Weight '{key}' must be numeric")
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"Invalid weight '{key}': {value}")

    # =========================
    # NORMALIZATION
    # =========================
    def _normalize_group(self, keys):
        values = np.array([self.weights[k] for k in keys], dtype=np.float64)
        total = np.sum(values)

        if total <= 0:
            raise ValueError(f"Invalid weight group: {keys}")

        normalized = values / total

        for k, v in zip(keys, normalized):
            self.weights[k] = float(v)

    def _normalize_weights(self):
        for group in WEIGHT_GROUPS.values():
            self._normalize_group(group)

        logger.debug("[WeightManager] Normalized weights: %s", self.weights)

    # =========================
    # ADJUST
    # =========================
    def adjust_weight(self, key: str, value: float) -> Dict[str, float]:
        with self._lock:

            if self.frozen:
                raise RuntimeError("Weights are frozen")

            self.weights[key] = float(value)

            self._validate_weights(self.weights)
            self._normalize_weights()

            logger.info("[WeightManager] Adjusted %s=%s", key, value)

            return self.get_weights()

    # =========================
    # CONFIDENCE-AWARE WEIGHTS
    # =========================
    def get_weighted(self, confidence: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        with self._lock:

            weights = self.weights.copy()

            if confidence:
                for k, conf in confidence.items():
                    if k in weights:
                        weights[k] *= float(np.clip(conf, 0.0, 1.0))

                # renormalize groups after confidence scaling
                for group in WEIGHT_GROUPS.values():
                    total = sum(weights[k] for k in group)
                    if total > 0:
                        for k in group:
                            weights[k] /= total

            return weights

    # =========================
    # ACCESS
    # =========================
    def get_weights(self) -> Dict[str, float]:
        with self._lock:
            return self.weights.copy()