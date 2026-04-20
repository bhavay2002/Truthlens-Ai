"""
File Name: weight_manager.py
Module: TruthLens AI - Aggregation Weight Manager
Description:
    Manages dynamic scoring weights used by the TruthLens aggregation system.
    Instead of hardcoding weights inside scoring modules, this component loads,
    validates, normalizes, and optionally adjusts weights from configuration.

    This enables flexible experimentation and tuning of the TruthLens scoring
    engine without modifying source code.

Dependencies:
    logging
    typing
    pathlib
    json
    numpy

Inputs:
    configuration file containing scoring weights

Outputs:
    validated and normalized weight dictionaries
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Any

import numpy as np


logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS: Dict[str, float] = {
    # Manipulation group (additive, normalised to sum=1)
    "bias": 0.40,
    "emotion": 0.30,
    "narrative": 0.20,
    "analysis_influence_manipulation": 0.10,
    # Credibility group (additive terms only; penalty excluded from normalisation)
    "discourse": 0.55,
    "graph": 0.35,
    "analysis_influence_credibility": 0.10,
    # Standalone penalty — used as subtraction, not normalised with group
    "credibility_bias_penalty": 0.20,
    # Final composite (normalised to sum=1)
    "final_credibility": 0.5,
    "final_manipulation": 0.3,
    "final_ideology": 0.2,
}

ALLOWED_WEIGHT_KEYS = set(DEFAULT_WEIGHTS.keys())

WEIGHT_GROUPS = {
    "manipulation": ("bias", "emotion", "narrative", "analysis_influence_manipulation"),
    # credibility_bias_penalty is used as a *subtraction* in the scoring
    # formula; normalising it together with additive terms would corrupt the
    # formula semantics.  Exclude it from group normalisation so it retains
    # its absolute value.
    "credibility": ("discourse", "graph", "analysis_influence_credibility"),
    "final": ("final_credibility", "final_manipulation", "final_ideology"),
}


class WeightManager:
    def __init__(self, weights: Dict[str, float] | None = None) -> None:
        self.weights = (weights or DEFAULT_WEIGHTS).copy()
        self._cached_weights: Dict[str, float] | None = None
        self._validate_weights(self.weights)
        self._normalize_weights()

    def load_weights_from_config(self, config_path: str | Path) -> Dict[str, float]:
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Weight config not found: {config_path}")

        try:
            with config_path.open("r", encoding="utf-8") as f:
                loaded = json.load(f)
        except Exception as exc:
            logger.exception("Failed to load weight configuration")
            raise RuntimeError("Weight configuration loading failed") from exc

        if not isinstance(loaded, dict):
            raise ValueError("Weight configuration must be a dictionary")

        merged = self.weights.copy()
        merged.update(loaded)

        self._validate_weights(merged)
        self.weights = merged
        self._normalize_weights()

        logger.info("Weights loaded from config: %s", config_path)
        return self.weights.copy()

    def _validate_weights(self, weights: Dict[str, Any]) -> None:
        if not isinstance(weights, dict) or not weights:
            raise ValueError("Weights must be a non-empty dictionary")

        unknown_keys = set(weights.keys()) - ALLOWED_WEIGHT_KEYS
        if unknown_keys:
            raise ValueError(f"Unknown weight keys: {sorted(unknown_keys)}")

        for key, value in weights.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"Weight '{key}' must be numeric (non-boolean), got {type(value).__name__}")
            fval = float(value)
            if not np.isfinite(fval):
                raise ValueError(f"Weight '{key}' must be finite, got {value}")
            if fval < 0:
                raise ValueError(f"Weight '{key}' cannot be negative")

    def _normalize_group(self, group_name: str) -> None:
        keys = WEIGHT_GROUPS[group_name]
        present = [k for k in keys if k in self.weights]
        if not present:
            return

        values = np.array([self.weights[k] for k in present], dtype=np.float64)
        total = float(np.sum(values))
        if total == 0:
            raise ValueError(f"Sum of weights in group '{group_name}' cannot be zero")

        normalized = values / total
        for key, val in zip(present, normalized):
            self.weights[key] = float(val)

    def _normalize_weights(self) -> None:
        for group_name in WEIGHT_GROUPS:
            self._normalize_group(group_name)
        self._cached_weights = self.weights.copy()

    def adjust_weight(self, key: str, value: float) -> Dict[str, float]:
        if key not in ALLOWED_WEIGHT_KEYS:
            raise KeyError(f"Unknown weight key: '{key}'")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ValueError("Weight value must be non-negative numeric (non-boolean)")

        self.weights[key] = float(value)
        self._validate_weights(self.weights)
        self._normalize_weights()

        logger.info("Weight adjusted: %s=%s", key, value)
        return self.weights.copy()

    def get_weights(self) -> Dict[str, float]:
        if self._cached_weights is None:
            self._cached_weights = self.weights.copy()
        return self._cached_weights.copy()