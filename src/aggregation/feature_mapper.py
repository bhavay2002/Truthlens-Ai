from __future__ import annotations

import logging
from typing import Dict, Any, Optional, List

import numpy as np

logger = logging.getLogger(__name__)

EPS = 1e-12

_LOGIT_CLIP = 88.0


# =========================================================
# DEFAULT FEATURE MAP
# =========================================================

DEFAULT_FEATURE_MAP: Dict[str, Dict[str, str]] = {
    "bias": {
        "probability": "bias_probability",
        "confidence": "bias_confidence",
        "entropy": "bias_entropy",
    },
    "emotion": {
        "intensity": "emotion_intensity",
        "confidence": "emotion_confidence",
        "entropy": "emotion_entropy",
    },
    "narrative": {
        "score": "narrative_score",
    },
    "graph": {
        "consistency": "graph_consistency",
    },
    "ideology": {
        "score": "ideology_score",
    },
}


# =========================================================
# UTILS
# =========================================================

def _safe_numeric(value: Any, strict: bool) -> Optional[float]:

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        if strict:
            raise TypeError(f"Invalid numeric: {value}")
        return None

    if not np.isfinite(value):
        if strict:
            raise ValueError(f"Non-finite: {value}")
        return None

    return float(value)


def _compute_entropy(probs: np.ndarray) -> float:
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, EPS, 1.0)
    return float(-np.sum(probs * np.log(probs)))


def _normalize_probs(arr: np.ndarray) -> np.ndarray:
    total = np.sum(arr) + EPS
    return arr / total


def _logits_to_prob(logits: np.ndarray) -> float:
    logits = np.asarray(logits, dtype=np.float64).ravel()
    clipped = np.clip(logits, -_LOGIT_CLIP, _LOGIT_CLIP)
    if logits.size == 1:
        prob = 1.0 / (1.0 + np.exp(-clipped[0]))
    else:
        e = np.exp(clipped - np.max(clipped))
        prob = float(np.max(e / (np.sum(e) + EPS)))
    return float(np.clip(prob, 0.0, 1.0))


# =========================================================
# FEATURE MAPPER
# =========================================================

class FeatureMapper:

    def __init__(
        self,
        feature_map: Optional[Dict[str, Dict[str, str]]] = None,
        *,
        strict: bool = False,
        normalize: bool = True,
    ):
        self.feature_map = feature_map or DEFAULT_FEATURE_MAP
        self.strict = strict
        self.normalize = normalize

        logger.info(
            "[FeatureMapper] init | strict=%s normalize=%s",
            strict,
            normalize,
        )

    # =====================================================
    # MAIN
    # =====================================================

    def map_features(self, raw_outputs: Dict[str, Any]) -> Dict[str, Dict[str, float]]:

        if not isinstance(raw_outputs, dict):
            raise ValueError("raw_outputs must be dict")

        profile = {}

        for section, mapping in self.feature_map.items():

            section_data = {}

            for feature_name, raw_key in mapping.items():

                if raw_key not in raw_outputs:
                    continue

                val = _safe_numeric(raw_outputs[raw_key], self.strict)

                if val is None:
                    continue

                val = float(np.clip(val, 0.0, 1.0))
                section_data[feature_name] = val

            if section_data:
                profile[section] = section_data

        if self.normalize:
            profile = self._normalize(profile)

        return profile

    # =====================================================
    # MULTI-TASK SUPPORT
    # =====================================================

    def map_from_model_outputs(
        self,
        model_outputs: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, float]]:

        flat: Dict[str, float] = {}

        for task, outputs in model_outputs.items():

            if not isinstance(outputs, dict):
                continue

            probs = outputs.get("probabilities")
            logits = outputs.get("logits")

            # -------------------------
            # PROBABILITIES (primary signal)
            # -------------------------
            if probs is not None:

                probs_arr = np.nan_to_num(
                    np.asarray(probs, dtype=np.float64),
                    nan=0.0, posinf=1.0, neginf=0.0,
                )

                if probs_arr.ndim > 1:
                    probs_arr = probs_arr[0]

                probs_arr = np.clip(probs_arr, 0.0, 1.0)
                probs_arr = _normalize_probs(probs_arr)

                conf = float(np.max(probs_arr))
                entropy = _compute_entropy(probs_arr)

                flat[f"{task}_probability"] = float(np.clip(conf, 0.0, 1.0))
                flat[f"{task}_confidence"] = float(np.clip(conf, 0.0, 1.0))
                flat[f"{task}_entropy"] = float(np.clip(entropy / np.log(max(probs_arr.size, 2)), 0.0, 1.0))

            # -------------------------
            # LOGITS — convert via sigmoid/softmax, not raw clip
            # -------------------------
            elif logits is not None:
                logit_arr = np.asarray(logits, dtype=np.float64).ravel()
                prob = _logits_to_prob(logit_arr)
                flat[f"{task}_logit"] = prob

        return self.map_features(flat)

    # =====================================================
    # CONFIDENCE EXTRACTION
    # =====================================================

    def extract_confidence(
        self,
        model_outputs: Dict[str, Dict[str, Any]],
    ) -> Dict[str, float]:

        confidence: Dict[str, float] = {}

        for task, outputs in model_outputs.items():

            if not isinstance(outputs, dict):
                continue

            probs = outputs.get("probabilities")

            if probs is None:
                continue

            probs_arr = np.nan_to_num(
                np.asarray(probs, dtype=np.float64),
                nan=0.0, posinf=1.0, neginf=0.0,
            )

            if probs_arr.ndim > 1:
                probs_arr = probs_arr[0]

            probs_arr = np.clip(probs_arr, 0.0, 1.0)
            conf = float(np.max(probs_arr)) if probs_arr.size > 0 else 0.0
            if not np.isfinite(conf):
                conf = 0.0
            confidence[task] = float(np.clip(conf, 0.0, 1.0))

        return confidence

    # =====================================================
    # NORMALIZATION (per-section max-norm, preserves [0,1])
    # =====================================================

    def _normalize(self, profile: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:

        for section in profile:

            values = list(profile[section].values())

            if not values:
                continue

            max_val = max(values) + EPS

            for k in profile[section]:
                profile[section][k] = float(
                    np.clip(profile[section][k] / max_val, 0.0, 1.0)
                )

        return profile

    # =====================================================
    # BATCH (vectorized per-task)
    # =====================================================

    def map_batch(
        self,
        batch_outputs: List[Dict[str, Any]],
    ) -> List[Dict[str, Dict[str, float]]]:

        return [self.map_features(x) for x in batch_outputs]
