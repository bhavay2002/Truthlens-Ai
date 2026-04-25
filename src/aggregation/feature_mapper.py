from __future__ import annotations

import logging
from typing import Dict, Any, Optional, List

import numpy as np

logger = logging.getLogger(__name__)

EPS = 1e-12


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
    probs = np.asarray(probs)
    return float(-np.sum(probs * np.log(probs + EPS)))


def _normalize_array(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    max_val = np.max(arr) + EPS
    return arr / max_val


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

                # 🔥 FIX 1: CLIP BEFORE USE
                val = float(np.clip(val, 0.0, 1.0))

                section_data[feature_name] = val

            if section_data:
                profile[section] = section_data

        if self.normalize:
            profile = self._normalize(profile)

        return profile

    # =====================================================
    # MULTI-TASK SUPPORT (FIXED)
    # =====================================================

    def map_from_model_outputs(
        self,
        model_outputs: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, float]]:

        flat = {}

        for task, outputs in model_outputs.items():

            if not isinstance(outputs, dict):
                continue

            probs = outputs.get("probabilities")
            logits = outputs.get("logits")

            # -------------------------
            # PROBABILITIES
            # -------------------------
            if probs is not None:

                probs = np.asarray(probs)

                # 🔥 FIX 2: PER-TASK NORMALIZATION
                if probs.ndim > 1:
                    probs = _normalize_array(probs[0])
                else:
                    probs = _normalize_array(probs)

                conf = float(np.max(probs))

                flat[f"{task}_probability"] = float(np.clip(conf, 0.0, 1.0))
                flat[f"{task}_confidence"] = float(np.clip(conf, 0.0, 1.0))

                entropy = _compute_entropy(probs)
                flat[f"{task}_entropy"] = float(np.clip(entropy, 0.0, 1.0))

            # -------------------------
            # LOGITS
            # -------------------------
            if logits is not None:
                val = float(np.mean(logits))
                flat[f"{task}_logit"] = float(np.clip(val, 0.0, 1.0))

        return self.map_features(flat)

    # =====================================================
    # CONFIDENCE EXTRACTION
    # =====================================================

    def extract_confidence(
        self,
        model_outputs: Dict[str, Dict[str, Any]],
    ) -> Dict[str, float]:

        confidence = {}

        for task, outputs in model_outputs.items():

            probs = outputs.get("probabilities")

            if probs is None:
                continue

            probs = np.asarray(probs)

            if probs.ndim > 1:
                probs = probs[0]

            conf = float(np.max(probs))

            confidence[task] = float(np.clip(conf, 0.0, 1.0))

        return confidence

    # =====================================================
    # NORMALIZATION
    # =====================================================

    def _normalize(self, profile):

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
    # BATCH
    # =====================================================

    def map_batch(
        self,
        batch_outputs: List[Dict[str, Any]],
    ) -> List[Dict[str, Dict[str, float]]]:

        return [self.map_features(x) for x in batch_outputs]