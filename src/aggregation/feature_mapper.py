from __future__ import annotations

import logging
from typing import Dict, Any, Optional

import numpy as np


logger = logging.getLogger(__name__)


# =========================================================
# DEFAULT FEATURE MAP (CAN BE OVERRIDDEN VIA CONFIG)
# =========================================================

DEFAULT_FEATURE_MAP: Dict[str, Dict[str, str]] = {
    "bias": {
        "logit": "bias_logit",
        "probability": "bias_probability",
        "prediction": "bias_prediction",
    },
    "emotion": {
        "intensity": "emotion_intensity",
        "confidence": "emotion_confidence",
    },
    "narrative": {
        "score": "narrative_score",
    },
    "discourse": {
        "coherence": "coherence_score",
    },
    "graph": {
        "consistency": "graph_consistency",
    },
    "ideology": {
        "score": "ideology_score",
    },
    "analysis": {
        "confidence": "analysis_confidence",
    },
}


# =========================================================
# VALIDATION UTIL
# =========================================================

def _validate_numeric(value: Any, *, strict: bool) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        if strict:
            raise TypeError(f"Invalid numeric value: {value}")
        return None

    if not np.isfinite(value):
        if strict:
            raise ValueError(f"Non-finite value: {value}")
        return None

    return float(value)


# =========================================================
# FEATURE MAPPER
# =========================================================

class FeatureMapper:
    """
    Maps raw model outputs → standardized aggregation profile.

    Converts heterogeneous outputs into:
    {
        "bias": {...},
        "emotion": {...},
        ...
    }
    """

    def __init__(
        self,
        feature_map: Optional[Dict[str, Dict[str, str]]] = None,
        *,
        strict: bool = False,
    ) -> None:
        self.feature_map = feature_map or DEFAULT_FEATURE_MAP
        self.strict = strict

        logger.info(
            "[FeatureMapper] Initialized | strict=%s | sections=%d",
            self.strict,
            len(self.feature_map),
        )

    # =========================================================
    # MAIN API
    # =========================================================
    def map_features(
        self,
        raw_outputs: Dict[str, Any],
    ) -> Dict[str, Dict[str, float]]:
        """
        Convert raw outputs into aggregation-ready profile.
        """

        if not isinstance(raw_outputs, dict):
            raise ValueError("raw_outputs must be a dictionary")

        profile: Dict[str, Dict[str, float]] = {}

        for section, mapping in self.feature_map.items():

            section_data: Dict[str, float] = {}

            for feature_name, raw_key in mapping.items():

                if raw_key not in raw_outputs:
                    continue

                val = _validate_numeric(raw_outputs[raw_key], strict=self.strict)
                if val is None:
                    continue

                section_data[feature_name] = val

            if section_data:
                profile[section] = section_data
            else:
                logger.debug(
                    "[FeatureMapper] No valid features for section: %s",
                    section,
                )

        return profile

    # =========================================================
    # MULTI-TASK HEAD SUPPORT
    # =========================================================
    def map_from_model_outputs(
        self,
        model_outputs: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, float]]:
        """
        Handles outputs from multi-head model.

        Expected format:
        {
            "bias": {"logits": ..., "probabilities": ..., "confidence": ...},
            "emotion": {...},
            ...
        }
        """

        if not isinstance(model_outputs, dict):
            raise ValueError("model_outputs must be a dictionary")

        flat: Dict[str, Any] = {}

        for task, outputs in model_outputs.items():
            if not isinstance(outputs, dict):
                continue

            for k, v in outputs.items():
                flat_key = f"{task}_{k}"
                flat[flat_key] = v

        return self.map_features(flat)

    # =========================================================
    # CONFIDENCE EXTRACTION
    # =========================================================
    def extract_confidence(
        self,
        model_outputs: Dict[str, Dict[str, Any]],
    ) -> Dict[str, float]:
        """
        Extract confidence per section (for weighting).
        """

        confidence: Dict[str, float] = {}

        for section, outputs in model_outputs.items():
            if not isinstance(outputs, dict):
                continue

            conf = outputs.get("confidence")

            if conf is None:
                continue

            val = _validate_numeric(conf, strict=self.strict)
            if val is None:
                continue

            confidence[section] = float(np.clip(val, 0.0, 1.0))

        return confidence

    # =========================================================
    # BATCH SUPPORT
    # =========================================================
    def map_batch(
        self,
        batch_outputs: list[Dict[str, Any]],
    ) -> list[Dict[str, Dict[str, float]]]:
        """
        Batch mapping for inference pipelines.
        """

        if not isinstance(batch_outputs, list):
            raise ValueError("batch_outputs must be a list")

        results = []

        for item in batch_outputs:
            mapped = self.map_features(item)
            results.append(mapped)

        return results