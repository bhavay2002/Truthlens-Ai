from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from src.analysis.feature_schema import get_schema

logger = logging.getLogger(__name__)


# =========================================================
# CONSTANTS
# =========================================================

EPS = 1e-9


# =========================================================
# CONFIG
# =========================================================

@dataclass(slots=True)
class BiasProfileConfig:

    # Section weights
    bias_weight: float = 1.0
    emotion_weight: float = 1.0
    narrative_weight: float = 1.0
    discourse_weight: float = 1.0
    ideology_weight: float = 0.6

    # Normalization
    normalize_values: bool = True
    normalization_method: str = "minmax"  # minmax | zscore | robust

    clip_values: bool = True
    clip_range: tuple[float, float] = (0.0, 1.0)

    # Advanced
    global_normalization: bool = True
    apply_softmax_to_ideology: bool = True

    # Score aggregation
    aggregation_method: str = "mean"  # mean | weighted_mean


# =========================================================
# BUILDER
# =========================================================

class BiasProfileBuilder:

    PROFILE_SECTIONS = (
        "bias",
        "emotion",
        "narrative",
        "discourse",
        "ideology",
    )

    def __init__(self, config: BiasProfileConfig | None = None):
        self.config = config or BiasProfileConfig()
        logger.info("BiasProfileBuilder initialized")

    # =====================================================
    # MAIN ENTRY
    # =====================================================

    def build_profile(
        self,
        *,
        bias: Dict[str, float],
        emotion: Dict[str, float],
        narrative: Dict[str, float],
        discourse: Dict[str, float],
        ideology: Dict[str, float],
    ) -> Dict[str, Any]:

        profile = {
            "metadata": {
                "created_at": int(time.time()),
                "sections": list(self.PROFILE_SECTIONS),
            }
        }

        # ---- Process each section ----
        for section_name, data in {
            "bias": bias,
            "emotion": emotion,
            "narrative": narrative,
            "discourse": discourse,
            "ideology": ideology,
        }.items():
            profile[section_name] = self._process_section(data)

        # ---- Ideology calibration ----
        if self.config.apply_softmax_to_ideology:
            profile["ideology"] = self._softmax(profile["ideology"])

        # ---- Global normalization ----
        if self.config.global_normalization:
            profile = self._global_normalize(profile)

        # ---- Metrics ----
        profile["metrics"] = self._compute_metrics(profile)

        # ---- Final score ----
        profile["bias_score"] = self._compute_bias_score(profile)

        return profile

    # =====================================================
    # SECTION PROCESSING
    # =====================================================

    def _process_section(self, data: Dict[str, Any]) -> Dict[str, float]:

        data = self._sanitize(data)

        if self.config.normalize_values:
            data = self._normalize(data)

        if self.config.clip_values:
            data = self._clip(data)

        return data

    # =====================================================
    # SANITIZATION
    # =====================================================

    def _sanitize(self, data: Dict[str, Any]) -> Dict[str, float]:

        cleaned = {}

        for k, v in data.items():
            try:
                v = float(v)
                if not np.isfinite(v):
                    v = 0.0
            except Exception:
                v = 0.0

            cleaned[k] = v

        return cleaned

    # =====================================================
    # NORMALIZATION
    # =====================================================

    def _normalize(self, data: Dict[str, float]) -> Dict[str, float]:

        if not data:
            return data

        values = np.array(list(data.values()), dtype=np.float32)

        if self.config.normalization_method == "zscore":
            mean, std = values.mean(), values.std()
            if std < EPS:
                return data
            norm = (values - mean) / (std + EPS)

        elif self.config.normalization_method == "robust":
            median = np.median(values)
            iqr = np.percentile(values, 75) - np.percentile(values, 25)
            if iqr < EPS:
                return data
            norm = (values - median) / (iqr + EPS)

        else:  # minmax
            min_v, max_v = values.min(), values.max()
            if max_v - min_v < EPS:
                return data
            norm = (values - min_v) / (max_v - min_v + EPS)

        return dict(zip(data.keys(), norm.astype(float)))

    # =====================================================
    # GLOBAL NORMALIZATION (FIXED)
    # =====================================================

    def _global_normalize(self, profile: Dict[str, Any]) -> Dict[str, Any]:

        values = []

        for section in self.PROFILE_SECTIONS:
            values.extend(profile.get(section, {}).values())

        if not values:
            return profile

        arr = np.array(values, dtype=np.float32)
        min_v, max_v = arr.min(), arr.max()

        if max_v - min_v < EPS:
            return profile

        scale = max_v - min_v + EPS

        for section in self.PROFILE_SECTIONS:
            profile[section] = {
                k: float((v - min_v) / scale)
                for k, v in profile[section].items()
            }

        return profile

    # =====================================================
    # SOFTMAX (STABLE)
    # =====================================================

    def _softmax(self, data: Dict[str, float]) -> Dict[str, float]:

        if not data:
            return data

        values = np.array(list(data.values()), dtype=np.float32)

        values = values - np.max(values)  # stability
        exp = np.exp(values)

        probs = exp / (exp.sum() + EPS)

        return dict(zip(data.keys(), probs.astype(float)))

    # =====================================================
    # CLIPPING
    # =====================================================

    def _clip(self, data: Dict[str, float]) -> Dict[str, float]:

        low, high = self.config.clip_range

        return {
            k: float(np.clip(v, low, high))
            for k, v in data.items()
        }

    # =====================================================
    # METRICS (IMPROVED)
    # =====================================================

    def _compute_metrics(self, profile: Dict[str, Any]) -> Dict[str, float]:

        ideology_vals = np.array(
            list(profile.get("ideology", {}).values()), dtype=np.float32
        )

        if ideology_vals.size > 0:
            p = ideology_vals / (ideology_vals.sum() + EPS)
            entropy = float(-np.sum(p * np.log(p + EPS)))
            dominance = float(np.max(p))
        else:
            entropy = dominance = 0.0

        bias_vals = np.array(
            list(profile.get("bias", {}).values()), dtype=np.float32
        )

        variance = float(np.var(bias_vals)) if bias_vals.size else 0.0

        return {
            "bias_variance": variance,
            "ideology_entropy": entropy,
            "ideology_dominance": dominance,
        }

    # =====================================================
    # FINAL SCORE (FIXED WEIGHTING)
    # =====================================================

    def _compute_bias_score(self, profile: Dict[str, Any]) -> float:

        weights = {
            "bias": self.config.bias_weight,
            "emotion": self.config.emotion_weight,
            "narrative": self.config.narrative_weight,
            "discourse": self.config.discourse_weight,
            "ideology": self.config.ideology_weight,
        }

        weighted_sum = 0.0
        total_weight = 0.0

        for section, weight in weights.items():
            values = list(profile.get(section, {}).values())

            if not values:
                continue

            section_mean = float(np.mean(values))

            weighted_sum += section_mean * weight
            total_weight += weight

        if total_weight <= EPS:
            return 0.0

        score = weighted_sum / (total_weight + EPS)

        return float(np.clip(score, 0.0, 1.0))


# =========================================================
# VECTORIZATION (STABLE)
# =========================================================

def bias_profile_vector(profile: Dict[str, Any]) -> np.ndarray:

    sections = {
        "bias": "framing",
        "emotion": "emotion_target",
        "narrative": "framing",
        "discourse": "discourse_coherence",
        "ideology": "ideology",
    }

    values: List[float] = []

    for section, schema_name in sections.items():
        keys = get_schema(schema_name)
        data = profile.get(section, {})

        for k in keys:
            values.append(float(data.get(k, 0.0)))

    if not values:
        raise ValueError("profile contains no values")

    return np.array(values, dtype=np.float32)