from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from src.analysis.feature_schema import get_schema

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

@dataclass(slots=True)
class BiasProfileConfig:

    bias_weight: float = 1.0
    emotion_weight: float = 1.0
    narrative_weight: float = 1.0
    discourse_weight: float = 1.0
    ideology_weight: float = 0.6

    normalize_values: bool = True
    normalization_method: str = "minmax"  # minmax | zscore

    clip_values: bool = True
    clip_range: tuple[float, float] = (0.0, 1.0)

    global_normalization: bool = True


# ---------------------------------------------------------
# Builder
# ---------------------------------------------------------

class BiasProfileBuilder:

    PROFILE_SECTIONS = (
        "bias",
        "emotion",
        "narrative",
        "discourse",
        "ideology",
    )

    def __init__(self, config: BiasProfileConfig | None = None) -> None:
        self.config = config or BiasProfileConfig()
        logger.info("BiasProfileBuilder initialized")

    # -----------------------------------------------------

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
                "weights": {
                    "bias": self.config.bias_weight,
                    "emotion": self.config.emotion_weight,
                    "narrative": self.config.narrative_weight,
                    "discourse": self.config.discourse_weight,
                    "ideology": self.config.ideology_weight,
                },
            },
            "bias": self._process_section(bias),
            "emotion": self._process_section(emotion),
            "narrative": self._process_section(narrative),
            "discourse": self._process_section(discourse),
            "ideology": self._process_section(ideology),
        }

        # 🔥 GLOBAL NORMALIZATION (CRITICAL)
        if self.config.global_normalization:
            profile = self._global_normalize(profile)

        profile["metrics"] = self._compute_profile_metrics(profile)
        profile["bias_score"] = self._compute_bias_score(profile)

        return profile

    # -----------------------------------------------------
    # Processing Pipeline
    # -----------------------------------------------------

    def _process_section(self, data: Dict[str, Any]) -> Dict[str, float]:

        data = self._sanitize_numeric_dict(data)

        if self.config.normalize_values:
            data = self._normalize_values(data)

        if self.config.clip_values:
            data = self._clip_values(data)

        return data

    # -----------------------------------------------------
    # Sanitization (STRONG)
    # -----------------------------------------------------

    def _sanitize_numeric_dict(self, data: Dict[str, Any]) -> Dict[str, float]:

        if not isinstance(data, dict):
            raise ValueError("Input must be dictionary")

        cleaned = {}

        for k, v in data.items():

            if isinstance(v, (int, float, np.number)):
                v = float(v)

                if np.isnan(v) or np.isinf(v):
                    v = 0.0
            else:
                v = 0.0

            cleaned[k] = v

        return cleaned

    # -----------------------------------------------------
    # Normalization
    # -----------------------------------------------------

    def _normalize_values(self, data: Dict[str, float]) -> Dict[str, float]:

        if not data:
            return data

        values = np.array(list(data.values()), dtype=np.float32)

        if self.config.normalization_method == "zscore":

            mean = values.mean()
            std = values.std()

            if std < 1e-9:
                return data

            norm = (values - mean) / std

        else:  # minmax

            min_v = values.min()
            max_v = values.max()

            if max_v - min_v < 1e-9:
                return data

            norm = (values - min_v) / (max_v - min_v)

        return {k: float(v) for k, v in zip(data.keys(), norm)}

    # -----------------------------------------------------
    # Global Normalization (NEW)
    # -----------------------------------------------------

    def _global_normalize(self, profile: Dict[str, Any]) -> Dict[str, Any]:

        all_values = []

        for section in self.PROFILE_SECTIONS:
            all_values.extend(profile[section].values())

        if not all_values:
            return profile

        arr = np.array(all_values, dtype=np.float32)

        min_v = arr.min()
        max_v = arr.max()

        if max_v - min_v < 1e-9:
            return profile

        def normalize_section(section_data):
            return {
                k: float((v - min_v) / (max_v - min_v))
                for k, v in section_data.items()
            }

        for section in self.PROFILE_SECTIONS:
            profile[section] = normalize_section(profile[section])

        return profile

    # -----------------------------------------------------
    # Clipping
    # -----------------------------------------------------

    def _clip_values(self, data: Dict[str, float]) -> Dict[str, float]:

        low, high = self.config.clip_range

        return {
            k: float(np.clip(v, low, high))
            for k, v in data.items()
        }

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

    def _compute_profile_metrics(self, profile: Dict[str, Any]) -> Dict[str, float]:

        ideology_vals = np.array(
            list(profile["ideology"].values()),
            dtype=np.float32,
        )

        if ideology_vals.size > 0:

            ideology_vals = np.clip(ideology_vals, 0.0, None)
            total = float(ideology_vals.sum())

            if total > 0:
                p = ideology_vals / total

                entropy = float(-np.sum(p * np.log(p + 1e-9)))
                dominance = float(np.max(p))
            else:
                entropy = 0.0
                dominance = 0.0

        else:
            entropy = 0.0
            dominance = 0.0

        bias_vals = np.array(
            list(profile["bias"].values()),
            dtype=np.float32,
        )

        variance = float(np.var(bias_vals)) if bias_vals.size > 0 else 0.0

        return {
            "bias_variance": variance,
            "ideology_entropy": entropy,
            "ideology_dominance": dominance,
        }

    # -----------------------------------------------------
    # Bias Score (CALIBRATED)
    # -----------------------------------------------------

    def _compute_bias_score(self, profile: Dict[str, Any]) -> float:

        weighted_values: List[float] = []

        def add(section, weight):
            weighted_values.extend(v * weight for v in section.values())

        add(profile["bias"], self.config.bias_weight)
        add(profile["emotion"], self.config.emotion_weight)
        add(profile["narrative"], self.config.narrative_weight)
        add(profile["discourse"], self.config.discourse_weight)
        add(profile["ideology"], self.config.ideology_weight)

        if not weighted_values:
            return 0.0

        score = float(np.mean(weighted_values))

        return float(np.clip(score, 0.0, 1.0))

# ---------------------------------------------------------
# Vector Conversion (SCHEMA-STABLE)
# ---------------------------------------------------------

def bias_profile_vector(profile: Dict[str, Any]) -> np.ndarray:

    if not isinstance(profile, dict):
        raise ValueError("profile must be dictionary")

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
        raise ValueError("profile contains no numeric values")

    return np.array(values, dtype=np.float32)