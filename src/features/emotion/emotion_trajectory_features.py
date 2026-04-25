# src/features/emotion/emotion_trajectory_features.py

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from src.features.base.base_feature import BaseFeature, FeatureContext
from src.features.base.feature_registry import register_feature

from src.features.emotion.emotion_schema import (
    EMOTION_LABELS,
    WORD_TO_EMOTION,
)

logger = logging.getLogger(__name__)

EPS = 1e-8
MAX_CLIP = 1.0


# ------------------------------------------------------------
# Sentence splitter
# ------------------------------------------------------------

def _split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]


# ------------------------------------------------------------
# Lexicon vector scoring (CRITICAL UPGRADE)
# ------------------------------------------------------------

def _lexicon_vector(text: str) -> np.ndarray:

    tokens = re.findall(r"\b\w+\b", text.lower())

    vec = np.zeros(len(EMOTION_LABELS), dtype=np.float32)

    for t in tokens:
        emo = WORD_TO_EMOTION.get(t)
        if emo:
            idx = EMOTION_LABELS.index(emo)
            vec[idx] += 1

    total = vec.sum()
    if total > 0:
        vec /= total

    return vec


# ------------------------------------------------------------
# Feature extractor
# ------------------------------------------------------------

@dataclass
@register_feature
class EmotionTrajectoryFeatures(BaseFeature):

    name: str = "emotion_trajectory_features"
    group: str = "emotion"
    description: str = "Emotion trajectory modeling (vector-based)"

    # -----------------------------------------------------

    def _segment_vectors(self, text: str) -> List[np.ndarray]:

        sentences = _split_sentences(text)

        if not sentences:
            return [np.zeros(len(EMOTION_LABELS))]

        return [_lexicon_vector(s) for s in sentences]

    # -----------------------------------------------------

    def extract(self, context: FeatureContext) -> Dict[str, float]:

        text = context.text.strip()
        if not text:
            return self._empty()

        vectors = self._segment_vectors(text)

        if len(vectors) == 1:
            vectors.append(vectors[0])

        mat = np.stack(vectors)  # shape: (T, E)

        # -------------------------
        # Intensity trajectory
        # -------------------------

        intensities = np.linalg.norm(mat, axis=1)

        # -------------------------
        # Core stats
        # -------------------------

        mean_val = float(np.mean(intensities))
        std_val = float(np.std(intensities))

        # normalized slope
        x = np.linspace(0, 1, len(intensities))
        slope = float(np.polyfit(x, intensities, 1)[0])

        # peak (smoothed)
        peak_idx = int(np.argmax(intensities))
        peak_position = peak_idx / max(len(intensities) - 1, 1)

        volatility = float(np.mean(np.abs(np.diff(intensities))))

        range_val = float(np.max(intensities) - np.min(intensities))

        # -------------------------
        # NEW: Distribution shift
        # -------------------------

        shifts = [
            np.linalg.norm(mat[i] - mat[i - 1])
            for i in range(1, len(mat))
        ]

        shift_mean = float(np.mean(shifts)) if shifts else 0.0

        # -------------------------
        # NEW: Entropy over time
        # -------------------------

        entropies = []
        for v in mat:
            if v.sum() > 0:
                e = -np.sum(v * np.log(v + EPS))
                e /= np.log(len(v))
                entropies.append(e)
            else:
                entropies.append(0.0)

        entropy_mean = float(np.mean(entropies))

        # -------------------------
        # Output (bounded)
        # -------------------------

        return {
            "emotion_traj_mean": self._safe(mean_val),
            "emotion_traj_std": self._safe(std_val),
            "emotion_traj_slope": self._safe((slope + 1) / 2),  # normalize
            "emotion_traj_peak_position": self._safe(peak_position),
            "emotion_traj_volatility": self._safe(volatility),
            "emotion_traj_range": self._safe(range_val),

            # advanced signals
            "emotion_traj_shift_mean": self._safe(shift_mean),
            "emotion_traj_entropy_mean": self._safe(entropy_mean),
        }

    # -----------------------------------------------------

    def _empty(self) -> Dict[str, float]:
        return {
            "emotion_traj_mean": 0.0,
            "emotion_traj_std": 0.0,
            "emotion_traj_slope": 0.0,
            "emotion_traj_peak_position": 0.0,
            "emotion_traj_volatility": 0.0,
            "emotion_traj_range": 0.0,
            "emotion_traj_shift_mean": 0.0,
            "emotion_traj_entropy_mean": 0.0,
        }

    def _safe(self, v: float) -> float:
        if not np.isfinite(v):
            return 0.0
        return float(np.clip(v, 0.0, MAX_CLIP))