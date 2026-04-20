from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Any

import numpy as np

from src.aggregation.score_schema import TruthLensScoreSchema


logger = logging.getLogger(__name__)

# =========================================================
# FEATURE IMPORTANCE CONFIG
# =========================================================

PRIMARY_FEATURES: Dict[str, set[str]] = {
    "bias": {"bias_prediction"},
    "emotion": {"emotion_intensity"},
    "narrative": {"narrative_score"},
    "discourse": {"coherence_score"},
    "graph": {"graph_consistency"},
    "ideology": {"ideology_score"},
    "analysis": {"analysis_confidence"},
}

SECTION_ALPHA: Dict[str, float] = {
    "bias": 0.80,
    "emotion": 0.75,
    "narrative": 0.75,
    "discourse": 0.70,
    "graph": 0.70,
    "ideology": 0.80,
    "analysis": 0.65,
}

DEFAULT_ALPHA = 0.75


# =========================================================
# WEIGHTS
# =========================================================

@dataclass(slots=True)
class ScoreWeights:
    bias: float = 0.40
    emotion: float = 0.30
    narrative: float = 0.20
    analysis_influence_manipulation: float = 0.10

    discourse: float = 0.55
    graph: float = 0.35
    analysis_influence_credibility: float = 0.10
    credibility_bias_penalty: float = 0.20

    final_credibility: float = 0.5
    final_manipulation: float = 0.3
    final_ideology: float = 0.2


SCORE_VECTOR_ORDER: tuple[str, ...] = (
    "truthlens_bias_score",
    "truthlens_emotion_score",
    "truthlens_narrative_score",
    "truthlens_discourse_score",
    "truthlens_graph_score",
    "truthlens_ideology_score",
    "truthlens_manipulation_risk",
    "truthlens_credibility_score",
    "truthlens_final_score",
)


# =========================================================
# CORE CALCULATOR
# =========================================================

class TruthLensScoreCalculator:

    def __init__(self, weights: Dict[str, float] | None = None) -> None:
        self.defaults = self._default_weights_dict()
        self.weights = self._prepare_weights(weights or self.defaults)

    def _default_weights_dict(self) -> Dict[str, float]:
        return vars(ScoreWeights())

    # -----------------------------
    # Weight Handling
    # -----------------------------

    def _prepare_weights(self, w: Dict[str, float]) -> Dict[str, float]:
        self._validate_weights(w)
        w = w.copy()

        self._normalize_group(w, ["bias", "emotion", "narrative", "analysis_influence_manipulation"])
        self._normalize_group(w, ["discourse", "graph", "analysis_influence_credibility"])
        self._normalize_group(w, ["final_credibility", "final_manipulation", "final_ideology"])

        return w

    @staticmethod
    def _validate_weights(w: Dict[str, float]) -> None:
        if not isinstance(w, dict):
            raise TypeError("weights must be a dictionary")

        for k, v in w.items():
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise TypeError(f"Weight '{k}' must be numeric")
            if not np.isfinite(v) or v < 0:
                raise ValueError(f"Invalid weight '{k}': {v}")

    @staticmethod
    def _normalize_group(w: Dict[str, float], keys: list[str]) -> None:
        total = sum(w.get(k, 0.0) for k in keys)
        if total <= 0:
            raise ValueError(f"Invalid weight group: {keys}")

        for k in keys:
            w[k] = w.get(k, 0.0) / total

    # -----------------------------
    # Public API
    # -----------------------------

    def compute_scores(
        self,
        profile: Dict[str, Any],
        *,
        weights: Dict[str, float] | None = None,
    ) -> TruthLensScoreSchema:

        if not isinstance(profile, dict):
            raise ValueError("profile must be a dictionary")

        w = self._prepare_weights(weights) if weights else self.weights

        # ✅ FIXED CALLS
        bias = self._aggregate("bias", profile.get("bias"))
        emotion = self._aggregate("emotion", profile.get("emotion"))
        narrative = self._aggregate("narrative", profile.get("narrative"))
        discourse = self._aggregate("discourse", profile.get("discourse"))
        graph = self._aggregate("graph", profile.get("graph"))
        ideology = self._aggregate("ideology", profile.get("ideology"))
        analysis = self._aggregate("analysis", profile.get("analysis"))

        manipulation = self._manipulation(bias, emotion, narrative, analysis, w)
        credibility = self._credibility(bias, discourse, graph, analysis, w)
        final_score = self._final(credibility, manipulation, ideology, w)

        return {
            "truthlens_bias_score": bias,
            "truthlens_emotion_score": emotion,
            "truthlens_narrative_score": narrative,
            "truthlens_discourse_score": discourse,
            "truthlens_graph_score": graph,
            "truthlens_ideology_score": ideology,
            "truthlens_manipulation_risk": manipulation,
            "truthlens_credibility_score": credibility,
            "truthlens_final_score": final_score,
        }

    # -----------------------------
    # Core Computations
    # -----------------------------

    def _aggregate(self, section_name: str, section: Any) -> float:
        if not isinstance(section, dict) or not section:
            return 0.0

        primary_keys = PRIMARY_FEATURES.get(section_name, set())
        alpha = SECTION_ALPHA.get(section_name, DEFAULT_ALPHA)

        primary_vals = []
        aux_vals = []

        for k, v in section.items():
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                continue

            v = float(v)
            if not np.isfinite(v):
                continue

            if k in primary_keys:
                primary_vals.append(v)
            else:
                aux_vals.append(v)

        primary_mean = np.mean(primary_vals) if primary_vals else None
        aux_mean = np.mean(aux_vals) if aux_vals else None

        if primary_mean is not None and aux_mean is not None:
            score = alpha * primary_mean + (1 - alpha) * aux_mean
        elif primary_mean is not None:
            score = primary_mean
        elif aux_mean is not None:
            score = aux_mean
        else:
            return 0.0

        return float(np.clip(score, 0.0, 1.0))

    def _manipulation(self, b: float, e: float, n: float, a: float, w: Dict[str, float]) -> float:
        score = (
            w["bias"] * b +
            w["emotion"] * e +
            w["narrative"] * n +
            w["analysis_influence_manipulation"] * a
        )
        return float(np.clip(score, 0.0, 1.0))

    def _credibility(self, b: float, d: float, g: float, a: float, w: Dict[str, float]) -> float:
        positive = (
            w["discourse"] * d +
            w["graph"] * g +
            w["analysis_influence_credibility"] * a
        )

        penalty = np.clip(w["credibility_bias_penalty"] * b, 0.0, 1.0)
        score = positive * (1.0 - penalty)

        return float(np.clip(score, 0.0, 1.0))

    def _final(self, c: float, m: float, i: float, w: Dict[str, float]) -> float:
        score = (
            w["final_credibility"] * c +
            w["final_manipulation"] * (1.0 - m) +
            w["final_ideology"] * (1.0 - i)
        )
        return float(np.clip(score, 0.0, 1.0))


# =========================================================
# VECTOR
# =========================================================

def truthlens_score_vector(scores: TruthLensScoreSchema) -> np.ndarray:
    if not isinstance(scores, dict) or not scores:
        raise ValueError("scores must be a non-empty dictionary")

    missing = [k for k in SCORE_VECTOR_ORDER if k not in scores]
    if missing:
        raise KeyError(f"Missing keys: {missing}")

    return np.asarray(
        [float(scores[k]) for k in SCORE_VECTOR_ORDER],
        dtype=np.float32,
    )