from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List

import numpy as np


logger = logging.getLogger(__name__)


# =========================================================
# FEATURE CONFIG (RESEARCH-GRADE)
# =========================================================

PRIMARY_FEATURES: Dict[str, set[str]] = {
    "bias": {"prediction"},
    "emotion": {"intensity"},
    "narrative": {"score"},
    "discourse": {"coherence"},
    "graph": {"consistency"},
    "ideology": {"score"},
    "analysis": {"confidence"},
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


# =========================================================
# CORE
# =========================================================

class TruthLensScoreCalculator:

    def __init__(
        self,
        *,
        weights: Optional[Dict[str, float]] = None,
        strict: bool = False,
    ) -> None:
        self.strict = strict
        self.defaults = asdict(ScoreWeights())
        self.weights = self._prepare_weights(weights or self.defaults)

        logger.info("[Calculator] Initialized")

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

    def _validate_weights(self, w: Dict[str, float]) -> None:
        for k, v in w.items():
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise TypeError(f"Weight '{k}' must be numeric")
            if not np.isfinite(v) or v < 0:
                raise ValueError(f"Invalid weight '{k}': {v}")

    def _normalize_group(self, w: Dict[str, float], keys: list[str]) -> None:
        total = sum(w.get(k, 0.0) for k in keys)
        if total <= 0:
            raise ValueError(f"Invalid weight group: {keys}")
        for k in keys:
            w[k] = w.get(k, 0.0) / total

    # =========================================================
    # PUBLIC API (SINGLE)
    # =========================================================
    def compute_scores(
        self,
        profile: Dict[str, Any],
        *,
        confidence: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:

        scores = {}

        sections = list(PRIMARY_FEATURES.keys())

        for sec in sections:
            val = self._aggregate(profile.get(sec), section_name=sec)

            if confidence and sec in confidence:
                val *= float(np.clip(confidence[sec], 0.0, 1.0))

            scores[sec] = val

        manipulation = self._manipulation(scores, self.weights)
        credibility = self._credibility(scores, self.weights)
        final_score = self._final(credibility, manipulation, scores.get("ideology", 0.0), self.weights)

        return {
            "truthlens_bias_score": scores["bias"],
            "truthlens_emotion_score": scores["emotion"],
            "truthlens_narrative_score": scores["narrative"],
            "truthlens_discourse_score": scores["discourse"],
            "truthlens_graph_score": scores["graph"],
            "truthlens_ideology_score": scores["ideology"],
            "truthlens_manipulation_risk": manipulation,
            "truthlens_credibility_score": credibility,
            "truthlens_final_score": final_score,
        }

    # =========================================================
    # PUBLIC API (BATCH)
    # =========================================================
    def compute_batch_scores(
        self,
        profiles: List[Dict[str, Any]],
        *,
        confidence_list: Optional[List[Dict[str, float]]] = None,
    ) -> List[Dict[str, float]]:

        results = []

        for i, profile in enumerate(profiles):
            confidence = None
            if confidence_list and i < len(confidence_list):
                confidence = confidence_list[i]

            result = self.compute_scores(profile, confidence=confidence)
            results.append(result)

        return results

    # =========================================================
    # AGGREGATION (ALPHA + PRIMARY FEATURES)
    # =========================================================
    def _aggregate(self, section_data: Any, *, section_name: str) -> float:

        if not isinstance(section_data, dict) or not section_data:
            return 0.0

        primary_keys = PRIMARY_FEATURES.get(section_name, set())
        alpha = SECTION_ALPHA.get(section_name, DEFAULT_ALPHA)

        primary_vals = []
        aux_vals = []

        for k, v in section_data.items():

            if not isinstance(v, (int, float)) or isinstance(v, bool):
                if self.strict:
                    raise TypeError(f"Invalid feature {k}")
                continue

            if not np.isfinite(v):
                if self.strict:
                    raise ValueError(f"Non-finite value {k}")
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

    # =========================================================
    # FINAL COMPONENTS
    # =========================================================
    def _manipulation(self, s: Dict[str, float], w: Dict[str, float]) -> float:
        return float(np.clip(
            w["bias"] * s["bias"] +
            w["emotion"] * s["emotion"] +
            w["narrative"] * s["narrative"] +
            w["analysis_influence_manipulation"] * s["analysis"],
            0.0, 1.0
        ))

    def _credibility(self, s: Dict[str, float], w: Dict[str, float]) -> float:
        positive = (
            w["discourse"] * s["discourse"] +
            w["graph"] * s["graph"] +
            w["analysis_influence_credibility"] * s["analysis"]
        )

        penalty = np.clip(w["credibility_bias_penalty"] * s["bias"], 0.0, 1.0)
        return float(np.clip(positive * (1.0 - penalty), 0.0, 1.0))

    def _final(self, c: float, m: float, i: float, w: Dict[str, float]) -> float:
        return float(np.clip(
            w["final_credibility"] * c +
            w["final_manipulation"] * (1.0 - m) +
            w["final_ideology"] * (1.0 - i),
            0.0, 1.0
        ))