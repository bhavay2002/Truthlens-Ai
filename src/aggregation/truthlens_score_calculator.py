from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional

import numpy as np

logger = logging.getLogger(__name__)
EPS = 1e-12


# =========================================================
# DEFAULT WEIGHTS (normalised per group)
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
# WEIGHT GROUPS (must sum to 1 within each group)
# =========================================================

_MANIPULATION_KEYS = ("bias", "emotion", "narrative", "analysis_influence_manipulation")
_CREDIBILITY_KEYS  = ("discourse", "graph", "analysis_influence_credibility")
_FINAL_KEYS        = ("final_credibility", "final_manipulation", "final_ideology")


def _renorm_group(weights: Dict[str, float], keys) -> None:
    total = sum(weights[k] for k in keys if k in weights) + EPS
    for k in keys:
        if k in weights:
            weights[k] = float(weights[k] / total)


# =========================================================
# CORE
# =========================================================

class TruthLensScoreCalculator:

    def __init__(
        self,
        *,
        weights: Optional[Dict[str, float]] = None,
        uncertainty_penalty: float = 0.2,
    ):
        self._base_weights: Dict[str, float] = asdict(ScoreWeights()) if weights is None else weights
        self.uncertainty_penalty = float(np.clip(uncertainty_penalty, 0.0, 1.0))

    # =====================================================
    # MAIN — accepts optional adaptive weights from WeightManager
    # =====================================================

    def compute_scores(
        self,
        profile: Dict[str, Any],
        *,
        confidence: Optional[Dict[str, float]] = None,
        entropy: Optional[Dict[str, float]] = None,
        explanation_scores: Optional[Dict[str, float]] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:

        # Use adaptive weights if provided, otherwise fall back to base
        w = dict(self._base_weights)
        if weights:
            w.update(weights)
            _renorm_group(w, _MANIPULATION_KEYS)
            _renorm_group(w, _CREDIBILITY_KEYS)
            _renorm_group(w, _FINAL_KEYS)

        section_scores: Dict[str, float] = {}
        section_debug: Dict[str, Any] = {}

        # Extract graph signal once (vectorized lookup)
        graph_signal = float(
            profile.get("graph", {}).get("graph_centrality_mean", 0.0)
        )

        for section, data in profile.items():

            base_val = self._aggregate(data)

            # graph cross-signal (capped at 0.1 contribution)
            val = base_val + 0.1 * graph_signal

            debug_info: Dict[str, Any] = {
                "base": base_val,
                "graph_signal": graph_signal,
                "graph_influence": 0.1 * graph_signal,
            }

            # confidence scaling
            if confidence and section in confidence:
                conf = float(np.clip(confidence[section], 0.0, 1.0))
                val *= conf
                debug_info["confidence"] = conf

            # uncertainty penalty (guard against NaN entropy)
            if entropy and section in entropy:
                raw_ent = entropy[section]
                if not np.isfinite(raw_ent):
                    raw_ent = 0.0
                ent = float(np.clip(raw_ent, 0.0, 1.0))
                penalty = float(np.clip(1.0 - self.uncertainty_penalty * ent, 0.0, 1.0))
                val *= penalty
                debug_info["entropy"] = ent
                debug_info["uncertainty_penalty"] = penalty

            # explanation alignment (equal blend)
            if explanation_scores and section in explanation_scores:
                exp_score = float(np.clip(explanation_scores[section], 0.0, 1.0))
                val = 0.5 * val + 0.5 * exp_score
                debug_info["explanation_score"] = exp_score

            final_val = float(np.clip(val, 0.0, 1.0))
            section_scores[section] = final_val
            section_debug[section] = {**debug_info, "final": final_val}

        # =====================================================
        # COMPOSITE SCORES
        # =====================================================

        manipulation = self._manipulation(section_scores, w)
        credibility  = self._credibility(section_scores, w)
        final_score  = self._final(credibility, manipulation, section_scores.get("ideology", 0.0), w)

        return {
            "section_scores": section_scores,
            "manipulation_risk": manipulation,
            "credibility_score": credibility,
            "final_score": final_score,
            "debug": {
                "inputs": profile,
                "confidence": confidence,
                "entropy": entropy,
                "explanation_scores": explanation_scores,
                "graph_signal": graph_signal,
                "section_breakdown": section_debug,
                "weights_used": w,
            },
        }

    # =====================================================
    # AGGREGATION — vectorized mean over finite values
    # =====================================================

    def _aggregate(self, section_data: Any) -> float:

        if not isinstance(section_data, dict):
            return 0.0

        vals = np.array(
            [v for v in section_data.values()
             if isinstance(v, (int, float)) and np.isfinite(v)],
            dtype=np.float64,
        )

        if vals.size == 0:
            return 0.0

        return float(np.clip(np.mean(vals), 0.0, 1.0))

    # =====================================================
    # COMPONENT SCORES
    # =====================================================

    def _manipulation(self, s: Dict[str, float], w: Dict[str, float]) -> float:
        val = (
            w.get("bias", 0.0) * s.get("bias", 0.0) +
            w.get("emotion", 0.0) * s.get("emotion", 0.0) +
            w.get("narrative", 0.0) * s.get("narrative", 0.0) +
            w.get("analysis_influence_manipulation", 0.0) * s.get("analysis", 0.0)
        )
        return float(np.clip(val, 0.0, 1.0))

    def _credibility(self, s: Dict[str, float], w: Dict[str, float]) -> float:
        positive = (
            w.get("discourse", 0.0) * s.get("discourse", 0.0) +
            w.get("graph", 0.0) * s.get("graph", 0.0) +
            w.get("analysis_influence_credibility", 0.0) * s.get("analysis", 0.0)
        )
        penalty = float(np.clip(w.get("credibility_bias_penalty", 0.2) * s.get("bias", 0.0), 0.0, 1.0))
        return float(np.clip(positive * (1.0 - penalty), 0.0, 1.0))

    def _final(self, c: float, m: float, i: float, w: Dict[str, float]) -> float:
        val = (
            w.get("final_credibility", 0.5) * c +
            w.get("final_manipulation", 0.3) * (1.0 - m) +
            w.get("final_ideology", 0.2) * (1.0 - i)
        )
        return float(np.clip(val, 0.0, 1.0))
