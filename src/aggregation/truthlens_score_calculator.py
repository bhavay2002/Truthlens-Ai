from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional

import numpy as np

logger = logging.getLogger(__name__)
EPS = 1e-12


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
        uncertainty_penalty: float = 0.2,
    ):
        self.weights = asdict(ScoreWeights()) if weights is None else weights
        self.uncertainty_penalty = uncertainty_penalty

    # =====================================================
    # MAIN
    # =====================================================

    def compute_scores(
        self,
        profile: Dict[str, Any],
        *,
        confidence: Optional[Dict[str, float]] = None,
        entropy: Optional[Dict[str, float]] = None,
        explanation_scores: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:

        section_scores = {}
        section_debug = {}

        # 🔥 Extract graph signal once
        graph_signal = float(
            profile.get("graph", {}).get("graph_centrality_mean", 0.0)
        )

        for section, data in profile.items():

            base_val = self._aggregate(data)

            # 🔥 Inject graph influence
            val = base_val + 0.1 * graph_signal

            debug_info = {
                "base": base_val,
                "graph_signal": graph_signal,
                "graph_influence": 0.1 * graph_signal,
            }

            # -------------------------
            # confidence scaling
            # -------------------------
            if confidence and section in confidence:
                val *= confidence[section]
                debug_info["confidence"] = confidence[section]

            # -------------------------
            # uncertainty penalty
            # -------------------------
            if entropy and section in entropy:
                penalty = (1.0 - self.uncertainty_penalty * entropy[section])
                val *= penalty
                debug_info["entropy"] = entropy[section]
                debug_info["uncertainty_penalty"] = penalty

            # -------------------------
            # explanation alignment
            # -------------------------
            if explanation_scores and section in explanation_scores:
                val = 0.5 * val + 0.5 * explanation_scores[section]
                debug_info["explanation_score"] = explanation_scores[section]

            final_val = float(np.clip(val, 0.0, 1.0))

            section_scores[section] = final_val
            section_debug[section] = {
                **debug_info,
                "final": final_val,
            }

        # =====================================================
        # FINAL COMPONENTS
        # =====================================================

        manipulation = self._manipulation(section_scores)
        credibility = self._credibility(section_scores)

        final_score = self._final(
            credibility,
            manipulation,
            section_scores.get("ideology", 0.0),
        )

        return {
            "section_scores": section_scores,
            "manipulation_risk": manipulation,
            "credibility_score": credibility,
            "final_score": final_score,

            # 🔥 DEBUG BLOCK
            "debug": {
                "inputs": profile,
                "confidence": confidence,
                "entropy": entropy,
                "explanation_scores": explanation_scores,
                "graph_signal": graph_signal,
                "section_breakdown": section_debug,
            }
        }

    # =====================================================
    # AGGREGATION
    # =====================================================

    def _aggregate(self, section_data: Any) -> float:

        if not isinstance(section_data, dict):
            return 0.0

        vals = [
            v for v in section_data.values()
            if isinstance(v, (int, float)) and np.isfinite(v)
        ]

        if not vals:
            return 0.0

        return float(np.clip(np.mean(vals), 0.0, 1.0))

    # =====================================================
    # COMPONENTS
    # =====================================================

    def _manipulation(self, s: Dict[str, float]) -> float:
        w = self.weights
        return float(np.clip(
            w["bias"] * s.get("bias", 0) +
            w["emotion"] * s.get("emotion", 0) +
            w["narrative"] * s.get("narrative", 0) +
            w["analysis_influence_manipulation"] * s.get("analysis", 0),
            0.0, 1.0
        ))

    def _credibility(self, s: Dict[str, float]) -> float:
        w = self.weights

        positive = (
            w["discourse"] * s.get("discourse", 0) +
            w["graph"] * s.get("graph", 0) +
            w["analysis_influence_credibility"] * s.get("analysis", 0)
        )

        penalty = w["credibility_bias_penalty"] * s.get("bias", 0)

        return float(np.clip(positive * (1 - penalty), 0.0, 1.0))

    def _final(self, c, m, i):
        w = self.weights
        return float(np.clip(
            w["final_credibility"] * c +
            w["final_manipulation"] * (1 - m) +
            w["final_ideology"] * (1 - i),
            0.0, 1.0
        ))