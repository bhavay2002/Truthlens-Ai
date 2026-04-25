from __future__ import annotations

import logging
from typing import Dict, Optional, List, Any

import numpy as np

logger = logging.getLogger(__name__)

EPS = 1e-12


# =========================================================
# CONFIG
# =========================================================

class RiskThresholds:
    def __init__(self, low: float = 0.3, medium: float = 0.6):
        if not (0.0 <= low < medium <= 1.0):
            raise ValueError("Invalid thresholds")

        self.low = low
        self.medium = medium


class RiskConfig:
    def __init__(
        self,
        default: RiskThresholds = RiskThresholds(),
        per_key: Optional[Dict[str, RiskThresholds]] = None,
        invert_keys: Optional[List[str]] = None,
        weights: Optional[Dict[str, float]] = None,
        uncertainty_penalty: float = 0.2,
    ):
        self.default = default
        self.per_key = per_key or {}
        self.invert_keys = set(invert_keys or [])
        self.weights = weights or {}
        self.uncertainty_penalty = uncertainty_penalty


# =========================================================
# UTILS
# =========================================================

def _validate(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("Invalid numeric value")

    if not np.isfinite(value):
        raise ValueError("Non-finite value")

    return float(np.clip(value, 0.0, 1.0))


def _entropy(probs):
    probs = np.asarray(probs)
    return -np.sum(probs * np.log(probs + EPS))


# =========================================================
# CORE RISK LOGIC
# =========================================================

def compute_risk_score(
    value: float,
    *,
    invert: bool,
    uncertainty: Optional[float],
    config: RiskConfig,
) -> float:

    value = _validate(value)

    if invert:
        value = 1.0 - value

    # uncertainty penalty
    if uncertainty is not None:
        value *= (1.0 - config.uncertainty_penalty * uncertainty)

    return float(np.clip(value, 0.0, 1.0))


def score_to_level(score: float, thresholds: RiskThresholds) -> str:

    if score < thresholds.low:
        return "LOW"
    elif score < thresholds.medium:
        return "MEDIUM"
    return "HIGH"


# =========================================================
# MAIN API (UPGRADED)
# =========================================================

def assess_risk_levels(
    scores: Dict[str, float],
    *,
    probabilities: Optional[Dict[str, np.ndarray]] = None,
    config: Optional[RiskConfig] = None,
    return_scores: bool = False,
) -> Dict[str, Any]:

    config = config or RiskConfig()

    results = {}
    continuous_scores = {}

    for key, value in scores.items():

        thresholds = config.per_key.get(key, config.default)
        invert = key in config.invert_keys
        weight = config.weights.get(key, 1.0)

        # -------------------------
        # uncertainty
        # -------------------------
        uncertainty = None

        if probabilities and key in probabilities:
            uncertainty = _entropy(probabilities[key])

        # -------------------------
        # score computation
        # -------------------------
        risk_score = compute_risk_score(
            value,
            invert=invert,
            uncertainty=uncertainty,
            config=config,
        )

        risk_score *= weight

        level = score_to_level(risk_score, thresholds)

        results[key] = level
        continuous_scores[key] = risk_score

    if return_scores:
        return {
            "levels": results,
            "scores": continuous_scores,
        }

    return results


# =========================================================
# BATCH SUPPORT
# =========================================================

def assess_batch(
    batch_scores: List[Dict[str, float]],
    *,
    config: Optional[RiskConfig] = None,
) -> List[Dict[str, str]]:

    return [
        assess_risk_levels(scores, config=config)
        for scores in batch_scores
    ]


# =========================================================
# TRUTHLENS WRAPPER (UPGRADED)
# =========================================================

TRUTHLENS_RISK_KEY_MAP = {
    "truthlens_manipulation_risk": "manipulation_risk",
    "truthlens_credibility_score": "credibility_level",
    "truthlens_final_score": "overall_truthlens_rating",
}


def assess_truthlens_risks(
    scores: Dict[str, float],
    *,
    probabilities: Optional[Dict[str, np.ndarray]] = None,
    config: Optional[RiskConfig] = None,
) -> Dict[str, Any]:

    config = config or RiskConfig(
        invert_keys=["truthlens_credibility_score"]
    )

    mapped = {}

    for k_in, k_out in TRUTHLENS_RISK_KEY_MAP.items():

        if k_in not in scores:
            continue

        result = assess_risk_levels(
            {k_in: scores[k_in]},
            probabilities=probabilities,
            config=config,
            return_scores=True,
        )

        mapped[k_out] = {
            "level": result["levels"][k_in],
            "score": result["scores"][k_in],
        }

    return mapped