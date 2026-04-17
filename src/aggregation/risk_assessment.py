"""
File Name: risk_assessment.py
Module: TruthLens AI - Aggregation Risk Assessment
Description:
    Converts numeric TruthLens scoring signals into interpretable risk
    levels for reporting and decision systems.

    This module maps continuous scores into categorical levels such as
    LOW, MEDIUM, and HIGH. It is used to make TruthLens outputs easier
    to interpret for dashboards, reports, and downstream applications.

Dependencies:
    logging
    typing
    numpy

Inputs:
    numeric TruthLens score values

Outputs:
    categorical risk levels
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np


logger = logging.getLogger(__name__)


LOW_THRESHOLD = 0.3
MEDIUM_THRESHOLD = 0.6

TRUTHLENS_RISK_KEY_MAP = {
    "truthlens_manipulation_risk": "manipulation_risk",
    "truthlens_credibility_score": "credibility_level",
    "truthlens_final_score": "overall_truthlens_rating",
}


def _validate_score(value: float) -> float:
    """Validate and clamp score to [0,1]."""

    if not isinstance(value, (int, float)):
        raise TypeError("Score must be numeric")

    if np.isnan(value) or np.isinf(value):
        raise ValueError(f"Invalid score value: {value}")

    return float(np.clip(value, 0.0, 1.0))


def score_to_risk_level(score: float) -> str:
    """
    Convert numeric score into risk level.

    Mapping (left-closed, right-open intervals):
        [0.0, 0.3) → LOW
        [0.3, 0.6) → MEDIUM
        [0.6, 1.0] → HIGH

    Notes:
        - 0.3 is classified as MEDIUM
        - 0.6 is classified as HIGH
    """

    score = _validate_score(score)

    if score < LOW_THRESHOLD:
        return "LOW"

    if score < MEDIUM_THRESHOLD:
        return "MEDIUM"

    return "HIGH"


def assess_risk_levels(
    scores: Dict[str, float],
    *,
    strict: bool = False,
    return_meta: bool = False,
) -> Dict[str, str] | Dict[str, object]:
    """
    Convert multiple numeric scores into categorical risk levels.
    """

    if not isinstance(scores, dict):
        raise ValueError("scores must be a dictionary")

    risk_levels: Dict[str, str] = {}
    skipped: list[str] = []

    for key, value in scores.items():

        if not isinstance(value, (int, float)) or isinstance(value, bool):
            msg = f"Non-numeric score for key '{key}': {type(value)}"
            if strict:
                raise TypeError(msg)
            logger.warning(msg)
            skipped.append(key)
            continue

        risk_levels[key] = score_to_risk_level(value)

    logger.info("Risk assessment completed")

    if return_meta:
        return {"risk_levels": risk_levels, "skipped_keys": skipped}

    return risk_levels


def assess_truthlens_risks(scores: Dict[str, float]) -> Dict[str, str]:
    """
    Generate human-readable risk assessment for core TruthLens metrics.
    """

    if not isinstance(scores, dict):
        raise ValueError("scores must be a dictionary")

    output: Dict[str, str] = {}
    for score_key, risk_key in TRUTHLENS_RISK_KEY_MAP.items():
        if score_key in scores:
            output[risk_key] = score_to_risk_level(scores[score_key])

    return output