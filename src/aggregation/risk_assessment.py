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


def _validate_score(value: float) -> float:
    """Validate and clamp score to [0,1]."""

    if not isinstance(value, (int, float)):
        raise TypeError("Score must be numeric")

    return float(np.clip(value, 0.0, 1.0))


def score_to_risk_level(score: float) -> str:
    """
    Convert numeric score into risk level.

    Mapping:
        0.0–0.3 → LOW
        0.3–0.6 → MEDIUM
        0.6–1.0 → HIGH
    """

    score = _validate_score(score)

    if score < LOW_THRESHOLD:
        return "LOW"

    if score < MEDIUM_THRESHOLD:
        return "MEDIUM"

    return "HIGH"


def assess_risk_levels(scores: Dict[str, float]) -> Dict[str, str]:
    """
    Convert multiple numeric scores into categorical risk levels.
    """

    if not isinstance(scores, dict):
        raise ValueError("scores must be a dictionary")

    risk_levels: Dict[str, str] = {}

    for key, value in scores.items():

        if not isinstance(value, (int, float)):
            continue

        risk_levels[key] = score_to_risk_level(value)

    logger.info("Risk assessment completed")

    return risk_levels


def assess_truthlens_risks(scores: Dict[str, float]) -> Dict[str, str]:
    """
    Generate human-readable risk assessment for core TruthLens metrics.
    """

    if not isinstance(scores, dict):
        raise ValueError("scores must be a dictionary")

    output = {}

    if "truthlens_manipulation_risk" in scores:
        output["manipulation_risk"] = score_to_risk_level(
            scores["truthlens_manipulation_risk"]
        )

    if "truthlens_credibility_score" in scores:
        output["credibility_level"] = score_to_risk_level(
            scores["truthlens_credibility_score"]
        )

    if "truthlens_final_score" in scores:
        output["overall_truthlens_rating"] = score_to_risk_level(
            scores["truthlens_final_score"]
        )

    return output