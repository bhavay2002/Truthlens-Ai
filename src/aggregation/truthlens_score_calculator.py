"""
File Name: truthlens_score_calculator.py
Module: TruthLens Analysis - Final Scoring Engine
Description:
    Computes the final TruthLens credibility and bias scores by aggregating
    outputs from multiple analytical modules such as bias analysis, emotion
    analysis, narrative detection, discourse analysis, and graph analysis.
    The module normalizes signals from different subsystems and produces
    interpretable scoring metrics used for ranking, reporting, and downstream
    decision systems.

Dependencies:
    logging
    typing
    numpy

Inputs:
    Aggregated analysis outputs from TruthLens modules

Outputs:
    Final TruthLens scoring dictionary and numerical score vector
"""

import logging
from typing import Dict, Any

import numpy as np


logger = logging.getLogger(__name__)


class TruthLensScoreCalculator:
    """
    Aggregates subsystem outputs to compute final TruthLens scores.
    """

    def __init__(self) -> None:
        """Initialize TruthLens score calculator."""
        logger.info("TruthLensScoreCalculator initialized")

    def compute_scores(self, profile: Dict[str, Any]) -> Dict[str, float]:
        """Compute overall TruthLens scoring metrics."""

        if not isinstance(profile, dict):
            raise ValueError("profile must be a dictionary")

        bias_score = self._aggregate_section(profile.get("bias", {}))
        emotion_score = self._aggregate_section(profile.get("emotion", {}))
        narrative_score = self._aggregate_section(profile.get("narrative", {}))
        discourse_score = self._aggregate_section(profile.get("discourse", {}))
        graph_score = self._aggregate_section(profile.get("graph", {}))
        ideology_score = self._aggregate_section(profile.get("ideology", {}))

        manipulation_risk = self._compute_manipulation_risk(
            bias_score,
            emotion_score,
            narrative_score,
        )

        credibility_score = self._compute_credibility(
            bias_score,
            discourse_score,
            graph_score,
        )

        truthlens_score = self._compute_final_score(
            credibility_score,
            manipulation_risk,
            ideology_score,
        )

        scores = {
            "truthlens_bias_score": float(bias_score),
            "truthlens_emotion_score": float(emotion_score),
            "truthlens_narrative_score": float(narrative_score),
            "truthlens_discourse_score": float(discourse_score),
            "truthlens_graph_score": float(graph_score),
            "truthlens_ideology_score": float(ideology_score),
            "truthlens_manipulation_risk": float(manipulation_risk),
            "truthlens_credibility_score": float(credibility_score),
            "truthlens_final_score": float(truthlens_score),
        }

        return scores

    def _aggregate_section(self, section: Dict[str, float]) -> float:
        """Aggregate numeric values from a feature section."""

        if not isinstance(section, dict) or not section:
            return 0.0

        values = [
            float(v)
            for v in section.values()
            if isinstance(v, (int, float))
        ]

        if not values:
            return 0.0

        try:
            return float(np.mean(np.array(values, dtype=np.float32)))
        except Exception as exc:
            logger.exception("Section aggregation failed")
            raise RuntimeError("Feature aggregation failed") from exc

    def _compute_manipulation_risk(
        self,
        bias_score: float,
        emotion_score: float,
        narrative_score: float,
    ) -> float:
        """Estimate narrative manipulation risk."""

        risk = (
            0.4 * bias_score +
            0.35 * emotion_score +
            0.25 * narrative_score
        )

        return float(min(max(risk, 0.0), 1.0))

    def _compute_credibility(
        self,
        bias_score: float,
        discourse_score: float,
        graph_score: float,
    ) -> float:
        """Estimate credibility based on discourse structure and bias signals."""

        credibility = (
            0.5 * discourse_score +
            0.3 * graph_score -
            0.2 * bias_score
        )

        credibility = max(min(credibility, 1.0), 0.0)

        return float(credibility)

    def _compute_final_score(
        self,
        credibility_score: float,
        manipulation_risk: float,
        ideology_score: float,
    ) -> float:
        """Compute the final TruthLens composite score."""

        score = (
            0.5 * credibility_score +
            0.3 * (1 - manipulation_risk) +
            0.2 * (1 - ideology_score)
        )

        score = max(min(score, 1.0), 0.0)

        return float(score)


def truthlens_score_vector(scores: Dict[str, float]) -> np.ndarray:
    """Convert TruthLens scoring dictionary into numeric vector."""

    if not isinstance(scores, dict) or not scores:
        raise ValueError("scores must be a non-empty dictionary")

    try:
        vector = np.array(list(scores.values()), dtype=np.float32)
        return vector
    except Exception as exc:
        logger.exception("TruthLens score vector conversion failed")
        raise RuntimeError("Failed to convert TruthLens scores") from exc