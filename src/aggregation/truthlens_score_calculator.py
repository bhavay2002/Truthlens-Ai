"""
File Name: truthlens_score_calculator.py
Module: TruthLens Analysis - Final Scoring Engine
Description:
    Computes the final TruthLens credibility and bias scores by aggregating
    outputs from multiple analytical modules such as bias analysis, emotion
    analysis, narrative detection, discourse analysis, and graph analysis.

    The module normalizes signals from different subsystems and produces
    interpretable scoring metrics used for ranking, reporting, and downstream
    decision systems. Includes configurable weighting and normalization
    safeguards suitable for research and production environments.

Author:
    TruthLens Engineering Team

Date:
    2026-04-02

Dependencies:
    logging
    typing
    dataclasses
    numpy
    src.aggregation.score_schema

Inputs:
    Aggregated analysis outputs from TruthLens modules

Outputs:
    Final TruthLens scoring dictionary and numerical score vector
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Any, Iterable

import numpy as np

from src.aggregation.score_schema import TruthLensScoreSchema


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ScoreWeights:
    """Weights used for TruthLens score computation."""

    bias: float = 0.4
    emotion: float = 0.35
    narrative: float = 0.25

    discourse: float = 0.5
    graph: float = 0.3
    credibility_bias_penalty: float = 0.2

    final_credibility: float = 0.5
    final_manipulation: float = 0.3
    final_ideology: float = 0.2


class TruthLensScoreCalculator:
    """
    Aggregates subsystem outputs to compute final TruthLens scores.
    """

    def __init__(self, weights: ScoreWeights | None = None) -> None:
        self.weights = weights or ScoreWeights()
        logger.info("TruthLensScoreCalculator initialized")

    def compute_scores(self, profile: Dict[str, Any]) -> TruthLensScoreSchema:
        """
        Compute overall TruthLens scoring metrics.
        """

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

        scores: TruthLensScoreSchema = {
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

    def _aggregate_section(self, section: Dict[str, Any]) -> float:
        """
        Aggregate numeric values from a feature section.
        """

        if not isinstance(section, dict) or not section:
            return 0.0

        values: Iterable[float] = [
            float(v)
            for v in section.values()
            if isinstance(v, (int, float))
        ]

        values = list(values)

        if not values:
            return 0.0

        try:
            return float(np.mean(np.asarray(values, dtype=np.float32)))
        except Exception as exc:
            logger.exception("Section aggregation failed")
            raise RuntimeError("Feature aggregation failed") from exc

    def _compute_manipulation_risk(
        self,
        bias_score: float,
        emotion_score: float,
        narrative_score: float,
    ) -> float:
        """
        Estimate narrative manipulation risk.
        """

        risk = (
            self.weights.bias * bias_score
            + self.weights.emotion * emotion_score
            + self.weights.narrative * narrative_score
        )

        return float(np.clip(risk, 0.0, 1.0))

    def _compute_credibility(
        self,
        bias_score: float,
        discourse_score: float,
        graph_score: float,
    ) -> float:
        """
        Estimate credibility based on discourse structure and bias signals.
        """

        credibility = (
            self.weights.discourse * discourse_score
            + self.weights.graph * graph_score
            - self.weights.credibility_bias_penalty * bias_score
        )

        return float(np.clip(credibility, 0.0, 1.0))

    def _compute_final_score(
        self,
        credibility_score: float,
        manipulation_risk: float,
        ideology_score: float,
    ) -> float:
        """
        Compute the final TruthLens composite score.
        """

        score = (
            self.weights.final_credibility * credibility_score
            + self.weights.final_manipulation * (1.0 - manipulation_risk)
            + self.weights.final_ideology * (1.0 - ideology_score)
        )

        return float(np.clip(score, 0.0, 1.0))


def truthlens_score_vector(scores: TruthLensScoreSchema) -> np.ndarray:
    """
    Convert TruthLens scoring dictionary into numeric vector.
    """

    if not isinstance(scores, dict) or not scores:
        raise ValueError("scores must be a non-empty dictionary")

    try:
        vector = np.asarray(list(scores.values()), dtype=np.float32)
        return vector
    except Exception as exc:
        logger.exception("TruthLens score vector conversion failed")
        raise RuntimeError("Failed to convert TruthLens scores") from exc