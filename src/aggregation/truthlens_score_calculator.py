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
from typing import Dict, Any

import numpy as np

from src.aggregation.score_schema import TruthLensScoreSchema


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ScoreWeights:
    # Manipulation group — additive terms must sum to 1.0
    bias: float = 0.40
    emotion: float = 0.30
    narrative: float = 0.20
    analysis_influence_manipulation: float = 0.10

    # Credibility group — additive terms must sum to 1.0
    # credibility_bias_penalty is a standalone subtraction, not normalised
    # with the additive terms (see BUG-9 / weight_manager.py).
    discourse: float = 0.55
    graph: float = 0.35
    analysis_influence_credibility: float = 0.10
    credibility_bias_penalty: float = 0.20

    # Final composite — sum must equal 1.0
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


class TruthLensScoreCalculator:
    @staticmethod
    def _default_weights_dict() -> Dict[str, float]:
        sw = ScoreWeights()
        return {
            "bias": sw.bias,
            "emotion": sw.emotion,
            "narrative": sw.narrative,
            "discourse": sw.discourse,
            "graph": sw.graph,
            "credibility_bias_penalty": sw.credibility_bias_penalty,
            "final_credibility": sw.final_credibility,
            "final_manipulation": sw.final_manipulation,
            "final_ideology": sw.final_ideology,
            "analysis_influence_manipulation": sw.analysis_influence_manipulation,
            "analysis_influence_credibility": sw.analysis_influence_credibility,
        }

    def __init__(self, weights: Dict[str, float] | None = None) -> None:
        self.defaults = self._default_weights_dict()
        self.weights = weights or self.defaults.copy()
        self._validate_runtime_weights(self.weights)
        logger.info("TruthLensScoreCalculator initialized")

    @staticmethod
    def _validate_runtime_weights(w: Dict[str, float]) -> None:
        if not isinstance(w, dict):
            raise TypeError("weights must be a dictionary")
        for k, v in w.items():
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise TypeError(f"Weight '{k}' must be numeric (non-boolean)")
            if not np.isfinite(v) or v < 0:
                raise ValueError(f"Invalid weight '{k}': {v}")

    def compute_scores(
        self,
        profile: Dict[str, Any],
        *,
        weights: Dict[str, float] | None = None,
    ) -> TruthLensScoreSchema:
        if not isinstance(profile, dict):
            raise ValueError("profile must be a dictionary")

        if weights is not None:
            self._validate_runtime_weights(weights)

        bias_score = self._aggregate_section(profile.get("bias", {}))
        emotion_score = self._aggregate_section(profile.get("emotion", {}))
        narrative_score = self._aggregate_section(profile.get("narrative", {}))
        discourse_score = self._aggregate_section(profile.get("discourse", {}))
        graph_score = self._aggregate_section(profile.get("graph", {}))
        ideology_score = self._aggregate_section(profile.get("ideology", {}))
        analysis_score = self._aggregate_section(profile.get("analysis", {}))

        manipulation_risk = self._compute_manipulation_risk(
            bias_score, emotion_score, narrative_score, analysis_score, weights
        )
        credibility_score = self._compute_credibility(
            bias_score, discourse_score, graph_score, analysis_score, weights
        )
        truthlens_score = self._compute_final_score(
            credibility_score, manipulation_risk, ideology_score, weights
        )

        return {
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

    def _aggregate_section(self, section: Dict[str, Any]) -> float:
        if not isinstance(section, dict) or not section:
            return 0.0
        values = [
            float(v)
            for v in section.values()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        ]
        if not values:
            return 0.0
        return sum(values) / len(values)

    def _compute_manipulation_risk(
        self,
        bias_score: float,
        emotion_score: float,
        narrative_score: float,
        analysis_score: float,
        weights: Dict[str, float] | None = None,
    ) -> float:
        w = weights or self.weights
        risk = (
            w.get("bias", self.defaults["bias"]) * bias_score
            + w.get("emotion", self.defaults["emotion"]) * emotion_score
            + w.get("narrative", self.defaults["narrative"]) * narrative_score
            + w.get("analysis_influence_manipulation", self.defaults["analysis_influence_manipulation"]) * analysis_score
        )
        return float(np.clip(risk, 0.0, 1.0))

    def _compute_credibility(
        self,
        bias_score: float,
        discourse_score: float,
        graph_score: float,
        analysis_score: float,
        weights: Dict[str, float] | None = None,
    ) -> float:
        w = weights or self.weights
        credibility = (
            w.get("discourse", self.defaults["discourse"]) * discourse_score
            + w.get("graph", self.defaults["graph"]) * graph_score
            - w.get("credibility_bias_penalty", self.defaults["credibility_bias_penalty"]) * bias_score
            + w.get("analysis_influence_credibility", self.defaults["analysis_influence_credibility"]) * analysis_score
        )
        return float(np.clip(credibility, 0.0, 1.0))

    def _compute_final_score(
        self,
        credibility_score: float,
        manipulation_risk: float,
        ideology_score: float,
        weights: Dict[str, float] | None = None,
    ) -> float:
        w = weights or self.weights
        score = (
            w.get("final_credibility", self.defaults["final_credibility"]) * credibility_score
            + w.get("final_manipulation", self.defaults["final_manipulation"]) * (1.0 - manipulation_risk)
            + w.get("final_ideology", self.defaults["final_ideology"]) * (1.0 - ideology_score)
        )
        return float(np.clip(score, 0.0, 1.0))


def truthlens_score_vector(scores: TruthLensScoreSchema) -> np.ndarray:
    if not isinstance(scores, dict) or not scores:
        raise ValueError("scores must be a non-empty dictionary")

    try:
        missing = [k for k in SCORE_VECTOR_ORDER if k not in scores]
        if missing:
            raise KeyError(f"Missing score keys for vector conversion: {missing}")

        return np.asarray([float(scores[k]) for k in SCORE_VECTOR_ORDER], dtype=np.float32)
    except Exception as exc:
        logger.exception("TruthLens score vector conversion failed")
        raise RuntimeError("Failed to convert TruthLens scores") from exc
