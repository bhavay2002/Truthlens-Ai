"""
File Name: score_schema.py
Module: TruthLens AI - Aggregation Score Schema
Description:
    Defines standardized schemas for TruthLens scoring outputs to ensure
    consistent data structures across aggregation, evaluation, and reporting
    modules.

    Using TypedDict allows static type checking and prevents inconsistent
    score formats from propagating across the system.

Dependencies:
    typing

Inputs:
    TruthLens scoring dictionaries

Outputs:
    Typed schemas for scores, risks, and aggregation outputs
"""

from __future__ import annotations

from typing import TypedDict, Dict, Any


class TruthLensScoreSchema(TypedDict):
    """
    Core TruthLens numeric scoring schema.
    """

    truthlens_bias_score: float
    truthlens_emotion_score: float
    truthlens_narrative_score: float
    truthlens_discourse_score: float
    truthlens_graph_score: float
    truthlens_ideology_score: float

    truthlens_manipulation_risk: float
    truthlens_credibility_score: float
    truthlens_final_score: float


class TruthLensRiskSchema(TypedDict, total=False):
    """
    Risk level schema derived from numeric scores.
    """

    manipulation_risk: str
    credibility_level: str
    overall_truthlens_rating: str


class TruthLensAggregationOutputSchema(TypedDict):
    """
    Complete aggregation pipeline output schema.
    """

    scores: TruthLensScoreSchema
    raw_scores: TruthLensScoreSchema
    risks: TruthLensRiskSchema
    explanations: Dict[str, Any]
    analysis_modules: Dict[str, Any]
