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

from typing import TypedDict, Dict, Any, Literal

from pydantic import BaseModel, field_validator


class TruthLensScoreSchema(TypedDict):
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
    manipulation_risk: str
    credibility_level: str
    overall_truthlens_rating: str


class TruthLensAggregationOutputSchema(TypedDict):
    scores: TruthLensScoreSchema
    raw_scores: TruthLensScoreSchema
    risks: TruthLensRiskSchema
    explanations: Dict[str, Any]
    analysis_modules: Dict[str, Any]


class TruthLensScoreModel(BaseModel):
    truthlens_bias_score: float
    truthlens_emotion_score: float
    truthlens_narrative_score: float
    truthlens_discourse_score: float
    truthlens_graph_score: float
    truthlens_ideology_score: float
    truthlens_manipulation_risk: float
    truthlens_credibility_score: float
    truthlens_final_score: float

    @field_validator("*")
    @classmethod
    def validate_range(cls, v: float) -> float:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise TypeError("Score must be numeric (non-boolean)")
        fv = float(v)
        if not (0.0 <= fv <= 1.0):
            raise ValueError(f"Score out of range [0,1]: {fv}")
        return fv


class TruthLensRiskModel(BaseModel):
    manipulation_risk: Literal["LOW", "MEDIUM", "HIGH"] | None = None
    credibility_level: Literal["LOW", "MEDIUM", "HIGH"] | None = None
    overall_truthlens_rating: Literal["LOW", "MEDIUM", "HIGH"] | None = None


class TruthLensAggregationOutputModel(BaseModel):
    scores: TruthLensScoreModel
    raw_scores: TruthLensScoreModel
    risks: TruthLensRiskModel
    explanations: Dict[str, Any]
    analysis_modules: Dict[str, Any]
