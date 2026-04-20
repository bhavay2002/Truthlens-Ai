from __future__ import annotations

from typing import TypedDict, Dict, Any, Literal

import math
from pydantic import BaseModel, field_validator, ConfigDict


# =========================================================
# TYPED STRUCTURES (STATIC TYPE CHECKING ONLY)
# =========================================================

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


# =========================================================
# RUNTIME VALIDATION MODELS
# =========================================================

class TruthLensScoreModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

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
    def validate_score(cls, v: float) -> float:

        # reject bool explicitly
        if isinstance(v, bool):
            raise TypeError("Score must be numeric (not boolean)")

        try:
            fv = float(v)
        except Exception:
            raise TypeError("Score must be numeric")

        if not math.isfinite(fv):
            raise ValueError("Score must be finite (no NaN/inf)")

        if not (0.0 <= fv <= 1.0):
            raise ValueError(f"Score out of range [0,1]: {fv}")

        return fv


# ---------------------------------------------------------

_ALLOWED_LEVELS = {"LOW", "MEDIUM", "HIGH"}


class TruthLensRiskModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    manipulation_risk: Literal["LOW", "MEDIUM", "HIGH"] | None = None
    credibility_level: Literal["LOW", "MEDIUM", "HIGH"] | None = None
    overall_truthlens_rating: Literal["LOW", "MEDIUM", "HIGH"] | None = None

    @field_validator("*", mode="before")
    @classmethod
    def normalize_levels(cls, v):
        if v is None:
            return v

        if not isinstance(v, str):
            raise TypeError("Risk level must be string")

        v = v.upper()

        if v not in _ALLOWED_LEVELS:
            raise ValueError(f"Invalid risk level: {v}")

        return v


# ---------------------------------------------------------

class TruthLensAggregationOutputModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    scores: TruthLensScoreModel
    raw_scores: TruthLensScoreModel
    risks: TruthLensRiskModel
    explanations: Dict[str, Any]
    analysis_modules: Dict[str, Any]

    @field_validator("explanations", "analysis_modules")
    @classmethod
    def validate_dicts(cls, v):

        if not isinstance(v, dict):
            raise TypeError("Must be a dictionary")

        # shallow validation only (avoid heavy cost)
        for k in v.keys():
            if not isinstance(k, str):
                raise TypeError("Keys must be strings")

        return v