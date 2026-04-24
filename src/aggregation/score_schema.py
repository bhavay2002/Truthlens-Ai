from __future__ import annotations

from typing import Dict, Any, Optional, Literal
import math

from pydantic import BaseModel, field_validator, ConfigDict


# =========================================================
# GLOBAL CONSTANTS
# =========================================================

_ALLOWED_LEVELS = {"LOW", "MEDIUM", "HIGH"}


# =========================================================
# BASE SCORE UNIT (PER TASK)
# =========================================================

class TaskScore(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    score: float
    confidence: Optional[float] = None

    @field_validator("score", "confidence")
    @classmethod
    def validate_numeric(cls, v):
        if v is None:
            return v

        if isinstance(v, bool):
            raise TypeError("Must be numeric (not boolean)")

        fv = float(v)

        if not math.isfinite(fv):
            raise ValueError("Must be finite")

        if not (0.0 <= fv <= 1.0):
            raise ValueError(f"Out of range [0,1]: {fv}")

        return fv


# =========================================================
# SCORE STRUCTURE (EXTENSIBLE)
# =========================================================

class TruthLensScoreModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    tasks: Dict[str, TaskScore]

    manipulation_risk: float
    credibility_score: float
    final_score: float

    @field_validator("tasks")
    @classmethod
    def validate_tasks(cls, v):
        if not v:
            raise ValueError("tasks cannot be empty")
        return v


# =========================================================
# RISK MODEL
# =========================================================

class TruthLensRiskModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    manipulation_risk: Optional[Literal["LOW", "MEDIUM", "HIGH"]] = None
    credibility_level: Optional[Literal["LOW", "MEDIUM", "HIGH"]] = None
    overall_truthlens_rating: Optional[Literal["LOW", "MEDIUM", "HIGH"]] = None

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


# =========================================================
# ATTRIBUTION STRUCTURE
# =========================================================

class TokenAttribution(BaseModel):
    token: str
    importance: float
    contribution: float
    direction: Literal["positive", "negative"]


class ExplanationSection(BaseModel):
    method: Literal[
        "integrated_gradients",
        "shap",
        "attention"
    ]
    top_features: list[str]
    attributions: list[TokenAttribution]


class ExplanationModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sections: Dict[str, ExplanationSection]


# =========================================================
# FINAL OUTPUT
# =========================================================

class TruthLensAggregationOutputModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str
    model_version: str

    scores: TruthLensScoreModel
    raw_scores: Dict[str, float]

    risks: TruthLensRiskModel
    explanations: ExplanationModel

    analysis_modules: Dict[str, Any]

    @field_validator("analysis_modules")
    @classmethod
    def validate_modules(cls, v):
        if not isinstance(v, dict):
            raise TypeError("Must be a dictionary")
        return v