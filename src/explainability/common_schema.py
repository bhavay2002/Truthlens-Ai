from __future__ import annotations

from typing import List, Dict, Optional, Literal, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict
import math
import numpy as np

EPS = 1e-12


# =========================================================
# BASE TOKEN UNIT
# =========================================================

class TokenImportance(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    token: str
    importance: float = Field(..., ge=0.0, le=1.0)

    @field_validator("token")
    @classmethod
    def validate_token(cls, v):
        if not v.strip():
            raise ValueError("token must be non-empty")
        return v

    @field_validator("importance")
    @classmethod
    def validate_importance(cls, v):
        if not math.isfinite(v):
            raise ValueError("importance must be finite")
        return float(v)


# =========================================================
# METHOD OUTPUT
# =========================================================

class ExplanationOutput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    method: Literal[
        "shap", "lime", "attention",
        "integrated_gradients", "propaganda", "custom"
    ]

    tokens: List[str]
    importance: List[float]
    structured: List[TokenImportance]

    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    entropy: Optional[float] = None
    raw: Optional[Any] = None

    # -------------------------
    # VALIDATION
    # -------------------------

    @field_validator("importance", mode="before")
    @classmethod
    def normalize_importance(cls, v):
        if not v:
            return v
        arr = np.array(v, dtype=float)

        if np.any(~np.isfinite(arr)):
            raise ValueError("importance must be finite")

        arr = np.abs(arr)
        total = float(np.sum(arr))

        if total <= 0:
            return [0.0] * len(arr)

        arr = arr / (total + EPS)

        return arr.tolist()

    @field_validator("structured")
    @classmethod
    def validate_structured(cls, v, info):
        tokens = info.data.get("tokens", [])
        importance = info.data.get("importance", [])

        if len(tokens) != len(importance):
            raise ValueError("tokens and importance must align")

        if len(v) != len(tokens):
            raise ValueError("structured must align with tokens")

        return v


# =========================================================
# AGGREGATED OUTPUT
# =========================================================

class AggregatedExplanation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    tokens: List[str]
    final_token_importance: List[float]

    structured: List[TokenImportance]  # 🔥 NEW

    method_weights: Dict[str, float]

    confidence_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    agreement_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    @field_validator("final_token_importance")
    @classmethod
    def validate_scores(cls, v):
        if not v:
            raise ValueError("final_token_importance cannot be empty")
        return v

    @field_validator("structured")
    @classmethod
    def validate_structured(cls, v, info):
        tokens = info.data.get("tokens", [])

        if len(v) != len(tokens):
            raise ValueError("structured must align with tokens")

        return v


# =========================================================
# METRICS
# =========================================================

class ConsistencyMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shap_vs_lime: Optional[float] = None
    shap_vs_attention: Optional[float] = None
    ig_vs_lime: Optional[float] = None
    ig_vs_attention: Optional[float] = None
    shap_vs_ig: Optional[float] = None

    overall_consistency: Optional[float] = None


class ExplanationMetricsOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    faithfulness: float
    comprehensiveness: float
    sufficiency: float
    deletion_score: float
    insertion_score: float

    overall_score: Optional[float] = None


# =========================================================
# FINAL OUTPUT
# =========================================================

class ExplainabilityResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prediction: Dict[str, Any]

    shap_explanation: Optional[ExplanationOutput] = None
    lime_explanation: Optional[ExplanationOutput] = None
    attention_explanation: Optional[ExplanationOutput] = None
    propaganda_explanation: Optional[ExplanationOutput] = None

    aggregated_explanation: Optional[AggregatedExplanation] = None

    consistency: Optional[ConsistencyMetrics] = None
    metrics: Optional[ExplanationMetricsOutput] = None

    explanation_quality_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    metadata: Dict[str, Any]