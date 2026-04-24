from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Literal

import yaml
from pydantic import BaseModel, Field, ConfigDict, field_validator


logger = logging.getLogger(__name__)


# =========================================================
# NORMALIZATION CONFIG
# =========================================================

class NormalizationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: Literal["minmax", "zscore", "robust"] = "minmax"
    feature_range: tuple[float, float] = (0.0, 1.0)
    per_feature: Optional[Dict[str, str]] = None  # feature → method

    @field_validator("feature_range")
    @classmethod
    def validate_range(cls, v):
        if not isinstance(v, tuple) or len(v) != 2:
            raise ValueError("feature_range must be a tuple of length 2")
        if v[0] >= v[1]:
            raise ValueError("feature_range must satisfy (min < max)")
        return v


# =========================================================
# WEIGHT CONFIG
# =========================================================

class WeightConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    weights: Dict[str, float] = Field(default_factory=dict)
    version: str = "default"
    allow_dynamic_adjustment: bool = True


# =========================================================
# RISK CONFIG
# =========================================================

class RiskConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    low_threshold: float = 0.3
    medium_threshold: float = 0.6

    @field_validator("medium_threshold")
    @classmethod
    def validate_thresholds(cls, v, info):
        low = info.data.get("low_threshold", 0.3)
        if v <= low:
            raise ValueError("medium_threshold must be greater than low_threshold")
        return v


# =========================================================
# ATTRIBUTION CONFIG
# =========================================================

class AttributionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: Literal[
        "integrated_gradients",
        "shap",
        "attention"
    ] = "integrated_gradients"

    top_k: int = 5
    normalize: bool = True
    use_confidence_weighting: bool = True

    @field_validator("top_k")
    @classmethod
    def validate_top_k(cls, v):
        if v <= 0:
            raise ValueError("top_k must be positive")
        return v


# =========================================================
# AGGREGATION CONFIG (ROOT)
# =========================================================

class AggregationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    normalization: NormalizationConfig = NormalizationConfig()
    weights: WeightConfig = WeightConfig()
    risk: RiskConfig = RiskConfig()
    attribution: AttributionConfig = AttributionConfig()

    # pipeline behavior
    strict_mode: bool = False
    enable_logging: bool = True
    enable_explanations: bool = True
    enable_risk: bool = True

    # versioning
    config_version: str = "v1"


# =========================================================
# LOADER
# =========================================================

def load_aggregation_config(
    config_path: Optional[str | Path] = None,
    *,
    override: Optional[Dict[str, Any]] = None,
) -> AggregationConfig:
    """
    Load aggregation config from YAML file or dict override.

    Priority:
        override dict > YAML file > defaults
    """

    config_data: Dict[str, Any] = {}

    # -----------------------------
    # Load from YAML
    # -----------------------------
    if config_path:
        path = Path(config_path)

        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        try:
            with path.open("r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f) or {}
        except Exception as e:
            logger.exception("Failed to load aggregation config")
            raise RuntimeError("Config loading failed") from e

    # -----------------------------
    # Apply overrides
    # -----------------------------
    if override:
        config_data.update(override)

    # -----------------------------
    # Validate & build
    # -----------------------------
    config = AggregationConfig(**config_data)

    logger.info(
        "[AggregationConfig] Loaded | version=%s | strict=%s",
        config.config_version,
        config.strict_mode,
    )

    return config


# =========================================================
# DEFAULT CONFIG (EXPORTABLE)
# =========================================================

def default_config_dict() -> Dict[str, Any]:
    """
    Returns default config as dictionary (useful for exporting YAML)
    """
    return AggregationConfig().model_dump()