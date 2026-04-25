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

    method: Literal["minmax", "zscore", "robust", "quantile"] = "minmax"
    feature_range: tuple[float, float] = (0.0, 1.0)
    clip: bool = True

    per_feature: Optional[Dict[str, str]] = None

    @field_validator("feature_range")
    @classmethod
    def validate_range(cls, v):
        if len(v) != 2 or v[0] >= v[1]:
            raise ValueError("feature_range must be (min < max)")
        return v


# =========================================================
# CALIBRATION CONFIG (🔥 NEW)
# =========================================================

class CalibrationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: Literal["temperature", "isotonic", "sigmoid", "none"] = "temperature"
    n_bins: int = 15
    enabled: bool = True


# =========================================================
# UNCERTAINTY CONFIG (🔥 NEW)
# =========================================================

class UncertaintyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enable_entropy: bool = True
    track_percentiles: bool = True

    p95_threshold: float = 0.8
    p99_threshold: float = 0.95

    @field_validator("p95_threshold", "p99_threshold")
    @classmethod
    def validate_range(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError("Threshold must be in [0,1]")
        return v


# =========================================================
# WEIGHT CONFIG (UPGRADED)
# =========================================================

class WeightConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    weights: Dict[str, float] = Field(default_factory=dict)
    version: str = "v2"

    allow_dynamic_adjustment: bool = True

    # 🔥 adaptive weighting
    use_confidence: bool = True
    use_entropy: bool = True
    use_explainability: bool = True

    smoothing: float = 0.1


# =========================================================
# RISK CONFIG (UPGRADED)
# =========================================================

class RiskConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    low_threshold: float = 0.3
    medium_threshold: float = 0.6

    uncertainty_penalty: float = 0.2

    @field_validator("medium_threshold")
    @classmethod
    def validate_thresholds(cls, v, info):
        low = info.data.get("low_threshold", 0.3)
        if v <= low:
            raise ValueError("medium_threshold must be greater than low_threshold")
        return v


# =========================================================
# ATTRIBUTION CONFIG (UPGRADED)
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
    use_entropy_weighting: bool = True

    @field_validator("top_k")
    @classmethod
    def validate_top_k(cls, v):
        if v <= 0:
            raise ValueError("top_k must be positive")
        return v


# =========================================================
# DRIFT CONFIG (🔥 NEW)
# =========================================================

class DriftConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True

    method: Literal["kl", "js", "psi"] = "js"
    threshold: float = 0.1


# =========================================================
# MONITORING CONFIG (🔥 NEW)
# =========================================================

class MonitoringConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True

    track_latency: bool = True
    track_confidence: bool = True
    track_entropy: bool = True


# =========================================================
# ROOT CONFIG (UPGRADED)
# =========================================================

class AggregationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    normalization: NormalizationConfig = NormalizationConfig()
    calibration: CalibrationConfig = CalibrationConfig()
    uncertainty: UncertaintyConfig = UncertaintyConfig()
    weights: WeightConfig = WeightConfig()
    risk: RiskConfig = RiskConfig()
    attribution: AttributionConfig = AttributionConfig()

    drift: DriftConfig = DriftConfig()
    monitoring: MonitoringConfig = MonitoringConfig()

    # pipeline behavior
    strict_mode: bool = False
    enable_logging: bool = True
    enable_explanations: bool = True
    enable_risk: bool = True

    # versioning
    config_version: str = "v2"


# =========================================================
# LOADER
# =========================================================

def load_aggregation_config(
    config_path: Optional[str | Path] = None,
    *,
    override: Optional[Dict[str, Any]] = None,
) -> AggregationConfig:

    config_data: Dict[str, Any] = {}

    if config_path:
        path = Path(config_path)

        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")

        try:
            with path.open("r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f) or {}
        except Exception as e:
            logger.exception("Failed to load config")
            raise RuntimeError("Config loading failed") from e

    if override:
        config_data.update(override)

    config = AggregationConfig(**config_data)

    logger.info(
        "[AggregationConfig] Loaded | version=%s",
        config.config_version,
    )

    return config


# =========================================================
# EXPORT
# =========================================================

def default_config_dict() -> Dict[str, Any]:
    return AggregationConfig().model_dump()