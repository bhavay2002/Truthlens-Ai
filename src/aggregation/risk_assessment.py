from __future__ import annotations

import logging
from typing import Dict, Optional, Union, List

import numpy as np


logger = logging.getLogger(__name__)


# =========================================================
# CONFIGURABLE THRESHOLDS
# =========================================================
class RiskThresholds:
    def __init__(
        self,
        low: float = 0.3,
        medium: float = 0.6,
    ):
        if not (0.0 <= low < medium <= 1.0):
            raise ValueError("Invalid threshold configuration")

        self.low = low
        self.medium = medium


DEFAULT_THRESHOLDS = RiskThresholds()


# =========================================================
# PER-METRIC THRESHOLDS (ADVANCED)
# =========================================================
class RiskConfig:
    def __init__(
        self,
        default: RiskThresholds = DEFAULT_THRESHOLDS,
        per_key: Optional[Dict[str, RiskThresholds]] = None,
        invert_keys: Optional[List[str]] = None,
    ):
        self.default = default
        self.per_key = per_key or {}
        self.invert_keys = set(invert_keys or [])


# =========================================================
# KEY MAP (TruthLens Core)
# =========================================================
TRUTHLENS_RISK_KEY_MAP = {
    "truthlens_manipulation_risk": "manipulation_risk",
    "truthlens_credibility_score": "credibility_level",
    "truthlens_final_score": "overall_truthlens_rating",
}


# =========================================================
# VALIDATION
# =========================================================
def _validate_score(
    value: float,
    *,
    strict: bool = False,
    key: Optional[str] = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"Score must be numeric (non-boolean). Got {type(value)}")

    if np.isnan(value) or np.isinf(value):
        raise ValueError(f"Invalid score value: {value}")

    if not (0.0 <= value <= 1.0):
        msg = f"Score out of range [0,1]: {value} (key={key})"
        if strict:
            raise ValueError(msg)
        logger.warning(msg)
        value = float(np.clip(value, 0.0, 1.0))

    return float(value)


# =========================================================
# CALIBRATION SUPPORT
# =========================================================
def _apply_calibration(
    value: float,
    calibrated: Optional[bool] = True,
) -> float:
    """
    Placeholder hook: assumes upstream calibration already applied.
    Can extend with temperature / sigmoid if needed.
    """
    if calibrated:
        return value

    # fallback sigmoid (safety)
    return float(1.0 / (1.0 + np.exp(-value)))


# =========================================================
# CORE MAPPING
# =========================================================
def score_to_risk_level(
    score: float,
    *,
    thresholds: RiskThresholds,
    invert: bool = False,
    calibrated: bool = True,
    strict: bool = False,
    key: Optional[str] = None,
) -> str:

    score = _validate_score(score, strict=strict, key=key)
    score = _apply_calibration(score, calibrated=calibrated)

    if invert:
        score = 1.0 - score

    if score < thresholds.low:
        return "LOW"
    elif score < thresholds.medium:
        return "MEDIUM"
    else:
        return "HIGH"


# =========================================================
# MAIN API
# =========================================================
def assess_risk_levels(
    scores: Dict[str, float],
    *,
    config: Optional[RiskConfig] = None,
    strict: bool = False,
    calibrated: bool = True,
    return_meta: bool = False,
) -> Dict[str, str] | Dict[str, object]:

    if not isinstance(scores, dict):
        raise ValueError("scores must be a dictionary")

    config = config or RiskConfig()

    risk_levels: Dict[str, str] = {}
    skipped: List[str] = []
    valid_values: List[float] = []

    for key, value in scores.items():

        thresholds = config.per_key.get(key, config.default)
        invert = key in config.invert_keys

        try:
            level = score_to_risk_level(
                value,
                thresholds=thresholds,
                invert=invert,
                calibrated=calibrated,
                strict=strict,
                key=key,
            )

            risk_levels[key] = level
            valid_values.append(value)

        except Exception as e:
            if strict:
                raise
            logger.warning(f"[RiskAssessment] Skipping key '{key}': {e}")
            skipped.append(key)

    # =========================
    # LOGGING
    # =========================
    if valid_values:
        logger.info(
            "[RiskAssessment] Completed | total=%d valid=%d skipped=%d",
            len(scores),
            len(valid_values),
            len(skipped),
        )
    else:
        logger.warning("[RiskAssessment] No valid scores processed")

    if return_meta:
        return {
            "risk_levels": risk_levels,
            "skipped_keys": skipped,
        }

    return risk_levels


# =========================================================
# TRUTHLENS WRAPPER
# =========================================================
def assess_truthlens_risks(
    scores: Dict[str, float],
    *,
    config: Optional[RiskConfig] = None,
    strict: bool = False,
    calibrated: bool = True,
) -> Dict[str, str]:

    config = config or RiskConfig(
        invert_keys=["truthlens_credibility_score"]
    )

    output: Dict[str, str] = {}

    for score_key, risk_key in TRUTHLENS_RISK_KEY_MAP.items():

        if score_key not in scores:
            continue

        thresholds = config.per_key.get(score_key, config.default)
        invert = score_key in config.invert_keys

        output[risk_key] = score_to_risk_level(
            scores[score_key],
            thresholds=thresholds,
            invert=invert,
            calibrated=calibrated,
            strict=strict,
            key=score_key,
        )

    return output