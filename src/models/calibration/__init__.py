"""
Package: src.models.calibration
Description:
    Post-hoc calibration utilities for probabilistic classification models.

    Exposes:
      • CalibrationMetrics / CalibrationMetricConfig — ECE, MCE, Brier, NLL
      • TemperatureScaler / TemperatureScalingConfig   — single-parameter logit scaling
      • IsotonicCalibrator / IsotonicCalibrationConfig  — isotonic regression calibration
"""

from src.models.calibration.calibration_metrics import (
    CalibrationMetricConfig,
    CalibrationMetrics,
)
from src.models.calibration.isotonic_calibration import (
    IsotonicCalibrationConfig,
    IsotonicCalibrator,
)
from src.models.calibration.temperature_scaling import (
    TemperatureScalingConfig,
    TemperatureScaler,
)

__all__ = [
    "CalibrationMetricConfig",
    "CalibrationMetrics",
    "IsotonicCalibrationConfig",
    "IsotonicCalibrator",
    "TemperatureScalingConfig",
    "TemperatureScaler",
]
