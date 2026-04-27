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

# NOTE: In this snapshot of the repo, the IsotonicCalibrator/Config classes
# physically live inside `temperature_scaling.py` (file naming was swapped
# during development). Import them from where they actually exist so the
# package interface remains stable.
from src.models.calibration.temperature_scaling import (
    IsotonicCalibrationConfig,
    IsotonicCalibrator,
)

# TemperatureScaler is implemented under `src.evaluation.calibration`.
# Re-expose it (with a small dataclass config) so the public package API
# documented above continues to work.
from dataclasses import dataclass

from src.evaluation.calibration import TemperatureScaler


@dataclass
class TemperatureScalingConfig:
    max_iter: int = 50
    lr: float = 0.01


__all__ = [
    "CalibrationMetricConfig",
    "CalibrationMetrics",
    "IsotonicCalibrationConfig",
    "IsotonicCalibrator",
    "TemperatureScalingConfig",
    "TemperatureScaler",
]
