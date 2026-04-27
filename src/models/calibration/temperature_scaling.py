"""Temperature scaling calibration.

This module owns ``TemperatureScaler`` (and its small dataclass config).
The actual numerical implementation lives in
``src.evaluation.calibration`` — historically the file naming inside
``src.models.calibration`` got swapped with ``isotonic_calibration.py``,
and this re-export kept the public API stable while we untangled the
files. Now each file in this package once again contains exactly what
its name says.

Public API:
  • :class:`TemperatureScaler`
  • :class:`TemperatureScalingConfig`
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.evaluation.calibration import TemperatureScaler  # noqa: F401  (re-export)

logger = logging.getLogger(__name__)


# =========================================================
# CONFIG
# =========================================================

@dataclass
class TemperatureScalingConfig:
    """Optimizer hyper-parameters for fitting a temperature scaler."""

    max_iter: int = 50
    lr: float = 0.01


__all__ = ["TemperatureScaler", "TemperatureScalingConfig"]
