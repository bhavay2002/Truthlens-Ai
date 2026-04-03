"""Compatibility package for legacy `pipelines.*` imports."""

import sys
from src.pipelines import prediction_pipeline as _prediction_pipeline

prediction_pipeline = _prediction_pipeline
sys.modules[__name__ + ".prediction_pipeline"] = _prediction_pipeline

__all__ = ["prediction_pipeline"]
