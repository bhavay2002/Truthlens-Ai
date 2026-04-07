"""Compatibility wrapper for the TruthLens multitask model."""

from __future__ import annotations

from .multitask_truthlens_model import MultiTaskTruthLensConfig, MultiTaskTruthLensModel


class TruthLensMultiTaskModel(MultiTaskTruthLensModel):
    """Initialize the multitask model from a base model name string."""

    def __init__(self, model_name: str = "roberta-base", **kwargs):
        config = MultiTaskTruthLensConfig(model_name=model_name, **kwargs)
        super().__init__(config)
