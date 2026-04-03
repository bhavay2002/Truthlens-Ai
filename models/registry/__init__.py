"""Compatibility package for legacy `models.registry.*` imports."""

import sys

from src.models.registry import model_registry as _model_registry

model_registry = _model_registry
sys.modules[__name__ + ".model_registry"] = _model_registry

__all__ = ["model_registry"]
