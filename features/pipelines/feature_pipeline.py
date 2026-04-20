"""Compatibility wrapper for feature pipeline imports."""

try:
    from src.features.pipelines.feature_pipeline import (  # noqa: F401
        FeaturePipeline,
        apply_feature_engineering,
        transform_feature_pipeline,
    )
except ImportError as exc:
    raise ImportError(
        "Failed to import src.features.pipelines.feature_pipeline. "
        "Ensure project root is on PYTHONPATH and src package is importable."
    ) from exc

__all__ = [
    "FeaturePipeline",
    "apply_feature_engineering",
    "transform_feature_pipeline",
]
