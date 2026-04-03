"""Compatibility wrapper for feature pipeline imports."""

from src.features.pipelines.feature_pipeline import (  # noqa: F401
    FeaturePipeline,
    apply_feature_engineering,
    transform_feature_pipeline,
)

__all__ = [
    "FeaturePipeline",
    "apply_feature_engineering",
    "transform_feature_pipeline",
]
