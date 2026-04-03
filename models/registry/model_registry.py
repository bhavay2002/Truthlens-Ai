"""Compatibility wrapper for model registry APIs."""

from src.models.registry.model_registry import (
    MODEL_DIR,
    VECTORIZER_PATH,
    ModelRegistry,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    get_model,
    joblib,
)

__all__ = [
    "MODEL_DIR",
    "VECTORIZER_PATH",
    "ModelRegistry",
    "RobertaForSequenceClassification",
    "RobertaTokenizer",
    "get_model",
    "joblib",
]
