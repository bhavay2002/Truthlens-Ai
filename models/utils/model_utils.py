"""Compatibility wrapper for model utility helpers."""

from src.models.utils.model_utils import load_model, preprocess_text, save_model

__all__ = ["save_model", "load_model", "preprocess_text"]
