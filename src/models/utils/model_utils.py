"""
File Name: model_utils.py
Module: models.utils
Description:
    Utility functions for model persistence and basic text preprocessing
    used in the TruthLens AI system. This module provides standardized
    helpers for saving and loading serialized models as well as minimal
    text normalization utilities for inference pipelines.

    These helpers are intentionally lightweight and framework-agnostic
    so they can be used across training, evaluation, and inference code.

Dependencies:
    logging
    pathlib
    typing
    re
    joblib
Inputs:
    model objects
    file paths
    raw text
Outputs:
    saved model files
    loaded model objects
    normalized text
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import joblib

logger = logging.getLogger(__name__)


def save_model(model: Any, path: str | Path) -> Path:
    """
    Save a trained model to disk using joblib serialization.

    Parameters
    ----------
    model : Any
        Trained model object.
    path : str | Path
        Destination file path.

    Returns
    -------
    Path
        Path where the model was saved.
    """

    path_obj = Path(path)

    try:
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(model, path_obj)

        logger.info("Model saved successfully: %s", path_obj)

        return path_obj

    except Exception as exc:
        logger.exception("Failed to save model")
        raise RuntimeError("Model saving failed") from exc


def load_model(path: str | Path) -> Any:
    """
    Load a serialized model from disk.

    Parameters
    ----------
    path : str | Path
        Path to saved model file.

    Returns
    -------
    Any
        Loaded model object.
    """

    path_obj = Path(path)

    if not path_obj.exists():
        raise FileNotFoundError(f"Model file not found: {path_obj}")

    try:
        model = joblib.load(path_obj)

        logger.info("Model loaded successfully: %s", path_obj)

        return model

    except Exception as exc:
        logger.exception("Failed to load model")
        raise RuntimeError("Model loading failed") from exc


def preprocess_text(text: str) -> str:
    """
    Perform lightweight text normalization for inference.

    Operations:
        • Remove newline and tab characters
        • Normalize whitespace
        • Strip leading/trailing spaces

    Parameters
    ----------
    text : str
        Input text.

    Returns
    -------
    str
        Normalized text.
    """

    if text is None:
        raise ValueError("Input text cannot be None")

    normalized = str(text).replace("\n", " ").replace("\t", " ")
    normalized = re.sub(r"\s+", " ", normalized).strip()

    return normalized