"""Production prediction pipeline for TruthLens AI."""

from __future__ import annotations

import logging
from typing import Any, Dict

import pandas as pd
import torch

from src.features.feature_pipeline import transform_feature_pipeline
from src.models.model_registry import ModelRegistry
from src.utils.input_validation import ensure_non_empty_text
from src.utils.settings import load_settings

logger = logging.getLogger(__name__)

SETTINGS = load_settings()
MAX_LENGTH = SETTINGS.model.max_length

ID2LABEL = {0: "REAL", 1: "FAKE"}

_model = None
_tokenizer = None
_vectorizer = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_assets() -> tuple[Any, Any, Any]:
    global _model, _tokenizer, _vectorizer

    if _model is None or _tokenizer is None:
        assets = ModelRegistry.load_model()
        _model = assets["model"]
        _tokenizer = assets["tokenizer"]
        _vectorizer = assets.get("vectorizer")

        _model.to(_device)
        _model.eval()

    return _model, _tokenizer, _vectorizer


def _prepare_model_text(text: str, vectorizer) -> str:
    if vectorizer is None:
        return text

    df = pd.DataFrame({"text": [text]})
    transformed_df = transform_feature_pipeline(
        df,
        vectorizer=vectorizer,
        text_column="text",
    )
    return str(transformed_df["engineered_text"].iloc[0])


def predict_text(text: str) -> Dict[str, Any]:
    """Run full prediction pipeline on a single news article."""

    ensure_non_empty_text(text)
    logger.info("Running prediction pipeline")

    model, tokenizer, vectorizer = _get_assets()

    model_text = _prepare_model_text(text, vectorizer)

    inputs = tokenizer(
        model_text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    model_inputs = {key: value.to(_device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**model_inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        confidence, pred_class = torch.max(probs, dim=1)

    prediction = ID2LABEL[pred_class.item()]
    confidence_value = float(confidence.item())

    probabilities = {
        ID2LABEL[i]: float(probs[0][i].item()) for i in range(len(ID2LABEL))
    }

    logger.info(
        "Prediction completed | class=%s | confidence=%.3f",
        prediction,
        confidence_value,
    )

    return {
        "prediction": prediction,
        "confidence": confidence_value,
        "probabilities": probabilities,
    }


def predict_batch(texts: list[str]) -> list[Dict[str, Any]]:
    """Run predictions on multiple texts."""

    return [predict_text(text) for text in texts]
