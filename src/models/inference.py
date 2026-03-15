"""Unified inference pipeline for TruthLens AI."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer

from src.explainability.model_explainer import explain_prediction_full
from src.features.bias.bias_detector import detect_bias
from src.features.emotion.emotion_detector import detect_emotion
from src.models.model_utils import preprocess_text
from src.utils.input_validation import ensure_non_empty_text
from src.utils.settings import load_settings

logger = logging.getLogger(__name__)

SETTINGS = load_settings()

MODEL_PATH = Path(SETTINGS.model.path)
MAX_LENGTH = SETTINGS.model.max_length

_tokenizer: RobertaTokenizer | None = None
_model: RobertaForSequenceClassification | None = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_label_indices(model) -> tuple[int, int]:
    label2id = getattr(model.config, "label2id", None) or {}
    normalized = {str(k).strip().lower(): int(v) for k, v in label2id.items()}

    real_idx = normalized.get("real", 0)
    fake_idx = normalized.get("fake", 1)

    if real_idx == fake_idx:
        return 0, 1

    return real_idx, fake_idx


def _load_model_and_tokenizer() -> (
    tuple[RobertaTokenizer, RobertaForSequenceClassification]
):
    global _tokenizer, _model

    if _tokenizer is None or _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. "
                "Train the model before inference."
            )

        logger.info("Loading inference model from %s", MODEL_PATH)

        _tokenizer = RobertaTokenizer.from_pretrained(str(MODEL_PATH))
        _model = RobertaForSequenceClassification.from_pretrained(
            str(MODEL_PATH)
        )

        _model.to(_device)
        _model.eval()

    return _tokenizer, _model


def _predict_with_assets(
    text: str,
    tokenizer: RobertaTokenizer,
    model: RobertaForSequenceClassification,
) -> Dict[str, Any]:
    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    model_inputs = {key: value.to(_device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**model_inputs)

    probs = torch.softmax(outputs.logits, dim=1)
    real_idx, fake_idx = _resolve_label_indices(model)

    fake_prob = float(probs[0][fake_idx].item())
    real_prob = float(probs[0][real_idx].item())

    label = "Fake" if fake_prob > real_prob else "Real"
    confidence = max(fake_prob, real_prob)

    return {
        "label": label,
        "fake_probability": fake_prob,
        "confidence": confidence,
    }


def predict(text: str) -> Dict[str, Any]:
    """
    Perform fake-news prediction with bias/emotion analysis and explanations.
    """

    ensure_non_empty_text(text)

    logger.info("Starting inference pipeline")

    clean_text = preprocess_text(text)

    bias_result = detect_bias(clean_text)
    bias_score = float(bias_result.get("bias_score", 0.0))

    emotion_result = detect_emotion(clean_text)
    emotion_score = float(emotion_result.get("emotion_score", 0.0))

    tokenizer, model = _load_model_and_tokenizer()

    prediction_payload = _predict_with_assets(clean_text, tokenizer, model)

    explanation = explain_prediction_full(
        text=clean_text,
        predict_fn=lambda candidate: _predict_with_assets(
            preprocess_text(candidate),
            tokenizer,
            model,
        ),
        model=model,
        tokenizer=tokenizer,
        use_lime=True,
        use_shap=True,
    )

    prediction = str(prediction_payload["label"]).upper()
    confidence = float(prediction_payload["confidence"])

    logger.info(
        "Prediction completed: %s (confidence %.3f)",
        prediction,
        confidence,
    )

    return {
        "prediction": prediction,
        "confidence": confidence,
        "bias_score": bias_score,
        "emotion_score": emotion_score,
        "explanation": explanation,
    }
