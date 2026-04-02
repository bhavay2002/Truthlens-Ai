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


def _resolve_label_maps(model) -> tuple[dict[int, str], dict[str, int]]:
    id2label_raw = getattr(model.config, "id2label", None) or {}
    label2id_raw = getattr(model.config, "label2id", None) or {}

    id2label: dict[int, str] = {}
    for key, value in id2label_raw.items():
        try:
            id2label[int(key)] = str(value)
        except (TypeError, ValueError):
            continue

    label2id: dict[str, int] = {}
    for key, value in label2id_raw.items():
        try:
            label2id[str(key).strip().lower()] = int(value)
        except (TypeError, ValueError):
            continue

    if not id2label and label2id:
        id2label = {idx: name for name, idx in label2id.items()}

    return id2label, label2id


def _render_label(raw_label: str) -> str:
    normalized = str(raw_label).strip()
    upper = normalized.upper()
    if upper == "REAL":
        return "Real"
    if upper == "FAKE":
        return "Fake"
    return normalized


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
    pred_idx = int(torch.argmax(probs, dim=1)[0].item())
    confidence = float(probs[0][pred_idx].item())

    id2label, label2id = _resolve_label_maps(model)
    label = _render_label(id2label.get(pred_idx, f"LABEL_{pred_idx}"))

    fake_idx = label2id.get("fake")
    if fake_idx is None or fake_idx >= int(probs.shape[1]):
        fake_prob = confidence
    else:
        fake_prob = float(probs[0][fake_idx].item())

    return {
        "label": label,
        "fake_probability": fake_prob,
        "confidence": confidence,
    }


def _predict_for_explainability(candidate_text: str) -> Dict[str, Any]:
    """
    Stable predictor for explainability modules.

    Using a named callable avoids per-request lambda identities and improves
    explainer cache reuse.
    """

    normalized_text = preprocess_text(candidate_text)
    tokenizer, model = _load_model_and_tokenizer()
    return _predict_with_assets(normalized_text, tokenizer, model)


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
        predict_fn=_predict_for_explainability,
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
