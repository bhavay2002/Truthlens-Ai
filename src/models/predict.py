"""
Prediction module for TruthLens AI
Loads trained model and performs fake news inference
"""

import torch
import logging
from typing import List, Dict, Union
from transformers import RobertaTokenizer, RobertaForSequenceClassification

from src.utils.settings import load_settings

logger = logging.getLogger(__name__)

# -------------------------------------------------
# Global lazy-loaded objects
# -------------------------------------------------

_tokenizer = None
_model = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SETTINGS = load_settings()
MODEL_PATH = SETTINGS.model.path
MAX_LENGTH = SETTINGS.model.max_length


def _resolve_label_indices(model) -> tuple[int, int]:
    """Resolve (real_idx, fake_idx) from model config; fallback to (0, 1)."""

    label2id = getattr(model.config, "label2id", None) or {}

    normalized = {str(k).strip().lower(): int(v) for k, v in label2id.items()}

    real_idx = normalized.get("real", 0)
    fake_idx = normalized.get("fake", 1)

    if real_idx == fake_idx:
        return 0, 1

    return real_idx, fake_idx


# -------------------------------------------------
# Model Loading
# -------------------------------------------------

def load_model_and_tokenizer():
    """Load model and tokenizer once (lazy loading)"""

    global _tokenizer, _model

    if _tokenizer is None or _model is None:

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. "
                f"Train the model first using: python main.py"
            )

        try:
            logger.info("Loading model and tokenizer...")

            _tokenizer = RobertaTokenizer.from_pretrained(str(MODEL_PATH))
            _model = RobertaForSequenceClassification.from_pretrained(str(MODEL_PATH))

            _model.to(_device)
            _model.eval()

            logger.info(f"Model loaded successfully on device: {_device}")

        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise

    return _tokenizer, _model


# -------------------------------------------------
# Text Prediction
# -------------------------------------------------

def predict(text: str) -> Dict[str, Union[str, float]]:
    """
    Predict fake news probability for a single text.

    Returns:
        {
            "label": "Fake" or "Real",
            "fake_probability": float,
            "confidence": float
        }
    """

    if not text or not text.strip():
        raise ValueError("Input text cannot be empty")

    tokenizer, model = load_model_and_tokenizer()

    try:

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH
        )

        inputs = {k: v.to(_device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.softmax(outputs.logits, dim=1)
        real_idx, fake_idx = _resolve_label_indices(model)

        fake_prob = probs[0][fake_idx].item()
        real_prob = probs[0][real_idx].item()

        label = "Fake" if fake_prob > real_prob else "Real"
        confidence = max(fake_prob, real_prob)

        return {
            "label": label,
            "fake_probability": fake_prob,
            "confidence": confidence
        }

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise


# -------------------------------------------------
# Batch Prediction
# -------------------------------------------------

def predict_batch(texts: List[str]) -> List[Dict[str, Union[str, float]]]:
    """
    Predict fake news probability for multiple texts
    """

    if not texts:
        raise ValueError("Input list cannot be empty")

    tokenizer, model = load_model_and_tokenizer()

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH
    )

    inputs = {k: v.to(_device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)
    real_idx, fake_idx = _resolve_label_indices(model)

    results = []

    for prob in probs:
        fake_prob = prob[fake_idx].item()
        real_prob = prob[real_idx].item()

        label = "Fake" if fake_prob > real_prob else "Real"
        confidence = max(fake_prob, real_prob)

        results.append({
            "label": label,
            "fake_probability": fake_prob,
            "confidence": confidence
        })

    return results
