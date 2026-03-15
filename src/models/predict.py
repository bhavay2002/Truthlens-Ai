"""
File: src/models/predict.py

Purpose
-------
Provides inference utilities for the TruthLens AI fake news detection system.
Loads the trained RoBERTa model and performs prediction on input news text.

The module ensures that inference follows the same preprocessing pipeline
used during training. If engineered text was used during training,
the TF-IDF feature pipeline is applied before model inference.

Key Features
------------
- Lazy loading of RoBERTa model and tokenizer
- Optional TF-IDF feature transformation
- Automatic CPU/GPU device selection
- Supports both single-text and batch prediction

Inputs
------
text : str
    A single news article or headline.

texts : List[str]
    List of news texts for batch prediction.

Outputs
-------
For single prediction:
Dict[str, Union[str, float]]

{
    "label": "Fake" | "Real",
    "fake_probability": float,
    "confidence": float
}

For batch prediction:
List[Dict[str, Union[str, float]]]

Dependencies
------------
torch
transformers
pandas
joblib
logging

Internal Modules
----------------
src.features.feature_pipeline
src.utils.settings
"""

import torch
import logging
from typing import List, Dict, Union
import pandas as pd
import joblib
from transformers import RobertaTokenizer, RobertaForSequenceClassification

from src.features.feature_pipeline import transform_feature_pipeline
from src.utils.settings import load_settings

logger = logging.getLogger(__name__)

# -------------------------------------------------
# Global lazy-loaded objects
# -------------------------------------------------

_tokenizer = None
_model = None
_vectorizer = None
_vectorizer_load_attempted = False
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SETTINGS = load_settings()
MODEL_PATH = SETTINGS.model.path
MAX_LENGTH = SETTINGS.model.max_length
VECTORIZER_PATH = SETTINGS.paths.tfidf_vectorizer_path
TRAINING_TEXT_COLUMN = SETTINGS.training.text_column
TOP_TERMS_PER_DOC = SETTINGS.features.tfidf_top_terms_per_doc


def _resolve_label_indices(model) -> tuple[int, int]:
    """Resolve (real_idx, fake_idx) from model config; fallback to (0, 1)."""

    label2id = getattr(model.config, "label2id", None) or {}
    normalized = {str(k).strip().lower(): int(v) for k, v in label2id.items()}

    real_idx = normalized.get("real", 0)
    fake_idx = normalized.get("fake", 1)

    if real_idx == fake_idx:
        return 0, 1

    return real_idx, fake_idx


def _load_vectorizer():
    """Lazy-load TF-IDF vectorizer used by feature pipeline."""

    global _vectorizer, _vectorizer_load_attempted

    if _vectorizer is None and not _vectorizer_load_attempted:
        _vectorizer_load_attempted = True

        if not VECTORIZER_PATH.exists():
            logger.warning(
                "Vectorizer file not found at %s. "
                "Falling back to raw text inference.",
                VECTORIZER_PATH,
            )
            return None

        try:
            _vectorizer = joblib.load(VECTORIZER_PATH)
        except Exception as e:
            logger.warning(
                "Failed to load vectorizer from %s (%s). "
                "Falling back to raw text inference.",
                VECTORIZER_PATH,
                e,
            )
            return None

    return _vectorizer


def _prepare_texts_for_inference(texts: List[str]) -> List[str]:
    """Prepare input text consistent with training feature pipeline."""

    df = pd.DataFrame({"text": texts})

    if TRAINING_TEXT_COLUMN == "text":
        return df["text"].astype(str).tolist()

    if TRAINING_TEXT_COLUMN == "engineered_text":
        vectorizer = _load_vectorizer()

        if vectorizer is None:
            return df["text"].astype(str).tolist()

        try:
            transformed_df = transform_feature_pipeline(
                df,
                vectorizer=vectorizer,
                text_column="text",
                top_terms_per_doc=TOP_TERMS_PER_DOC,
            )

            return transformed_df["engineered_text"].astype(str).tolist()

        except Exception as e:
            logger.warning(
                "Feature transform failed during inference (%s). "
                "Falling back to raw text.",
                e,
            )
            return df["text"].astype(str).tolist()

    logger.warning(
        "Configured training text column '%s' not supported during inference. "
        "Using raw text.",
        TRAINING_TEXT_COLUMN,
    )

    return df["text"].astype(str).tolist()


# -------------------------------------------------
# Model Loading
# -------------------------------------------------


def load_model_and_tokenizer():
    """Load model and tokenizer once using lazy loading."""

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
            _model = RobertaForSequenceClassification.from_pretrained(
                str(MODEL_PATH)
            )

            _model.to(_device)
            _model.eval()

            logger.info(f"Model loaded successfully on device: {_device}")

        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise

    return _tokenizer, _model


# -------------------------------------------------
# Single Prediction
# -------------------------------------------------


def predict(text: str) -> Dict[str, Union[str, float]]:
    """
    Predict fake news probability for a single text.
    """

    if not text or not text.strip():
        raise ValueError("Input text cannot be empty")

    tokenizer, model = load_model_and_tokenizer()
    model_text = _prepare_texts_for_inference([text])[0]

    inputs = tokenizer(
        model_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
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
        "confidence": confidence,
    }


# -------------------------------------------------
# Batch Prediction
# -------------------------------------------------


def predict_batch(texts: List[str]) -> List[Dict[str, Union[str, float]]]:
    """
    Predict fake news probability for multiple texts.
    """

    if not texts:
        raise ValueError("Input list cannot be empty")

    tokenizer, model = load_model_and_tokenizer()
    model_texts = _prepare_texts_for_inference(texts)

    inputs = tokenizer(
        model_texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
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

        results.append(
            {
                "label": label,
                "fake_probability": fake_prob,
                "confidence": confidence,
            }
        )

    return results
