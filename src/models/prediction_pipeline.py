"""
File: prediction_pipeline.py

Purpose
-------
Production prediction pipeline for TruthLens AI.

This module coordinates feature engineering, tokenization,
model inference, and result formatting.

Input
-----
text : str

Output
------
dict
    {
        prediction : str
        confidence : float
        probabilities : dict
    }
"""

import logging
import torch
import pandas as pd
from typing import Dict, Any

from src.models.model_registry import ModelRegistry
from src.features.feature_pipeline import transform_feature_pipeline
from src.utils.settings import load_settings
from src.utils.input_validation import ensure_non_empty_text


# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Settings
# ---------------------------------------------------------

SETTINGS = load_settings()

MAX_LENGTH = SETTINGS.model.max_length

ID2LABEL = {0: "REAL", 1: "FAKE"}


# ---------------------------------------------------------
# Load Model Assets
# ---------------------------------------------------------

_assets = ModelRegistry.load_model()

MODEL = _assets["model"]
TOKENIZER = _assets["tokenizer"]
VECTORIZER = _assets["vectorizer"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL.to(DEVICE)
MODEL.eval()


# ---------------------------------------------------------
# Prediction Pipeline
# ---------------------------------------------------------

def predict_text(text: str) -> Dict[str, Any]:
    """
    Run full prediction pipeline on a single news article.
    """

    try:

        ensure_non_empty_text(text)

        logger.info("Running prediction pipeline")

        # -------------------------------------------------
        # Create dataframe for feature pipeline
        # -------------------------------------------------

        df = pd.DataFrame({"text": [text]})

        # -------------------------------------------------
        # Feature engineering
        # -------------------------------------------------

        df = transform_feature_pipeline(
            df,
            vectorizer=VECTORIZER,
            text_column="text",
        )

        engineered_text = df["engineered_text"].iloc[0]

        # -------------------------------------------------
        # Tokenization
        # -------------------------------------------------

        inputs = TOKENIZER(
            engineered_text,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        # -------------------------------------------------
        # Model inference
        # -------------------------------------------------

        with torch.no_grad():

            outputs = MODEL(**inputs)

            logits = outputs.logits

            probs = torch.softmax(logits, dim=1)

            confidence, pred_class = torch.max(probs, dim=1)

        prediction = ID2LABEL[pred_class.item()]

        confidence = float(confidence.item())

        probabilities = {
            ID2LABEL[i]: float(probs[0][i].item())
            for i in range(len(ID2LABEL))
        }

        logger.info(
            "Prediction completed | class=%s | confidence=%.3f",
            prediction,
            confidence,
        )

        return {
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": probabilities,
        }

    except Exception:

        logger.exception("Prediction pipeline failed")

        raise


# ---------------------------------------------------------
# Batch Prediction
# ---------------------------------------------------------

def predict_batch(texts: list[str]) -> list[Dict[str, Any]]:
    """
    Run predictions on multiple texts.
    """

    results = []

    for text in texts:

        results.append(predict_text(text))

    return results