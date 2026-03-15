"""
File: inference.py

Purpose
-------
Unified inference pipeline for TruthLens AI.

Responsibilities
----------------
1. Load trained RoBERTa model and tokenizer
2. Validate input text
3. Extract bias features
4. Extract emotion features
5. Run fake-news classification
6. Generate explainability output

Input
-----
text : str

Output
------
dict
    {
      prediction : str
      confidence : float
      bias_score : float
      emotion_score : float
      explanation : dict
    }
"""

import logging
from pathlib import Path
from typing import Dict, Any

import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer

from src.features.bias.bias_detector import detect_bias
from src.features.emotion.emotion_detector import detect_emotion
from src.explainability.model_explainer import explain_prediction
from src.models.model_utils import preprocess_text
from src.utils.settings import load_settings
from src.utils.input_validation import ensure_non_empty_text


# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Load Settings
# ---------------------------------------------------------

SETTINGS = load_settings()

MODEL_PATH = Path(SETTINGS.model.path)
MAX_LENGTH = SETTINGS.model.max_length

ID2LABEL = {0: "REAL", 1: "FAKE"}


# ---------------------------------------------------------
# Load Model + Tokenizer
# ---------------------------------------------------------

tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)
model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


# ---------------------------------------------------------
# Prediction Function
# ---------------------------------------------------------

def predict(text: str) -> Dict[str, Any]:
    """
    Perform fake news prediction with bias and emotion analysis.

    Parameters
    ----------
    text : str
        News article text.

    Returns
    -------
    dict
        Prediction result with metadata.
    """

    try:

        ensure_non_empty_text(text)

        logger.info("Starting inference pipeline")

        # -------------------------------------------------
        # Preprocess Text
        # -------------------------------------------------

        clean_text = preprocess_text(text)

        # -------------------------------------------------
        # Bias Detection
        # -------------------------------------------------

        bias_result = detect_bias(clean_text)

        bias_score = bias_result.get("bias_score", 0.0)

        # -------------------------------------------------
        # Emotion Detection
        # -------------------------------------------------

        emotion_result = detect_emotion(clean_text)

        emotion_score = emotion_result.get("emotion_score", 0.0)

        # -------------------------------------------------
        # Tokenization
        # -------------------------------------------------

        inputs = tokenizer(
            clean_text,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        # -------------------------------------------------
        # Model Prediction
        # -------------------------------------------------

        with torch.no_grad():

            outputs = model(**inputs)

            logits = outputs.logits

            probs = torch.softmax(logits, dim=1)

            confidence, pred_class = torch.max(probs, dim=1)

            prediction = ID2LABEL[pred_class.item()]

        confidence = float(confidence.item())

        # -------------------------------------------------
        # Explainability
        # -------------------------------------------------

        explanation = explain_prediction(
            model=model,
            tokenizer=tokenizer,
            text=clean_text,
        )

        logger.info(
            "Prediction completed: %s (confidence %.3f)",
            prediction,
            confidence,
        )

        return {
            "prediction": prediction,
            "confidence": confidence,
            "bias_score": float(bias_score),
            "emotion_score": float(emotion_score),
            "explanation": explanation,
        }

    except Exception as e:

        logger.exception("Inference pipeline failed")

        raise