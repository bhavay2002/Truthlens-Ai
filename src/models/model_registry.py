"""
File: model_registry.py

Purpose
-------
Centralized registry for loading and managing models used in TruthLens AI.

This module standardizes how trained models, tokenizers, and supporting
artifacts (e.g., TF-IDF vectorizers) are retrieved for inference or analysis.

Input
-----
model_name : str

Output
------
dict
    {
        model : HuggingFace model
        tokenizer : HuggingFace tokenizer
        vectorizer : TfidfVectorizer | None
    }
"""

import logging
from pathlib import Path
from typing import Dict, Any

import joblib
from transformers import RobertaForSequenceClassification, RobertaTokenizer

from src.utils.settings import load_settings

# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Settings
# ---------------------------------------------------------

SETTINGS = load_settings()

MODEL_DIR = Path(SETTINGS.model.path)
VECTORIZER_PATH = Path(SETTINGS.paths.tfidf_vectorizer_path)


# ---------------------------------------------------------
# Model Registry
# ---------------------------------------------------------


class ModelRegistry:
    """
    Central model loading interface.
    """

    @staticmethod
    def load_model(model_name: str = "truthlens_roberta") -> Dict[str, Any]:
        """
        Load model and associated artifacts.

        Parameters
        ----------
        model_name : str

        Returns
        -------
        dict
        """

        try:

            logger.info("Loading model from registry: %s", model_name)

            model_path = MODEL_DIR

            # -------------------------------------------------
            # Load Model
            # -------------------------------------------------

            model = RobertaForSequenceClassification.from_pretrained(
                model_path
            )

            # -------------------------------------------------
            # Load Tokenizer
            # -------------------------------------------------

            tokenizer = RobertaTokenizer.from_pretrained(model_path)

            # -------------------------------------------------
            # Load TF-IDF Vectorizer (Optional)
            # -------------------------------------------------

            vectorizer = None

            if VECTORIZER_PATH.exists():

                vectorizer = joblib.load(VECTORIZER_PATH)

                logger.info("TF-IDF vectorizer loaded")

            else:

                logger.warning("Vectorizer not found")

            logger.info("Model registry load complete")

            return {
                "model": model,
                "tokenizer": tokenizer,
                "vectorizer": vectorizer,
            }

        except Exception:

            logger.exception("Failed to load model from registry")

            raise


# ---------------------------------------------------------
# Convenience Helper
# ---------------------------------------------------------


def get_model():
    """
    Quick helper for retrieving the default model.
    """

    return ModelRegistry.load_model()
