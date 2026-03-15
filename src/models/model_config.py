"""
File: model_config.py

Purpose
-------
Central configuration module for TruthLens AI models.

This file defines model architecture parameters, tokenization settings,
training hyperparameters, and feature pipeline configuration.

All model-related modules should reference this configuration
to ensure consistent behavior across the system.
"""

from dataclasses import dataclass

# ---------------------------------------------------------
# Model Configuration
# ---------------------------------------------------------


@dataclass(frozen=True)
class ModelConfig:
    """
    Configuration settings for the TruthLens AI model.
    """

    # -------------------------------------------------
    # Base Model
    # -------------------------------------------------

    MODEL_NAME: str = "roberta-base"

    NUM_LABELS: int = 2

    ID2LABEL = {
        0: "REAL",
        1: "FAKE",
    }

    LABEL2ID = {
        "REAL": 0,
        "FAKE": 1,
    }

    # -------------------------------------------------
    # Tokenization
    # -------------------------------------------------

    MAX_LENGTH: int = 256

    TRUNCATION: bool = True

    PADDING: str = "max_length"

    # -------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------

    BATCH_SIZE: int = 8

    EPOCHS: int = 4

    LEARNING_RATE: float = 2e-5

    WEIGHT_DECAY: float = 0.01

    GRADIENT_ACCUMULATION_STEPS: int = 2

    # -------------------------------------------------
    # Feature Pipeline
    # -------------------------------------------------

    TFIDF_MAX_FEATURES: int = 5000

    TOP_TFIDF_TERMS_PER_DOC: int = 4

    # -------------------------------------------------
    # Early Stopping
    # -------------------------------------------------

    EARLY_STOPPING_PATIENCE: int = 2

    # -------------------------------------------------
    # Random Seed
    # -------------------------------------------------

    SEED: int = 42


# ---------------------------------------------------------
# Path Configuration
# ---------------------------------------------------------


@dataclass(frozen=True)
class ModelPaths:
    """
    Paths used for model artifacts.
    """

    MODEL_DIR: str = "models/truthlens_roberta"

    VECTORIZER_PATH: str = "models/tfidf_vectorizer.joblib"

    LOG_DIR: str = "logs"

    CHECKPOINT_DIR: str = "models/checkpoints"


# ---------------------------------------------------------
# Inference Configuration
# ---------------------------------------------------------


@dataclass(frozen=True)
class InferenceConfig:
    """
    Settings used during inference.
    """

    DEVICE_AUTO_DETECT: bool = True

    RETURN_PROBABILITIES: bool = True

    BATCH_SIZE: int = 16
