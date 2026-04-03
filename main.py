"""
TruthLens Multi-Task Training Pipeline

Single Transformer with 5 prediction heads:
1. Bias
2. Ideology
3. Propaganda
4. Narrative
5. Emotion
"""

import logging
import sys
import pandas as pd
from transformers import AutoTokenizer

from src.models.multitask_model import TruthLensMultiTaskModel
from src.training.multitask_trainer import train_multitask_model

from src.utils.settings import load_settings
from src.utils.logging_utils import configure_logging

# -----------------------------------------------------
# Settings
# -----------------------------------------------------

SETTINGS = load_settings()
configure_logging(log_file=SETTINGS.paths.training_log_path)

logger = logging.getLogger(__name__)

TRAIN_PATH = "/content/drive/MyDrive/truthlens-data/train.csv"
VAL_PATH = "/content/drive/MyDrive/truthlens-data/val.csv"
TEST_PATH = "/content/drive/MyDrive/truthlens-data/test.csv"


# -----------------------------------------------------
# Load Dataset
# -----------------------------------------------------

def load_data():

    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    test_df = pd.read_csv(TEST_PATH)

    # combine title + text
    train_df["text"] = train_df["title"].fillna("") + " " + train_df["text"]
    val_df["text"] = val_df["title"].fillna("") + " " + val_df["text"]
    test_df["text"] = test_df["title"].fillna("") + " " + test_df["text"]

    return train_df, val_df, test_df


# -----------------------------------------------------
# Label groups
# -----------------------------------------------------

BIAS_LABEL = "bias_label"

IDEOLOGY_LABEL = "ideology_label"

PROPAGANDA_LABEL = "propaganda_label"

NARRATIVE_COLUMNS = [
    "frame",
    "CO",
    "EC",
    "HI",
    "MO",
    "RE",
    "hero",
    "villain",
    "victim",
]

EMOTION_COLUMNS = [f"emotion_{i}" for i in range(20)]


# -----------------------------------------------------
# Main
# -----------------------------------------------------

def main():

    try:

        logger.info("Loading dataset")

        train_df, val_df, test_df = load_data()

        tokenizer = AutoTokenizer.from_pretrained(
            "roberta-base"
        )

        logger.info("Initializing multi-task model")

        model = TruthLensMultiTaskModel(
            encoder_name="roberta-base",
            narrative_dim=len(NARRATIVE_COLUMNS),
            emotion_dim=len(EMOTION_COLUMNS),
        )

        logger.info("Starting training")

        train_multitask_model(
            model=model,
            tokenizer=tokenizer,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            text_column="text",
            bias_column=BIAS_LABEL,
            ideology_column=IDEOLOGY_LABEL,
            propaganda_column=PROPAGANDA_LABEL,
            narrative_columns=NARRATIVE_COLUMNS,
            emotion_columns=EMOTION_COLUMNS,
        )

        logger.info("Training completed successfully")

    except Exception as e:

        logger.error("Pipeline failed: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()