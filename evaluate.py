"""
Standalone evaluation script for trained models
Run this after training to get detailed evaluation metrics
"""
import logging
import sys

import pandas as pd
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from src.evaluation.evaluate_model import evaluate, save_evaluation_results
from src.visualization.visualize import plot_confusion_matrix
from src.utils.config_loader import get_config_value, get_path, load_config
from src.utils.logging_utils import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

CONFIG = load_config()
MODEL_PATH = get_path(CONFIG, "model", "path", default="models/roberta_model")
TEST_SET_PATH = get_path(
    CONFIG,
    "data",
    "test_set_path",
    default="data/processed/test_set.csv",
)
MAX_LENGTH = int(get_config_value(CONFIG, "model", "max_length", default=256))
REPORTS_DIR = get_path(CONFIG, "paths", "reports_dir", default="reports")
CONFUSION_MATRIX_PATH = get_path(
    CONFIG,
    "paths",
    "confusion_matrix_path",
    default="reports/confusion_matrix.png",
)


def _resolve_fake_index(model) -> int:
    """Resolve fake class index from model config; default to 1."""
    label2id = getattr(model.config, "label2id", None) or {}
    normalized = {str(k).strip().lower(): int(v) for k, v in label2id.items()}
    return normalized.get("fake", 1)


def evaluate_saved_model():
    """Evaluate the saved model on test set"""
    try:
        # Check if model exists
        if not MODEL_PATH.exists():
            logger.error("Model not found. Please train the model first: python main.py")
            sys.exit(1)
        
        # Check if test set exists
        if not TEST_SET_PATH.exists():
            logger.error("Test set not found. Please run training first: python main.py")
            sys.exit(1)
        
        logger.info("Loading model and tokenizer...")
        tokenizer = RobertaTokenizer.from_pretrained(str(MODEL_PATH))
        model = RobertaForSequenceClassification.from_pretrained(str(MODEL_PATH))
        model.eval()
        fake_idx = _resolve_fake_index(model)
        
        logger.info("Loading test data...")
        test_df = pd.read_csv(TEST_SET_PATH)
        logger.info(f"Test samples: {len(test_df)}")
        text_column = "engineered_text" if "engineered_text" in test_df.columns else "text"
        
        # Make predictions
        logger.info("Making predictions...")
        predictions = []
        probabilities = []
        
        for idx, row in test_df.iterrows():
            text = row[text_column]
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=MAX_LENGTH
            )
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            probs = torch.softmax(outputs.logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            prob = probs[0][fake_idx].item()  # Probability of fake
            
            predictions.append(pred)
            probabilities.append(prob)
            
            if (idx + 1) % 100 == 0:
                logger.info(f"Processed {idx + 1}/{len(test_df)} samples")
        
        # Evaluate
        y_true = test_df['label'].values
        y_pred = predictions
        
        results = evaluate(y_true, y_pred, probabilities)
        
        # Save results
        save_evaluation_results(results)
        
        # Plot confusion matrix
        import matplotlib.pyplot as plt
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        fig, _ = plot_confusion_matrix(results['confusion_matrix'])
        fig.savefig(CONFUSION_MATRIX_PATH)
        plt.close(fig)
        logger.info("Confusion matrix saved to %s", CONFUSION_MATRIX_PATH)
        
        logger.info("Evaluation complete!")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    evaluate_saved_model()
