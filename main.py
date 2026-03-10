from src.data.load_data import merge_datasets
from src.data.clean_data import clean_dataframe
from src.data.data_augmentation import augment_dataset
from src.models.train_roberta import train_model
# from src.training.hyperparameter_tuning import run_optuna  # TODO: Fix signature mismatch
# from src.training.cross_validation import cross_validate_model  # TODO: Fix signature mismatch
from src.evaluation.evaluate_model import evaluate, save_evaluation_results
from src.visualization.visualize import plot_confusion_matrix

import logging
import sys
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------
# Logging Setup
# --------------------------------------------------

Path("logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/training.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# Main Pipeline
# --------------------------------------------------

def main():

    try:

        Path("models").mkdir(exist_ok=True)
        Path("data/interim").mkdir(parents=True, exist_ok=True)
        Path("data/processed").mkdir(parents=True, exist_ok=True)
        Path("reports").mkdir(parents=True, exist_ok=True)

        logger.info("=" * 50)
        logger.info("Starting TruthLens AI Training Pipeline")
        logger.info("=" * 50)

        # --------------------------------------------------
        # Merge datasets
        # --------------------------------------------------

        logger.info("Merging datasets (ISOT + FakeNewsNet + LIAR)...")

        df = merge_datasets()

        logger.info(f"Total samples loaded: {len(df)}")
        logger.info(f"Fake samples: {(df['label'] == 1).sum()}")
        logger.info(f"Real samples: {(df['label'] == 0).sum()}")

        df.to_csv("data/interim/merged_dataset.csv", index=False)

        # --------------------------------------------------
        # Clean dataset
        # --------------------------------------------------

        logger.info("Cleaning dataset...")

        before_clean = len(df)

        df = clean_dataframe(df)

        logger.info(f"Removed {before_clean - len(df)} samples during cleaning")
        logger.info(f"Dataset after cleaning: {len(df)}")

        cleaned_path = Path("data/processed/cleaned_dataset.csv")

        df.to_csv(cleaned_path, index=False)

        cleaning_report = {
            "raw_rows": int(before_clean),
            "cleaned_rows": int(len(df)),
            "rows_removed": int(before_clean - len(df)),
            "retention_rate": float(len(df) / before_clean if before_clean else 0),
            "label_distribution": {
                str(k): int(v)
                for k, v in df["label"].value_counts().to_dict().items()
            },
        }

        with open("reports/data_cleaning_report.json", "w") as f:
            json.dump(cleaning_report, f, indent=2)

        logger.info(f"Cleaned dataset saved to {cleaned_data_path}")
        logger.info(f"Data cleaning report saved to {cleaning_report_path}")

        # --------------------------------------------------
        # Data Augmentation (DISABLED - multiplier=1 does nothing)
        # --------------------------------------------------
        # Uncomment and increase multiplier (e.g., 2-3) to actually augment data
        # logger.info("Applying data augmentation...")
        # df = augment_dataset(df, text_column="text", multiplier=2)
        # logger.info(f"Dataset size after augmentation: {len(df)}")

        # --------------------------------------------------
        # Cross Validation (DISABLED - needs fixing)
        # --------------------------------------------------
        # TODO: Fix cross_validation.py to match train_model signature
        # logger.info("Running cross-validation...")
        # cv_score = cross_validate_model(df, train_model)
        # logger.info(f"Cross validation score: {cv_score:.4f}")

        # --------------------------------------------------
        # Hyperparameter tuning (DISABLED - needs fixing)
        # --------------------------------------------------
        # TODO: Fix run_optuna to accept dataframe or prepare datasets first
        # logger.info("Running Optuna hyperparameter tuning...")
        # best_params = run_optuna(df)
        # logger.info(f"Best parameters: {best_params}")
        best_params = None  # Use defaults

        # --------------------------------------------------
        # Final model training
        # --------------------------------------------------

        logger.info("Training final model...")

        trainer, test_dataset = train_model(df, best_params)

        # --------------------------------------------------
        # Model Evaluation
        # --------------------------------------------------

        logger.info("Evaluating model...")

        prediction_output = trainer.predict(test_dataset)

        logits = prediction_output.predictions
        y_true = prediction_output.label_ids
        y_pred = np.argmax(logits, axis=1)

        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probabilities = exp_logits / exp_logits.sum(axis=1, keepdims=True)

        fake_probabilities = probabilities[:, 1]

        evaluation_results = evaluate(y_true, y_pred, fake_probabilities)

        save_evaluation_results(
            evaluation_results,
            "reports/evaluation_results.json"
        )

        fig, _ = plot_confusion_matrix(evaluation_results["confusion_matrix"])

        fig.savefig("reports/confusion_matrix.png")

        plt.close(fig)

        logger.info("Evaluation report saved")

        logger.info("=" * 50)
        logger.info("Training Complete!")
        logger.info("=" * 50)

        logger.info("Model saved to ./models/roberta_model")
        logger.info("Run API using:")
        logger.info("uvicorn api.app:app --reload")

    except Exception as e:

        logger.error(f"Pipeline failed: {e}", exc_info=True)

        sys.exit(1)


if __name__ == "__main__":
    main()