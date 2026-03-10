from src.data.merge_datasets import merge_datasets
from src.data.clean_data import clean_dataframe
from src.data.data_augmentation import augment_dataset
from src.features.feature_pipeline import apply_feature_engineering, save_vectorizer
from src.models.train_roberta import train_model
# from src.training.hyperparameter_tuning import run_optuna  # TODO: Fix signature mismatch
# from src.training.cross_validation import cross_validate_model  # TODO: Fix signature mismatch
from src.evaluation.evaluate_model import evaluate, save_evaluation_results
from src.visualization.visualize import plot_confusion_matrix
from src.utils.config_loader import get_config_value, get_path, load_config
from src.utils.logging_utils import configure_logging

import logging
import sys
import json
import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------
# Logging Setup
# --------------------------------------------------

config = load_config()
training_log_path = get_path(
    config,
    "paths",
    "training_log_path",
    default="logs/training.log",
)
configure_logging(log_file=training_log_path)

models_dir = get_path(config, "paths", "models_dir", default="models")
reports_dir = get_path(config, "paths", "reports_dir", default="reports")
logs_dir = get_path(config, "paths", "logs_dir", default="logs")
merged_dataset_path = get_path(
    config,
    "data",
    "merged_dataset_path",
    default="data/interim/merged_dataset.csv",
)
cleaned_dataset_path = get_path(
    config,
    "data",
    "cleaned_dataset_path",
    default="data/processed/cleaned_dataset.csv",
)
cleaning_report_path = get_path(
    config,
    "paths",
    "cleaning_report_path",
    default="reports/data_cleaning_report.json",
)
evaluation_results_path = get_path(
    config,
    "paths",
    "evaluation_results_path",
    default="reports/evaluation_results.json",
)
confusion_matrix_path = get_path(
    config,
    "paths",
    "confusion_matrix_path",
    default="reports/confusion_matrix.png",
)
tfidf_vectorizer_path = get_path(
    config,
    "paths",
    "tfidf_vectorizer_path",
    default="models/tfidf_vectorizer.joblib",
)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# Main Pipeline
# --------------------------------------------------

def main():

    try:

        models_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        merged_dataset_path.parent.mkdir(parents=True, exist_ok=True)
        cleaned_dataset_path.parent.mkdir(parents=True, exist_ok=True)

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

        df.to_csv(merged_dataset_path, index=False)

        # --------------------------------------------------
        # Clean dataset
        # --------------------------------------------------

        logger.info("Cleaning dataset...")

        before_clean = len(df)

        df = clean_dataframe(df)

        logger.info(f"Removed {before_clean - len(df)} samples during cleaning")
        logger.info(f"Dataset after cleaning: {len(df)}")

        df.to_csv(cleaned_dataset_path, index=False)

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

        with cleaning_report_path.open("w", encoding="utf-8") as f:
            json.dump(cleaning_report, f, indent=2)

        logger.info(f"Cleaned dataset saved to {cleaned_dataset_path}")
        logger.info(f"Data cleaning report saved to {cleaning_report_path}")

        # --------------------------------------------------
        # Data Augmentation
        # --------------------------------------------------
        augmentation_multiplier = int(
            get_config_value(config, "data", "augmentation_multiplier", default=2)
        )

        if augmentation_multiplier > 1:
            logger.info("Applying data augmentation with multiplier=%s", augmentation_multiplier)
            df = augment_dataset(df, text_column="text", multiplier=augmentation_multiplier)
            logger.info("Dataset size after augmentation: %s", len(df))
        else:
            logger.info("Data augmentation skipped (multiplier <= 1)")

        # --------------------------------------------------
        # Feature Engineering
        # --------------------------------------------------

        tfidf_max_features = int(
            get_config_value(config, "features", "tfidf_max_features", default=5000)
        )
        tfidf_top_terms = int(
            get_config_value(config, "features", "tfidf_top_terms_per_doc", default=4)
        )

        logger.info(
            "Applying feature pipeline (tfidf_max_features=%s, top_terms_per_doc=%s)",
            tfidf_max_features,
            tfidf_top_terms,
        )
        df, tfidf_vectorizer = apply_feature_engineering(
            df,
            text_column="text",
            tfidf_max_features=tfidf_max_features,
            top_terms_per_doc=tfidf_top_terms,
        )
        save_vectorizer(tfidf_vectorizer, tfidf_vectorizer_path)

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

        trainer, test_dataset = train_model(
            df,
            params=best_params,
            text_column="engineered_text",
        )

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
            evaluation_results_path
        )

        fig, _ = plot_confusion_matrix(evaluation_results["confusion_matrix"])

        fig.savefig(confusion_matrix_path)

        plt.close(fig)

        logger.info("Evaluation report saved")

        logger.info("=" * 50)
        logger.info("Training Complete!")
        logger.info("=" * 50)

        model_path = get_path(config, "model", "path", default="models/roberta_model")
        logger.info("Model saved to %s", model_path)
        logger.info("Run API using:")
        logger.info("uvicorn api.app:app --reload")

    except Exception as e:

        logger.error(f"Pipeline failed: {e}", exc_info=True)

        sys.exit(1)


if __name__ == "__main__":
    main()
