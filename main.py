from src.data.merge_datasets import merge_datasets
from src.data.clean_data import clean_dataframe
from src.data.data_augmentation import augment_dataset
from src.features.feature_pipeline import apply_feature_engineering, save_vectorizer
from src.models.train_roberta import train_model
from src.training.hyperparameter_tuning import run_optuna
from src.training.cross_validation import cross_validate_model
from src.evaluation.evaluate_model import evaluate, save_evaluation_results
from src.visualization.visualize import plot_confusion_matrix
from src.utils.logging_utils import configure_logging
from src.utils.input_validation import ensure_dataframe
from src.utils.settings import load_settings

import logging
import sys
import json
import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------
# Logging Setup
# --------------------------------------------------

SETTINGS = load_settings()
configure_logging(log_file=SETTINGS.paths.training_log_path)

models_dir = SETTINGS.paths.models_dir
reports_dir = SETTINGS.paths.reports_dir
logs_dir = SETTINGS.paths.logs_dir
merged_dataset_path = SETTINGS.data.merged_dataset_path
cleaned_dataset_path = SETTINGS.data.cleaned_dataset_path
cleaning_report_path = SETTINGS.paths.cleaning_report_path
evaluation_results_path = SETTINGS.paths.evaluation_results_path
confusion_matrix_path = SETTINGS.paths.confusion_matrix_path
tfidf_vectorizer_path = SETTINGS.paths.tfidf_vectorizer_path

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
        ensure_dataframe(df, name="merged_df", required_columns=["text", "label"], min_rows=1)

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
        ensure_dataframe(df, name="cleaned_df", required_columns=["text", "label"], min_rows=1)

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
        augmentation_multiplier = SETTINGS.data.augmentation_multiplier

        if augmentation_multiplier > 1:
            logger.info("Applying data augmentation with multiplier=%s", augmentation_multiplier)
            df = augment_dataset(df, text_column="text", multiplier=augmentation_multiplier)
            logger.info("Dataset size after augmentation: %s", len(df))
        else:
            logger.info("Data augmentation skipped (multiplier <= 1)")

        # --------------------------------------------------
        # Feature Engineering
        # --------------------------------------------------

        tfidf_max_features = SETTINGS.features.tfidf_max_features
        tfidf_top_terms = SETTINGS.features.tfidf_top_terms_per_doc

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

        text_column = SETTINGS.training.text_column
        if text_column not in df.columns:
            raise ValueError(
                f"Configured text column '{text_column}' does not exist. "
                f"Available columns: {list(df.columns)}"
            )

        # --------------------------------------------------
        # Cross Validation (optional)
        # --------------------------------------------------
        if SETTINGS.training.run_cross_validation:
            logger.info("Running cross-validation...")
            cv_results = cross_validate_model(
                df,
                train_model,
                n_splits=SETTINGS.training.cross_validation_splits,
                text_column=text_column,
                metric_name=SETTINGS.training.cross_validation_metric,
            )
            logger.info(
                "Cross-validation %s: mean=%.4f std=%.4f folds=%s",
                cv_results["metric_name"],
                cv_results["mean_score"],
                cv_results["std_score"],
                cv_results["n_splits"],
            )

        # --------------------------------------------------
        # Hyperparameter tuning (optional)
        # --------------------------------------------------
        best_params = None
        if SETTINGS.training.run_hyperparameter_tuning:
            logger.info("Running Optuna hyperparameter tuning...")
            tuning_results = run_optuna(
                df,
                train_function=train_model,
                text_column=text_column,
                n_trials=SETTINGS.training.optuna_trials,
                metric_name=SETTINGS.training.optuna_metric,
                direction=SETTINGS.training.optuna_direction,
            )
            best_params = tuning_results["best_params"]
            logger.info(
                "Best parameters found (best_%s=%.4f): %s",
                tuning_results["metric_name"],
                tuning_results["best_value"],
                best_params,
            )

        # --------------------------------------------------
        # Final model training
        # --------------------------------------------------

        logger.info("Training final model...")

        trainer, test_dataset = train_model(
            df,
            params=best_params,
            text_column=text_column,
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

        logger.info("Model saved to %s", SETTINGS.model.path)
        logger.info("Run API using:")
        logger.info("uvicorn api.app:app --reload")

    except Exception as e:

        logger.error(f"Pipeline failed: {e}", exc_info=True)

        sys.exit(1)


if __name__ == "__main__":
    main()
