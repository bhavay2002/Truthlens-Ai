"""
TruthLens AI Training Pipeline

Supports multi-model training for:

- Fake News Detection
- Bias Detection
- Propaganda Detection
- Narrative Classification
- Ideology Classification
- Emotion Detection
"""

import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from src.data.clean_data import clean_dataframe
from src.data.data_augmentation import augment_dataset
from src.data.merge_datasets import merge_datasets

from src.evaluation.evaluate_model import evaluate, save_evaluation_results

from features.pipelines.feature_pipeline import (
    fit_feature_pipeline,
    save_vectorizer,
    transform_feature_pipeline,
)

from training.train_transformer_model import train_model

from src.training.cross_validation import cross_validate_model
from src.training.hyperparameter_tuning import run_optuna

from src.utils.input_validation import ensure_dataframe
from src.utils.logging_utils import configure_logging
from src.utils.settings import load_settings

from src.visualization.visualize import plot_confusion_matrix


# -----------------------------------------------------
# Settings
# -----------------------------------------------------

SETTINGS = load_settings()
configure_logging(log_file=SETTINGS.paths.training_log_path)

logger = logging.getLogger(__name__)

models_dir = SETTINGS.paths.models_dir
reports_dir = SETTINGS.paths.reports_dir
logs_dir = SETTINGS.paths.logs_dir

merged_dataset_path = SETTINGS.data.merged_dataset_path
cleaned_dataset_path = SETTINGS.data.cleaned_dataset_path

cleaning_report_path = SETTINGS.paths.cleaning_report_path
evaluation_results_path = SETTINGS.paths.evaluation_results_path
confusion_matrix_path = SETTINGS.paths.confusion_matrix_path
tfidf_vectorizer_path = SETTINGS.paths.tfidf_vectorizer_path


# -----------------------------------------------------
# Multi-model task configuration
# -----------------------------------------------------

TASKS = {
    "fake_news": {"label_column": "label"},
    "bias": {"label_column": "bias_label"},
    "propaganda": {"label_column": "propaganda_label"},
    "narrative": {"label_column": "narrative_label"},
    "ideology": {"label_column": "ideology_label"},
    "emotion": {"label_column": "emotion_label"},
}


# -----------------------------------------------------
# Dataset Split
# -----------------------------------------------------

def split_dataset(df):

    holdout_size = SETTINGS.training.validation_size + SETTINGS.training.test_size

    if not (0.0 < holdout_size < 1.0):
        raise ValueError("validation_size + test_size must be between 0 and 1")

    train_df, holdout_df = train_test_split(
        df,
        test_size=holdout_size,
        random_state=SETTINGS.training.seed,
        stratify=df["label"],
    )

    val_fraction = SETTINGS.training.validation_size / holdout_size

    val_df, test_df = train_test_split(
        holdout_df,
        test_size=(1 - val_fraction),
        random_state=SETTINGS.training.seed,
        stratify=holdout_df["label"],
    )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


# -----------------------------------------------------
# Training Pipeline
# -----------------------------------------------------

def main():

    try:

        models_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 60)
        logger.info("TruthLens Multi-Model Training Pipeline")
        logger.info("=" * 60)

        # -----------------------------------------------------
        # Merge datasets
        # -----------------------------------------------------

        logger.info("Merging datasets...")

        df = merge_datasets()

        ensure_dataframe(df, name="merged_df", required_columns=["text", "label"], min_rows=1)

        logger.info("Total samples loaded: %s", len(df))

        df.to_csv(merged_dataset_path, index=False)

        # -----------------------------------------------------
        # Clean dataset
        # -----------------------------------------------------

        logger.info("Cleaning dataset...")

        before_clean = len(df)

        df = clean_dataframe(df)

        ensure_dataframe(df, name="cleaned_df", required_columns=["text", "label"], min_rows=1)

        df.to_csv(cleaned_dataset_path, index=False)

        cleaning_report = {
            "raw_rows": int(before_clean),
            "cleaned_rows": int(len(df)),
            "rows_removed": int(before_clean - len(df)),
            "retention_rate": float(len(df) / before_clean if before_clean else 0),
        }

        with open(cleaning_report_path, "w") as f:
            json.dump(cleaning_report, f, indent=2)

        logger.info("Dataset cleaned: %s rows", len(df))

        # -----------------------------------------------------
        # Split dataset
        # -----------------------------------------------------

        train_df, val_df, test_df = split_dataset(df)

        logger.info(
            "Dataset split -> train=%s val=%s test=%s",
            len(train_df),
            len(val_df),
            len(test_df),
        )

        # -----------------------------------------------------
        # Data augmentation
        # -----------------------------------------------------

        augmentation_multiplier = SETTINGS.data.augmentation_multiplier

        if augmentation_multiplier > 1:

            logger.info("Applying augmentation (multiplier=%s)", augmentation_multiplier)

            train_df = augment_dataset(
                train_df,
                text_column="text",
                multiplier=augmentation_multiplier,
            )

        # -----------------------------------------------------
        # Feature engineering
        # -----------------------------------------------------

        logger.info("Running feature pipeline...")

        train_df, tfidf_vectorizer = fit_feature_pipeline(
            train_df,
            text_column="text",
            tfidf_max_features=SETTINGS.features.tfidf_max_features,
            top_terms_per_doc=SETTINGS.features.tfidf_top_terms_per_doc,
        )

        val_df = transform_feature_pipeline(
            val_df,
            vectorizer=tfidf_vectorizer,
            text_column="text",
            top_terms_per_doc=SETTINGS.features.tfidf_top_terms_per_doc,
        )

        test_df = transform_feature_pipeline(
            test_df,
            vectorizer=tfidf_vectorizer,
            text_column="text",
            top_terms_per_doc=SETTINGS.features.tfidf_top_terms_per_doc,
        )

        save_vectorizer(tfidf_vectorizer, tfidf_vectorizer_path)

        text_column = SETTINGS.training.text_column

        # -----------------------------------------------------
        # Cross Validation
        # -----------------------------------------------------

        if SETTINGS.training.run_cross_validation:

            logger.info("Running cross validation...")

            cv_results = cross_validate_model(
                train_df,
                train_model,
                n_splits=SETTINGS.training.cross_validation_splits,
                text_column=text_column,
                metric_name=SETTINGS.training.cross_validation_metric,
            )

            logger.info("CV Mean Score: %.4f", cv_results["mean_score"])

        # -----------------------------------------------------
        # Hyperparameter tuning
        # -----------------------------------------------------

        best_params = None

        if SETTINGS.training.run_hyperparameter_tuning:

            logger.info("Running Optuna hyperparameter tuning...")

            tuning_results = run_optuna(
                train_df,
                train_function=train_model,
                validation_df=val_df,
                text_column=text_column,
                n_trials=SETTINGS.training.optuna_trials,
                metric_name=SETTINGS.training.optuna_metric,
                direction=SETTINGS.training.optuna_direction,
            )

            best_params = tuning_results["best_params"]

        # -----------------------------------------------------
        # Multi-model training loop
        # -----------------------------------------------------

        all_results = {}

        for task_name, task_cfg in TASKS.items():

            label_column = task_cfg["label_column"]

            if label_column not in train_df.columns:

                logger.warning("Skipping task %s (missing column)", task_name)
                continue

            logger.info("=" * 40)
            logger.info("Training model: %s", task_name)
            logger.info("=" * 40)

            trainer, test_dataset = train_model(
                train_df,
                params=best_params,
                text_column=text_column,
                label_column=label_column,
                validation_df=val_df,
                test_df=test_df,
                task_name=task_name,
            )

            logger.info("Evaluating %s...", task_name)

            predictions = trainer.predict(test_dataset)

            logits = predictions.predictions
            y_true = predictions.label_ids
            y_pred = np.argmax(logits, axis=1)

            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

            results = evaluate(y_true, y_pred, probs[:, 1])

            task_report_path = reports_dir / f"{task_name}_evaluation.json"

            save_evaluation_results(results, task_report_path)

            fig, _ = plot_confusion_matrix(results["confusion_matrix"])

            fig.savefig(reports_dir / f"{task_name}_confusion_matrix.png")

            plt.close(fig)

            all_results[task_name] = results

        # -----------------------------------------------------
        # Combined evaluation report
        # -----------------------------------------------------

        combined_report_path = reports_dir / "truthlens_multi_model_results.json"

        with open(combined_report_path, "w") as f:
            json.dump(all_results, f, indent=2)

        logger.info("Combined evaluation report saved")

        logger.info("=" * 60)
        logger.info("Training Completed Successfully")
        logger.info("=" * 60)

    except Exception as e:

        logger.error("Pipeline failed: %s", e, exc_info=True)

        sys.exit(1)


if __name__ == "__main__":
    main()