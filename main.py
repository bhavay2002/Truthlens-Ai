import json
import logging
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from src.data.clean_data import clean_dataframe
from src.data.data_augmentation import augment_dataset
from src.data.merge_datasets import merge_datasets
from src.evaluation.evaluate_model import evaluate, save_evaluation_results
from src.features.feature_pipeline import (
    fit_feature_pipeline,
    save_vectorizer,
    transform_feature_pipeline,
)
from src.models.train_roberta import train_model
from src.training.cross_validation import cross_validate_model
from src.training.hyperparameter_tuning import run_optuna
from src.utils.input_validation import ensure_dataframe
from src.utils.logging_utils import configure_logging
from src.utils.settings import load_settings
from src.visualization.visualize import plot_confusion_matrix


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


def _split_clean_dataset(df):
    holdout_size = SETTINGS.training.validation_size + SETTINGS.training.test_size
    if not (0.0 < holdout_size < 1.0):
        raise ValueError("training.validation_size + training.test_size must be between 0 and 1")

    train_df, holdout_df = train_test_split(
        df,
        test_size=holdout_size,
        random_state=SETTINGS.training.seed,
        stratify=df["label"],
    )

    val_fraction_within_holdout = SETTINGS.training.validation_size / holdout_size
    val_df, test_df = train_test_split(
        holdout_df,
        test_size=(1.0 - val_fraction_within_holdout),
        random_state=SETTINGS.training.seed,
        stratify=holdout_df["label"],
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


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

        logger.info("Merging datasets (ISOT + FakeNewsNet + LIAR)...")
        df = merge_datasets()
        ensure_dataframe(df, name="merged_df", required_columns=["text", "label"], min_rows=1)

        logger.info("Total samples loaded: %s", len(df))
        logger.info("Fake samples: %s", (df["label"] == 1).sum())
        logger.info("Real samples: %s", (df["label"] == 0).sum())
        df.to_csv(merged_dataset_path, index=False)

        logger.info("Cleaning dataset...")
        before_clean = len(df)
        df = clean_dataframe(df)
        ensure_dataframe(df, name="cleaned_df", required_columns=["text", "label"], min_rows=1)

        logger.info("Removed %s samples during cleaning", before_clean - len(df))
        logger.info("Dataset after cleaning: %s", len(df))
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

        logger.info("Cleaned dataset saved to %s", cleaned_dataset_path)
        logger.info("Data cleaning report saved to %s", cleaning_report_path)

        # --------------------------------------------------
        # Leakage-safe split BEFORE augmentation/features
        # --------------------------------------------------
        train_df, val_df, test_df = _split_clean_dataset(df)
        logger.info(
            "Leakage-safe split complete -> train=%s val=%s test=%s",
            len(train_df),
            len(val_df),
            len(test_df),
        )

        # --------------------------------------------------
        # Augmentation on train only
        # --------------------------------------------------
        augmentation_multiplier = SETTINGS.data.augmentation_multiplier
        if augmentation_multiplier > 1:
            logger.info("Applying data augmentation to training split only (multiplier=%s)", augmentation_multiplier)
            train_df = augment_dataset(train_df, text_column="text", multiplier=augmentation_multiplier)
            logger.info("Training split size after augmentation: %s", len(train_df))
        else:
            logger.info("Data augmentation skipped (multiplier <= 1)")

        # --------------------------------------------------
        # Feature engineering (fit on train only)
        # --------------------------------------------------
        tfidf_max_features = SETTINGS.features.tfidf_max_features
        tfidf_top_terms = SETTINGS.features.tfidf_top_terms_per_doc

        logger.info(
            "Applying leakage-safe feature pipeline (fit train only, transform val/test) "
            "(tfidf_max_features=%s, top_terms_per_doc=%s)",
            tfidf_max_features,
            tfidf_top_terms,
        )

        train_df, tfidf_vectorizer = fit_feature_pipeline(
            train_df,
            text_column="text",
            tfidf_max_features=tfidf_max_features,
            top_terms_per_doc=tfidf_top_terms,
        )
        val_df = transform_feature_pipeline(
            val_df,
            vectorizer=tfidf_vectorizer,
            text_column="text",
            top_terms_per_doc=tfidf_top_terms,
        )
        test_df = transform_feature_pipeline(
            test_df,
            vectorizer=tfidf_vectorizer,
            text_column="text",
            top_terms_per_doc=tfidf_top_terms,
        )
        save_vectorizer(tfidf_vectorizer, tfidf_vectorizer_path)

        text_column = SETTINGS.training.text_column
        if text_column not in train_df.columns:
            raise ValueError(
                f"Configured text column '{text_column}' does not exist. "
                f"Available columns: {list(train_df.columns)}"
            )

        # --------------------------------------------------
        # Optional CV (train split only)
        # --------------------------------------------------
        if SETTINGS.training.run_cross_validation:
            logger.info("Running cross-validation on training split only...")
            cv_results = cross_validate_model(
                train_df,
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
        # Optional tuning (train/val only)
        # --------------------------------------------------
        best_params = None
        if SETTINGS.training.run_hyperparameter_tuning:
            logger.info("Running hyperparameter tuning (train+val splits)...")
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
            logger.info(
                "Best parameters found (best_%s=%.4f): %s",
                tuning_results["metric_name"],
                tuning_results["best_value"],
                best_params,
            )

        logger.info("Training final model...")
        trainer, test_dataset = train_model(
            train_df,
            params=best_params,
            text_column=text_column,
            validation_df=val_df,
            test_df=test_df,
        )

        logger.info("Evaluating model...")
        prediction_output = trainer.predict(test_dataset)

        logits = prediction_output.predictions
        y_true = prediction_output.label_ids
        y_pred = np.argmax(logits, axis=1)

        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probabilities = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        fake_probabilities = probabilities[:, 1]

        evaluation_results = evaluate(y_true, y_pred, fake_probabilities)
        save_evaluation_results(evaluation_results, evaluation_results_path)

        fig, _ = plot_confusion_matrix(evaluation_results["confusion_matrix"])
        fig.savefig(confusion_matrix_path)
        plt.close(fig)

        logger.info("Evaluation report saved")
        logger.info("=" * 50)
        logger.info("Training Complete!")
        logger.info("=" * 50)
        logger.info("Model saved to %s", SETTINGS.model.path)
        logger.info("Run API using: uvicorn api.app:app --reload")

    except Exception as e:
        logger.error("Pipeline failed: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
