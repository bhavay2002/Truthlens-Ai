from src.data.load_data import merge_datasets
from src.data.clean_data import clean_dataframe
from src.models.train_roberta import train_model
from src.evaluation.evaluate_model import evaluate, save_evaluation_results
from src.visualization.visualize import plot_confusion_matrix
import logging
import sys
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Ensure log directory exists before FileHandler initialization
Path("logs").mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def resolve_data_paths():
    """Resolve supported fake/real dataset locations."""
    candidates = [
        ("data/raw/fake.csv", "data/raw/real.csv"),
        ("data/raw/Fake.csv", "data/raw/True.csv"),
        ("data/raw/dataset1/Fake.csv", "data/raw/dataset1/True.csv"),
    ]

    for fake_path, real_path in candidates:
        if Path(fake_path).exists() and Path(real_path).exists():
            return fake_path, real_path

    return candidates[0]


def main():
    """Main pipeline for fake news detection model training"""
    try:
        # Ensure directories exist
        Path("logs").mkdir(exist_ok=True)
        Path("models").mkdir(exist_ok=True)
        Path("data/processed").mkdir(parents=True, exist_ok=True)
        Path("reports").mkdir(parents=True, exist_ok=True)
        
        logger.info("=" * 50)
        logger.info("Starting TruthLens AI Training Pipeline")
        logger.info("=" * 50)
        
        # Check if data files exist
        fake_path, real_path = resolve_data_paths()
        
        if not Path(fake_path).exists() or not Path(real_path).exists():
            logger.error(f"Data files not found!")
            logger.error(f"Please ensure one of the supported pairs exists:")
            logger.error("- data/raw/fake.csv + data/raw/real.csv")
            logger.error("- data/raw/Fake.csv + data/raw/True.csv")
            logger.error("- data/raw/dataset1/Fake.csv + data/raw/dataset1/True.csv")
            logger.error("Sample datasets can be downloaded from Kaggle:")
            logger.error("https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset")
            sys.exit(1)
        
        # Load data
        logger.info("Loading datasets...")
        df = merge_datasets(fake_path, real_path)
        logger.info(f"Total samples loaded: {len(df)}")
        logger.info(f"Fake news samples: {(df['label'] == 1).sum()}")
        logger.info(f"Real news samples: {(df['label'] == 0).sum()}")
        
        # Clean data
        logger.info("Cleaning data...")
        df_before = len(df)
        df = clean_dataframe(df)
        logger.info(f"Removed {df_before - len(df)} samples during cleaning")
        logger.info(f"Final dataset size: {len(df)}")

        # Save cleaned data and cleaning report
        cleaned_data_path = Path("data/processed/cleaned_dataset.csv")
        df.to_csv(cleaned_data_path, index=False)
        cleaning_report = {
            "raw_rows": int(df_before),
            "cleaned_rows": int(len(df)),
            "rows_removed": int(df_before - len(df)),
            "retention_rate": float((len(df) / df_before) if df_before else 0.0),
            "label_distribution": {
                str(k): int(v) for k, v in df["label"].value_counts().to_dict().items()
            },
            "cleaned_data_path": str(cleaned_data_path),
        }
        cleaning_report_path = Path("reports/data_cleaning_report.json")
        with cleaning_report_path.open("w", encoding="utf-8") as f:
            json.dump(cleaning_report, f, indent=2)
        logger.info(f"Cleaned data saved to: {cleaned_data_path}")
        logger.info(f"Data cleaning report saved to: {cleaning_report_path}")
        
        # Train model
        logger.info("Starting model training...")
        trainer, test_dataset = train_model(df)
        
        # Evaluate on test set
        logger.info("Evaluating model on test set...")
        results = trainer.evaluate(test_dataset)

        # Save full evaluation report
        prediction_output = trainer.predict(test_dataset)
        logits = prediction_output.predictions
        y_true = prediction_output.label_ids
        y_pred = np.argmax(logits, axis=1)
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probabilities = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        fake_probabilities = probabilities[:, 1]

        evaluation_results = evaluate(y_true, y_pred, fake_probabilities)
        save_evaluation_results(evaluation_results, "reports/evaluation_results.json")
        fig, _ = plot_confusion_matrix(evaluation_results["confusion_matrix"])
        fig.savefig("reports/confusion_matrix.png")
        plt.close(fig)
        logger.info("Evaluation report saved to: reports/evaluation_results.json")
        logger.info("Confusion matrix saved to: reports/confusion_matrix.png")
        
        logger.info("=" * 50)
        logger.info("Training Complete! Results:")
        eval_loss = results.get("eval_loss")
        if eval_loss is not None:
            logger.info(f"Test Loss: {eval_loss:.4f}")
        else:
            logger.info("Test Loss: N/A")
        logger.info("=" * 50)
        logger.info("Model saved to: ./models/roberta_model")
        logger.info("You can now run the API with: uvicorn api.app:app --reload")
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
