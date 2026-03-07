"""
Run EDA Analysis on Merged Dataset
This script loads, merges, and analyzes the fake news dataset
"""
import sys
import json
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_data import merge_datasets
from src.data.eda import FakeNewsEDA
import logging

logging.basicConfig(level=logging.INFO)
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


def save_eda_report(eda, fake_path, real_path, output_path="reports/eda_report.json"):
    """Save EDA summary report to JSON."""
    df = eda.df.copy()

    if "text_length" not in df.columns and "text" in df.columns:
        df["text_length"] = df["text"].astype(str).str.len()
    if "word_count" not in df.columns and "text" in df.columns:
        df["word_count"] = df["text"].astype(str).str.split().str.len()

    report = {
        "data_files": {
            "fake_path": fake_path,
            "real_path": real_path
        },
        "rows": int(len(df)),
        "columns": list(df.columns),
        "label_distribution": {
            str(k): int(v) for k, v in df["label"].value_counts().to_dict().items()
        } if "label" in df.columns else {},
        "text_length_stats": {
            "mean": float(df["text_length"].mean()),
            "median": float(df["text_length"].median()),
            "min": int(df["text_length"].min()),
            "max": int(df["text_length"].max()),
        } if "text_length" in df.columns and len(df) else {},
        "word_count_stats": {
            "mean": float(df["word_count"].mean()),
            "median": float(df["word_count"].median()),
            "min": int(df["word_count"].min()),
            "max": int(df["word_count"].max()),
        } if "word_count" in df.columns and len(df) else {},
        "figures_dir": "reports/figures",
    }

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return output_file


def main():
    """Run full EDA on the dataset"""
    
    # Check if data files exist
    fake_path, real_path = resolve_data_paths()
    
    if not Path(fake_path).exists() or not Path(real_path).exists():
        logger.error("Data files not found!")
        logger.error("Please ensure one of the supported pairs exists:")
        logger.error("- data/raw/fake.csv + data/raw/real.csv")
        logger.error("- data/raw/Fake.csv + data/raw/True.csv")
        logger.error("- data/raw/dataset1/Fake.csv + data/raw/dataset1/True.csv")
        logger.error("Download from: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset")
        sys.exit(1)
    
    # Load and merge data
    logger.info("Loading datasets...")
    df = merge_datasets(fake_path, real_path)
    logger.info(f"Total samples: {len(df)}")
    
    # Run EDA
    logger.info("Starting EDA analysis...")
    eda = FakeNewsEDA(df)
    eda.run()
    report_path = save_eda_report(eda, fake_path, real_path)
    
    logger.info("\n" + "=" * 70)
    logger.info("EDA Complete! Check reports/figures/ for visualizations")
    logger.info(f"EDA report saved to: {report_path}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
