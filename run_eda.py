"""
Run EDA Analysis on Merged Dataset
This script loads, merges, and analyzes the fake news dataset
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.load_data import merge_datasets
from src.data.eda import FakeNewsEDA
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Run full EDA on the dataset"""
    
    # Check if data files exist
    fake_path = "data/raw/fake.csv"
    real_path = "data/raw/real.csv"
    
    if not Path(fake_path).exists() or not Path(real_path).exists():
        logger.error("Data files not found!")
        logger.error(f"Please ensure {fake_path} and {real_path} exist")
        logger.error("Download from: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset")
        sys.exit(1)
    
    # Load and merge data
    logger.info("Loading datasets...")
    df = merge_datasets(fake_path, real_path)
    logger.info(f"Total samples: {len(df)}")
    
    # Run EDA
    logger.info("Starting EDA analysis...")
    eda = FakeNewsEDA(df)
    report = eda.generate_full_report(output_dir='reports/figures')
    
    logger.info("\n" + "=" * 70)
    logger.info("EDA Complete! Check reports/figures/ for visualizations")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
