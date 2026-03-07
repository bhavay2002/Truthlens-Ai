from src.data.load_data import merge_datasets
from src.data.clean_data import clean_dataframe
from src.models.train_roberta import train_model
import logging
import sys
from pathlib import Path

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


def main():
    """Main pipeline for fake news detection model training"""
    try:
        # Ensure directories exist
        Path("logs").mkdir(exist_ok=True)
        Path("models").mkdir(exist_ok=True)
        Path("data/processed").mkdir(parents=True, exist_ok=True)
        
        logger.info("=" * 50)
        logger.info("Starting TruthLens AI Training Pipeline")
        logger.info("=" * 50)
        
        # Check if data files exist
        fake_path = "data/raw/fake.csv"
        real_path = "data/raw/real.csv"
        
        if not Path(fake_path).exists() or not Path(real_path).exists():
            logger.error(f"Data files not found!")
            logger.error(f"Please ensure {fake_path} and {real_path} exist")
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
        
        # Train model
        logger.info("Starting model training...")
        trainer, test_dataset = train_model(df)
        
        # Evaluate on test set
        logger.info("Evaluating model on test set...")
        results = trainer.evaluate(test_dataset)
        
        logger.info("=" * 50)
        logger.info("Training Complete! Results:")
        logger.info(f"Test Loss: {results.get('eval_loss', 'N/A'):.4f}")
        logger.info("=" * 50)
        logger.info("Model saved to: ./models/roberta_model")
        logger.info("You can now run the API with: uvicorn api.app:app --reload")
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
