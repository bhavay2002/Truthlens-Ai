import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_csv(path: str):
    """Load a CSV file with validation"""
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    
    try:
        df = pd.read_csv(path)
        logger.info(f"Loaded {len(df)} rows from {path}")
        return df
    except Exception as e:
        logger.error(f"Failed to load CSV from {path}: {e}")
        raise


def merge_datasets(fake_path, real_path):
    """Merge fake and real news datasets with validation"""
    # Validate paths
    fake_path_obj = Path(fake_path)
    real_path_obj = Path(real_path)
    
    if not fake_path_obj.exists():
        raise FileNotFoundError(f"Fake news dataset not found: {fake_path}")
    if not real_path_obj.exists():
        raise FileNotFoundError(f"Real news dataset not found: {real_path}")
    
    try:
        fake = pd.read_csv(fake_path)
        real = pd.read_csv(real_path)
        
        logger.info(f"Loaded {len(fake)} fake news articles")
        logger.info(f"Loaded {len(real)} real news articles")
        
        fake["label"] = 1
        real["label"] = 0
        
        df = pd.concat([fake, real], ignore_index=True)
        
        logger.info(f"Merged dataset contains {len(df)} total articles")
        
        return df
    except Exception as e:
        logger.error(f"Failed to merge datasets: {e}")
        raise
