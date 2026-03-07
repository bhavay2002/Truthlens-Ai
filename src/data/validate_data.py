"""
Data validation utilities
Ensures data quality before training
"""
import pandas as pd
import logging
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataValidator:
    """Validates dataset quality and structure"""
    
    def __init__(self, required_columns: List[str] = None):
        self.required_columns = required_columns or ['text', 'label']
        self.validation_errors = []
    
    def validate_schema(self, df: pd.DataFrame) -> bool:
        """Check if dataframe has required columns"""
        missing_cols = set(self.required_columns) - set(df.columns)
        if missing_cols:
            error = f"Missing required columns: {missing_cols}"
            logger.error(error)
            self.validation_errors.append(error)
            return False
        return True
    
    def validate_nulls(self, df: pd.DataFrame, max_null_ratio: float = 0.1) -> bool:
        """Check for excessive null values"""
        null_ratios = df[self.required_columns].isnull().sum() / len(df)
        problematic = null_ratios[null_ratios > max_null_ratio]
        
        if not problematic.empty:
            error = f"Columns with excessive nulls: {problematic.to_dict()}"
            logger.warning(error)
            self.validation_errors.append(error)
            return False
        return True
    
    def validate_duplicates(self, df: pd.DataFrame, max_dup_ratio: float = 0.2) -> bool:
        """Check for excessive duplicates"""
        dup_count = df.duplicated(subset=['text']).sum()
        dup_ratio = dup_count / len(df)
        
        if dup_ratio > max_dup_ratio:
            warning = f"High duplicate ratio: {dup_ratio:.2%} ({dup_count} duplicates)"
            logger.warning(warning)
            self.validation_errors.append(warning)
            return False
        return True
    
    def validate_label_distribution(self, df: pd.DataFrame, min_ratio: float = 0.2) -> bool:
        """Check for severe class imbalance"""
        if 'label' not in df.columns:
            return True
        
        label_counts = df['label'].value_counts(normalize=True)
        min_class_ratio = label_counts.min()
        
        if min_class_ratio < min_ratio:
            warning = f"Severe class imbalance detected: {label_counts.to_dict()}"
            logger.warning(warning)
            self.validation_errors.append(warning)
            return False
        return True
    
    def validate_text_quality(self, df: pd.DataFrame, min_length: int = 10) -> bool:
        """Check text quality"""
        if 'text' not in df.columns:
            return True
        
        # Check for very short texts
        short_texts = (df['text'].str.len() < min_length).sum()
        short_ratio = short_texts / len(df)
        
        if short_ratio > 0.1:
            warning = f"Too many short texts: {short_ratio:.2%} ({short_texts} texts < {min_length} chars)"
            logger.warning(warning)
            self.validation_errors.append(warning)
            return False
        return True
    
    def validate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run all validations"""
        logger.info("Running data validation...")
        self.validation_errors = []
        
        results = {
            'schema_valid': self.validate_schema(df),
            'nulls_acceptable': self.validate_nulls(df),
            'duplicates_acceptable': self.validate_duplicates(df),
            'labels_balanced': self.validate_label_distribution(df),
            'text_quality_good': self.validate_text_quality(df),
            'errors': self.validation_errors
        }
        
        all_passed = all(v for k, v in results.items() if k != 'errors')
        results['all_passed'] = all_passed
        
        if all_passed:
            logger.info("✓ All validations passed!")
        else:
            logger.warning(f"⚠ Validation issues found: {len(self.validation_errors)}")
            for error in self.validation_errors:
                logger.warning(f"  - {error}")
        
        return results


def validate_dataset(csv_path: str) -> Dict[str, Any]:
    """Convenience function to validate a CSV file"""
    df = pd.read_csv(csv_path)
    validator = DataValidator()
    return validator.validate(df)
