from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)
import logging
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate(y_true, y_pred, y_proba=None):
    """
    Comprehensive model evaluation
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional)
    
    Returns:
        dict: Evaluation metrics
    """
    try:
        results = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average='binary'),
            "recall": recall_score(y_true, y_pred, average='binary'),
            "f1": f1_score(y_true, y_pred, average='binary'),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
        }
        
        # Add ROC-AUC if probabilities provided
        if y_proba is not None:
            results["roc_auc"] = roc_auc_score(y_true, y_proba)
        
        # Classification report
        report = classification_report(y_true, y_pred, target_names=['Real', 'Fake'])
        results["classification_report"] = report
        
        logger.info("Evaluation Results:")
        logger.info(f"Accuracy: {results['accuracy']:.4f}")
        logger.info(f"Precision: {results['precision']:.4f}")
        logger.info(f"Recall: {results['recall']:.4f}")
        logger.info(f"F1 Score: {results['f1']:.4f}")
        if y_proba is not None:
            logger.info(f"ROC-AUC: {results['roc_auc']:.4f}")
        logger.info(f"\nClassification Report:\n{report}")
        
        return results
    
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise


def save_evaluation_results(results, output_path="reports/evaluation_results.json"):
    """Save evaluation results to JSON file"""
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        results_serializable = {}
        for key, value in results.items():
            if isinstance(value, (list, str, int, float)):
                results_serializable[key] = value
            elif hasattr(value, 'tolist'):
                results_serializable[key] = value.tolist()
            else:
                results_serializable[key] = str(value)
        
        with open(output_path, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        logger.info(f"Evaluation results saved to {output_path}")
    
    except Exception as e:
        logger.error(f"Error saving evaluation results: {e}")
        raise
