"""
LIME Explanation Module for TruthLens AI
Provides interpretable explanations for model predictions
"""

import logging
from typing import Callable, Dict, Any, List
from lime.lime_text import LimeTextExplainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------
# Global Explainer (lazy initialization)
# -------------------------------------------------

_explainer = None


def get_explainer():
    """Initialize LIME explainer once"""
    global _explainer

    if _explainer is None:
        logger.info("Initializing LIME explainer...")
        _explainer = LimeTextExplainer(
            class_names=["Real", "Fake"]
        )

    return _explainer


# -------------------------------------------------
# Prediction Wrapper
# -------------------------------------------------

def lime_predict_wrapper(texts: List[str], predict_fn: Callable):
    """
    LIME requires a function that accepts list[str] and returns probabilities
    """

    probs = []

    for text in texts:
        result = predict_fn(text)

        # expected output from predict_fn
        # {"fake_probability": 0.7}

        fake_prob = result["fake_probability"]
        real_prob = 1 - fake_prob

        probs.append([real_prob, fake_prob])

    return probs


# -------------------------------------------------
# Generate Explanation
# -------------------------------------------------

def explain_prediction(
    predict_fn: Callable,
    text: str,
    num_features: int = 10
) -> Dict[str, Any]:
    """
    Generate LIME explanation for a text prediction

    Args:
        predict_fn : prediction function
        text : input text
        num_features : number of important words to show

    Returns:
        explanation dictionary
    """

    try:

        explainer = get_explainer()

        exp = explainer.explain_instance(
            text,
            lambda x: lime_predict_wrapper(x, predict_fn),
            num_features=num_features
        )

        explanation = {
            "text": text,
            "important_features": exp.as_list()
        }

        logger.info("LIME explanation generated")

        return explanation

    except Exception as e:
        logger.error(f"LIME explanation failed: {e}")
        raise


# -------------------------------------------------
# Save Explanation
# -------------------------------------------------

def save_explanation_html(
    predict_fn: Callable,
    text: str,
    output_path: str = "reports/lime_explanation.html",
    num_features: int = 10
):
    """
    Save LIME explanation as interactive HTML
    """

    try:

        explainer = get_explainer()

        exp = explainer.explain_instance(
            text,
            lambda x: lime_predict_wrapper(x, predict_fn),
            num_features=num_features
        )

        exp.save_to_file(output_path)

        logger.info(f"LIME explanation saved to {output_path}")

    except Exception as e:
        logger.error(f"Error saving LIME explanation: {e}")
        raise


# -------------------------------------------------
# Notebook Visualization
# -------------------------------------------------

def show_explanation_notebook(
    predict_fn: Callable,
    text: str,
    num_features: int = 10
):
    """
    Show explanation in Jupyter Notebook
    """

    explainer = get_explainer()

    exp = explainer.explain_instance(
        text,
        lambda x: lime_predict_wrapper(x, predict_fn),
        num_features=num_features
    )

    exp.show_in_notebook()