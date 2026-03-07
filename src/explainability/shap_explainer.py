"""
SHAP Explainability Module for TruthLens AI
Provides feature importance explanations for transformer predictions
"""

import shap
import logging
from typing import List, Callable

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------
# Global SHAP Explainer
# -------------------------------------------------

_explainer = None


# -------------------------------------------------
# Prediction Wrapper
# -------------------------------------------------

def shap_predict_wrapper(texts: List[str], predict_fn: Callable):
    """
    Wrapper so SHAP can call the model.

    SHAP expects:
        list[str] -> probabilities
    """

    outputs = []

    for text in texts:
        result = predict_fn(text)

        fake_prob = result["fake_probability"]
        real_prob = 1 - fake_prob

        outputs.append([real_prob, fake_prob])

    return outputs


# -------------------------------------------------
# Initialize Explainer
# -------------------------------------------------

def get_explainer(predict_fn: Callable):
    """
    Initialize SHAP explainer once
    """

    global _explainer

    if _explainer is None:

        logger.info("Initializing SHAP explainer...")

        masker = shap.maskers.Text()

        _explainer = shap.Explainer(
            lambda x: shap_predict_wrapper(x, predict_fn),
            masker
        )

    return _explainer


# -------------------------------------------------
# Generate Explanation
# -------------------------------------------------

def explain_text(
    predict_fn: Callable,
    text: str
):
    """
    Generate SHAP explanation for text prediction
    """

    try:

        explainer = get_explainer(predict_fn)

        shap_values = explainer([text])

        logger.info("SHAP explanation generated")

        return shap_values

    except Exception as e:
        logger.error(f"SHAP explanation failed: {e}")
        raise


# -------------------------------------------------
# Plot Explanation
# -------------------------------------------------

def plot_explanation(
    predict_fn: Callable,
    text: str
):
    """
    Show SHAP explanation plot
    """

    shap_values = explain_text(predict_fn, text)

    shap.plots.text(shap_values[0])


# -------------------------------------------------
# Save HTML Visualization
# -------------------------------------------------

def save_explanation_html(
    predict_fn: Callable,
    text: str,
    output_path: str = "reports/shap_explanation.html"
):
    """
    Save SHAP explanation as HTML visualization
    """

    try:

        shap_values = explain_text(predict_fn, text)

        html = shap.plots.text(shap_values[0], display=False)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

        logger.info(f"SHAP explanation saved to {output_path}")

    except Exception as e:
        logger.error(f"Failed to save SHAP explanation: {e}")
        raise