"""
File: feature_importance.py

Purpose
-------
Compute feature importance for TruthLens AI models.

This module analyzes trained models and estimates which
feature categories contribute most to predictions.

Useful for:
- research analysis
- feature engineering evaluation
- explainability reports
"""

import logging
from typing import Dict, List
import numpy as np


# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Feature Category Mapping
# ---------------------------------------------------------

def categorize_feature(feature_name: str) -> str:
    """
    Determine category of a feature based on token prefix.
    """

    if feature_name.startswith("kw_"):
        return "keywords"

    if feature_name.startswith("sem_bias"):
        return "bias"

    if feature_name.startswith("sem_emotion") or feature_name.startswith("sem_top_emo"):
        return "emotion"

    if feature_name.startswith("meta_"):
        return "metadata"

    if feature_name.startswith("meta_domain") or feature_name.startswith("meta_srccred"):
        return "source"

    return "other"


# ---------------------------------------------------------
# Compute Feature Importance
# ---------------------------------------------------------

def compute_feature_importance(
    model,
    feature_names: List[str],
) -> Dict[str, float]:
    """
    Compute feature importance grouped by feature category.

    Parameters
    ----------
    model : trained sklearn model

    feature_names : list[str]

    Returns
    -------
    dict
        feature importance by category
    """

    try:

        logger.info("Computing feature importance")

        if not hasattr(model, "coef_"):
            raise ValueError("Model does not support coefficient-based importance")

        importance = np.abs(model.coef_[0])

        category_scores = {}

        for idx, feature in enumerate(feature_names):

            category = categorize_feature(feature)

            score = importance[idx]

            category_scores.setdefault(category, 0.0)

            category_scores[category] += score

        total = sum(category_scores.values())

        if total > 0:

            category_scores = {
                k: v / total
                for k, v in category_scores.items()
            }

        logger.info("Feature importance computed")

        return category_scores

    except Exception:

        logger.exception("Feature importance computation failed")

        raise


# ---------------------------------------------------------
# Top Features
# ---------------------------------------------------------

def get_top_features(
    model,
    feature_names: List[str],
    top_k: int = 20,
):
    """
    Return top individual features by importance.
    """

    if not hasattr(model, "coef_"):
        raise ValueError("Model does not support coefficient-based importance")

    importance = np.abs(model.coef_[0])

    indices = np.argsort(importance)[::-1][:top_k]

    return [
        (feature_names[i], float(importance[i]))
        for i in indices
    ]