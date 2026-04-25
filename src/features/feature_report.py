from __future__ import annotations

"""
Feature Report Module

Generates diagnostics and summaries for feature datasets.

Includes:
- dataset summary
- top variance features
- constant features
- skewness analysis
- correlation warnings

Designed for:
- debugging
- experiment tracking
- feature quality monitoring
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any

import numpy as np

from src.features.feature_statistics import FeatureStatistics

logger = logging.getLogger(__name__)

FeatureVector = Dict[str, float]


# =========================================================
# REPORT
# =========================================================

@dataclass
class FeatureReport:
    """
    Generate feature diagnostics and reports.
    """

    top_k: int = 20
    skew_threshold: float = 2.0
    variance_threshold: float = 1e-6

    # -----------------------------------------------------

    def generate(
        self,
        features: List[FeatureVector],
        *,
        save_path: str | Path | None = None,
    ) -> Dict[str, Any]:
        """
        Generate full feature report.
        """

        if not features:
            raise ValueError("Feature list cannot be empty")

        stats = FeatureStatistics()

        logger.info("Generating feature report...")

        # -------------------------------------------------
        # BASIC STATS
        # -------------------------------------------------

        summary = stats.dataset_summary(features)
        variance = stats.compute_variance(features)
        skewness = stats.compute_skewness(features)
        constant = stats.detect_constant_features(features)

        # -------------------------------------------------
        # TOP FEATURES (by variance)
        # -------------------------------------------------

        sorted_variance = sorted(
            variance.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        top_features = sorted_variance[: self.top_k]

        # -------------------------------------------------
        # LOW VARIANCE FEATURES
        # -------------------------------------------------

        low_variance = [
            k for k, v in variance.items()
            if v < self.variance_threshold
        ]

        # -------------------------------------------------
        # HIGH SKEW FEATURES
        # -------------------------------------------------

        high_skew = [
            (k, v) for k, v in skewness.items()
            if abs(v) > self.skew_threshold
        ]

        # -------------------------------------------------
        # CORRELATION WARNINGS
        # -------------------------------------------------

        corr_matrix, keys = stats.compute_correlation_matrix(features)

        high_corr_pairs = []

        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                if abs(corr_matrix[i, j]) > 0.95:
                    high_corr_pairs.append(
                        (keys[i], keys[j], float(corr_matrix[i, j]))
                    )

        # -------------------------------------------------
        # FINAL REPORT
        # -------------------------------------------------

        report: Dict[str, Any] = {
            "summary": summary,
            "top_features": top_features,
            "low_variance_features": low_variance[:50],
            "constant_features": constant[:50],
            "high_skew_features": high_skew[:50],
            "high_correlation_pairs": high_corr_pairs[:50],
        }

        logger.info(
            "Feature report generated | features=%d samples=%d",
            int(summary["num_features"]),
            int(summary["num_samples"]),
        )

        # -------------------------------------------------
        # SAVE (OPTIONAL)
        # -------------------------------------------------

        if save_path:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            with open(path, "w") as f:
                json.dump(report, f, indent=2)

            logger.info("Feature report saved: %s", path)

        return report