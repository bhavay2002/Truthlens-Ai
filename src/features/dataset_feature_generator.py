"""
File Name: dataset_feature_generator.py
Module: Feature Engineering - Dataset Feature Generator
Description:
    Generates feature matrices for entire datasets using the TruthLens
    feature engineering pipeline. This module orchestrates batch feature
    extraction, optional caching, scaling, and feature selection to produce
    ready-to-train feature matrices.

    The generator integrates with:
        • FeaturePipeline
        • BatchFeaturePipeline
        • CacheManager
        • FeatureScalingPipeline
        • FeatureSelectionPipeline

    It is designed for large-scale dataset preprocessing and supports
    deterministic, reproducible feature generation for ML experiments.

Dependencies:
    dataclasses
    typing
    logging
    numpy
    pandas (optional)

Inputs:
    List[str] texts
    Optional labels

Outputs:
    Feature matrix (numpy.ndarray)
    Feature names
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except Exception:  # noqa: BLE001
    PANDAS_AVAILABLE = False

from src.features.base.base_feature import FeatureContext
from src.features.pipelines.batch_feature_pipeline import BatchFeaturePipeline
from src.features.cache.cache_manager import CacheManager

logger = logging.getLogger(__name__)

FeatureVector = Dict[str, float]


@dataclass
class DatasetFeatureGenerator:
    """
    High-level dataset feature generation system.
    """

    pipeline: BatchFeaturePipeline
    cache_manager: Optional[CacheManager] = None
    cache_namespace: str = "dataset_features"

    _feature_order: List[str] = field(default_factory=list, init=False)

    def _build_contexts(self, texts: List[str]) -> List[FeatureContext]:
        """
        Convert raw texts to FeatureContext objects.
        """

        contexts = []

        for text in texts:
            if not text or not isinstance(text, str):
                raise ValueError("Input texts must be non-empty strings")

            contexts.append(FeatureContext(text=text))

        return contexts

    def _cached_extract(self, contexts: List[FeatureContext]) -> List[FeatureVector]:
        """
        Extract features with caching support.
        """

        results: List[FeatureVector] = []

        for ctx in contexts:

            if self.cache_manager:

                features = self.cache_manager.get_or_compute(
                    namespace=self.cache_namespace,
                    context=ctx,
                    compute_fn=self.pipeline.pipeline.extract,
                )

            else:

                features = self.pipeline.pipeline.extract(ctx)

            results.append(features)

        return results

    def generate(
        self,
        texts: List[str],
        labels: Optional[List[int]] = None,
        fit: bool = False,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Generate feature matrix from raw texts.
        """

        if not texts:
            raise ValueError("Input text list cannot be empty")

        contexts = self._build_contexts(texts)

        logger.info(
            "Dataset feature generation started | samples=%d",
            len(contexts),
        )

        if self.cache_manager:
            features = self._cached_extract(contexts)
            pipeline = self.pipeline.pipeline
            if pipeline.scaler:
                if fit:
                    pipeline.fit_scaler(features)
                features = pipeline.transform_scaler(features)
            if pipeline.selector:
                if fit:
                    pipeline.fit_selector(features, labels)
                features = pipeline.transform_selector(features)
        else:
            features = self.pipeline.pipeline.process(
                contexts,
                labels=labels,
                fit=fit,
            )

        if not features:
            raise RuntimeError("Feature extraction produced empty result")

        feature_names = sorted(features[0].keys())

        matrix = np.array(
            [[f.get(name, 0.0) for name in feature_names] for f in features],
            dtype=np.float32,
        )

        self._feature_order = feature_names

        logger.info(
            "Dataset feature generation completed | samples=%d features=%d",
            matrix.shape[0],
            matrix.shape[1],
        )

        return matrix, feature_names

    def generate_dataframe(
        self,
        texts: List[str],
        labels: Optional[List[int]] = None,
        fit: bool = False,
    ):
        """
        Generate pandas DataFrame (if pandas available).
        """

        if not PANDAS_AVAILABLE:
            raise RuntimeError("pandas is required for DataFrame output")

        matrix, feature_names = self.generate(texts, labels, fit)

        df = pd.DataFrame(matrix, columns=feature_names)

        if labels is not None:
            df["label"] = labels

        return df

    def get_feature_order(self) -> List[str]:
        """
        Retrieve stored feature order.
        """

        if not self._feature_order:
            raise RuntimeError("Features have not been generated yet")

        return self._feature_order