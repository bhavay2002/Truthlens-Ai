"""
File Name: feature_pipeline.py
Module: Feature Engineering - Feature Pipeline
Description:
    Implements the orchestrated feature extraction pipeline used across the
    TruthLens system. The pipeline coordinates feature discovery, execution,
    fusion, optional scaling, and optional feature selection.

    The pipeline integrates with:
        • BaseFeature abstractions
        • FeatureRegistry
        • FeatureFusion
        • FeatureScalingPipeline
        • FeatureSelectionPipeline

    This module is responsible for producing deterministic, reproducible
    feature vectors from raw text inputs.

Dependencies:
    dataclasses
    typing
    logging

Inputs:
    FeatureContext

Outputs:
    Dict[str, float] feature vector
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.features.base.base_feature import BaseFeature, FeatureContext
from src.features.base.feature_registry import FeatureRegistry
from src.features.fusion.feature_fusion import FeatureFusion
from src.features.fusion.feature_scaling import FeatureScalingPipeline
from src.features.fusion.feature_selection import FeatureSelectionPipeline

logger = logging.getLogger(__name__)


@dataclass
class FeaturePipeline:
    """
    Main feature extraction pipeline.

    Responsibilities:
        • initialize feature extractors
        • execute feature extraction
        • fuse outputs
        • optionally scale features
        • optionally apply feature selection
    """

    feature_names: Optional[List[str]] = None
    scaler: Optional[FeatureScalingPipeline] = None
    selector: Optional[FeatureSelectionPipeline] = None

    features: List[BaseFeature] = field(default_factory=list)
    fusion: Optional[FeatureFusion] = None

    def initialize(self) -> None:
        """
        Initialize feature extractors using FeatureRegistry.
        """

        if self.feature_names is None:
            self.feature_names = FeatureRegistry.list_features()

        self.features = [
            FeatureRegistry.create_feature(name) for name in self.feature_names
        ]

        self.fusion = FeatureFusion(self.features)

        logger.info(
            "FeaturePipeline initialized | feature_count=%d",
            len(self.features),
        )

    def extract(self, context: FeatureContext) -> Dict[str, float]:
        """
        Extract features from a single FeatureContext.
        """

        if self.fusion is None:
            raise RuntimeError("FeaturePipeline must be initialized before extraction")

        features = self.fusion.extract(context)

        logger.debug("Feature extraction completed | feature_count=%d", len(features))

        return features

    def batch_extract(self, contexts: List[FeatureContext]) -> List[Dict[str, float]]:
        """
        Extract features for multiple contexts.
        """

        if not contexts:
            raise ValueError("Context list cannot be empty")

        results = []

        for ctx in contexts:
            results.append(self.extract(ctx))

        logger.info(
            "Batch feature extraction completed | samples=%d",
            len(results),
        )

        return results

    def fit_scaler(self, features: List[Dict[str, float]]) -> None:
        """
        Fit scaling pipeline.
        """

        if self.scaler is None:
            raise RuntimeError("No scaler configured")

        self.scaler.fit(features)

        logger.info("Feature scaler fitted")

    def transform_scaler(
        self, features: List[Dict[str, float]]
    ) -> List[Dict[str, float]]:
        """
        Apply scaling transformation.
        """

        if self.scaler is None:
            return features

        return self.scaler.transform(features)

    def fit_selector(
        self,
        features: List[Dict[str, float]],
        labels: Optional[List[int]] = None,
    ) -> None:
        """
        Fit feature selector.
        """

        if self.selector is None:
            raise RuntimeError("No feature selector configured")

        self.selector.fit(features, labels)

        logger.info("Feature selector fitted")

    def transform_selector(
        self, features: List[Dict[str, float]]
    ) -> List[Dict[str, float]]:
        """
        Apply feature selection.
        """

        if self.selector is None:
            return features

        return self.selector.transform(features)

    def process(
        self,
        contexts: List[FeatureContext],
        labels: Optional[List[int]] = None,
        fit: bool = False,
    ) -> List[Dict[str, float]]:
        """
        Full pipeline execution.

        Steps:
            1. Feature extraction
            2. Optional scaling
            3. Optional feature selection
        """

        features = self.batch_extract(contexts)

        if self.scaler:

            if fit:
                self.fit_scaler(features)

            features = self.transform_scaler(features)

        if self.selector:

            if fit:
                self.fit_selector(features, labels)

            features = self.transform_selector(features)

        logger.info(
            "FeaturePipeline processing complete | samples=%d features=%d",
            len(features),
            len(features[0]) if features else 0,
        )

        return features