"""
File Name: feature_fusion.py
Module: Feature Engineering - Feature Fusion
Description:
    Combines outputs from multiple feature extractors into a unified
    feature representation. This module orchestrates feature extraction,
    handles safe execution, resolves feature name collisions, and
    produces a consistent fused feature dictionary.

    The fusion system integrates with the FeatureRegistry and BaseFeature
    abstractions to dynamically build feature pipelines. It supports:

        • modular feature execution
        • deterministic ordering
        • feature namespace management
        • error-safe extraction
        • optional normalization hooks

    This module acts as the central integration layer between
    feature extractors and downstream ML pipelines.

Dependencies:
    dataclasses
    typing
    logging

Inputs:
    FeatureContext
    List[BaseFeature]

Outputs:
    Dict[str, float] containing fused feature vector
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List

from src.features.base.base_feature import BaseFeature, FeatureContext

logger = logging.getLogger(__name__)


@dataclass
class FeatureFusion:
    """
    Orchestrates extraction and fusion of features from multiple
    feature extractors.
    """

    features: List[BaseFeature] = field(default_factory=list)
    enforce_unique_names: bool = True

    def _validate_feature_names(self) -> None:
        """
        Ensure that feature extractors have unique names to avoid
        collisions during fusion.
        """
        names = [f.name for f in self.features]

        if len(names) != len(set(names)):
            counts = Counter(names)
            duplicates = {name for name, cnt in counts.items() if cnt > 1}
            raise ValueError(f"Duplicate feature extractors detected: {duplicates}")

    def _ensure_initialized(self) -> None:
        if not hasattr(self, "_initialized"):
            for feature in self.features:
                feature.initialize()
            self._initialized = True

    def _ensure_validated(self) -> None:
        if self.enforce_unique_names and not hasattr(self, "_validated"):
            self._validate_feature_names()
            self._validated = True

    def extract(self, context: FeatureContext) -> Dict[str, float]:
        """
        Execute all feature extractors and fuse outputs.

        Parameters
        ----------
        context : FeatureContext

        Returns
        -------
        Dict[str, float]
            Unified feature dictionary.
        """

        self._ensure_validated()
        self._ensure_initialized()

        fused_features: Dict[str, float] = {}

        for feature in self.features:
            feature_output = feature.safe_extract(context)

            if False:  # disable completely in production
                overlapping_keys = fused_features.keys() & feature_output.keys()
                for key in overlapping_keys:
                    logger.warning("Feature collision detected: %s (overwriting)", key)

            for key, value in feature_output.items():
                fused_features[key] = float(value)

        logger.debug("Feature fusion completed | feature_count=%d", len(fused_features))

        return fused_features

    def extract_batch(self, contexts: List[FeatureContext]) -> List[Dict[str, float]]:
        """
        Extract fused features for a batch of contexts.
        """

        self._ensure_validated()
        self._ensure_initialized()

        results = [{} for _ in contexts]
        features_list = self.features

        for feature in features_list:

            if hasattr(feature, "extract_batch"):
                batch_outputs = feature.extract_batch(contexts)

                for index, feature_output in enumerate(batch_outputs):
                    if not feature_output:
                        continue

                    res = results[index]
                    res_local = res

                    for key, value in feature_output.items():
                        res_local[key] = float(value)

                continue

            for index, context in enumerate(contexts):
                feature_output = feature.safe_extract(context)

                if not feature_output:
                    continue

                res = results[index]
                res_local = res

                for key, value in feature_output.items():
                    res_local[key] = float(value)

        return results

    def list_features(self) -> List[str]:
        """
        List feature extractor names in fusion pipeline.
        """
        return [feature.name for feature in self.features]

    def add_feature(self, feature: BaseFeature) -> None:
        """
        Add a feature extractor to the pipeline.
        """
        self.features.append(feature)
        if hasattr(self, "_validated"):
            delattr(self, "_validated")
        if hasattr(self, "_initialized"):
            delattr(self, "_initialized")

    def remove_feature(self, feature_name: str) -> None:
        """
        Remove a feature extractor by name.
        """
        self.features = [f for f in self.features if f.name != feature_name]
        if hasattr(self, "_validated"):
            delattr(self, "_validated")
        if hasattr(self, "_initialized"):
            delattr(self, "_initialized")

    def clear(self) -> None:
        """
        Remove all feature extractors.
        """
        self.features.clear()
        if hasattr(self, "_validated"):
            delattr(self, "_validated")
        if hasattr(self, "_initialized"):
            delattr(self, "_initialized")