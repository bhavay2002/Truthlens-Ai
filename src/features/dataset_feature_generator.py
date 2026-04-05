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

    Explicit integration of bias, framing, and ideological feature extractors:

        BiasFeatures (bias_*) — 10 features
            Loaded language, subjectivity, uncertainty, polarization,
            evaluative language, phrase-level bias signals.

        FramingFeatures (frame_*) — 10 features
            Economic, moral, security, human-interest, and conflict frames;
            frame diversity, dominance, and entropy.

        IdeologicalFeatures (ideology_*) — 8 features
            Left/right lexicon ratios, ideological balance and entropy,
            polarizing terms, group references, phrase-level signals.

    These are extracted automatically as part of the full pipeline run.
    Use generate_by_section() to obtain separate matrices per module
    section, or generate() for the combined matrix.

    Designed for large-scale dataset preprocessing and supports
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
from src.features.pipelines.feature_pipeline import (
    partition_feature_sections,
    BIAS_FEATURE_NAMES,
    FRAMING_FEATURE_NAMES,
    IDEOLOGICAL_FEATURE_NAMES,
)
from src.features.cache.cache_manager import CacheManager

from src.features.bias.bias_features import BiasFeatures          # noqa: F401
from src.features.bias.framing_features import FramingFeatures    # noqa: F401
from src.features.bias.ideological_features import IdeologicalFeatures  # noqa: F401

logger = logging.getLogger(__name__)

FeatureVector = Dict[str, float]


@dataclass
class DatasetFeatureGenerator:
    """
    High-level dataset feature generation system.

    Produces full feature matrices including BiasFeatures (bias_*),
    FramingFeatures (frame_*), and IdeologicalFeatures (ideology_*) from
    the underlying BatchFeaturePipeline. Supports:

        generate()           → numpy matrix of all features
        generate_by_section() → per-section matrices (bias, framing, ideology, …)
        generate_dataframe() → pandas DataFrame (optional dependency)
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

        The resulting matrix includes columns for all registered features,
        including BiasFeatures (bias_*), FramingFeatures (frame_*), and
        IdeologicalFeatures (ideology_*). Column order is sorted
        alphabetically and stored in self._feature_order.

        Parameters
        ----------
        texts : List[str]
            Raw article texts.
        labels : List[int], optional
            Supervision labels for fit=True scaler/selector.
        fit : bool
            If True, fit the scaler and selector on this batch.

        Returns
        -------
        matrix : np.ndarray of shape (n_samples, n_features)
        feature_names : List[str] sorted feature names
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

    def generate_by_section(
        self,
        texts: List[str],
        sections: Optional[List[str]] = None,
    ) -> Dict[str, Tuple[np.ndarray, List[str]]]:
        """
        Generate separate feature matrices for each module section.

        Runs the full pipeline once, then partitions the output into
        section-specific matrices. This is useful for training section-specific
        models (e.g. a bias classifier trained only on bias_* + frame_* +
        ideology_* features).

        Available sections:
            bias, framing, ideology, emotion, narrative, discourse, graph, other

        Parameters
        ----------
        texts : List[str]
            Raw article texts.
        sections : List[str], optional
            Subset of section names to return. If None, all non-empty
            sections are returned.

        Returns
        -------
        Dict[section_name, (matrix, feature_names)]
            Each value is a (np.ndarray, List[str]) tuple matching the
            shape returned by generate().
        """
        if not texts:
            raise ValueError("Input text list cannot be empty")

        contexts = self._build_contexts(texts)

        if not self.pipeline._initialized:
            self.pipeline.initialize()

        raw_features = self.pipeline._sequential_extract(contexts)

        partitioned: Dict[str, List[FeatureVector]] = {}
        for sample in raw_features:
            sample_sections = partition_feature_sections(sample)
            for sec_name, sec_features in sample_sections.items():
                partitioned.setdefault(sec_name, []).append(sec_features)

        result: Dict[str, Tuple[np.ndarray, List[str]]] = {}

        for sec_name, sec_samples in partitioned.items():
            if sections is not None and sec_name not in sections:
                continue

            if not sec_samples or not sec_samples[0]:
                continue

            sec_feature_names = sorted(sec_samples[0].keys())
            if not sec_feature_names:
                continue

            sec_matrix = np.array(
                [
                    [s.get(name, 0.0) for name in sec_feature_names]
                    for s in sec_samples
                ],
                dtype=np.float32,
            )

            result[sec_name] = (sec_matrix, sec_feature_names)

            logger.info(
                "Section '%s' matrix: shape=%s",
                sec_name,
                sec_matrix.shape,
            )

        return result

    def generate_dataframe(
        self,
        texts: List[str],
        labels: Optional[List[int]] = None,
        fit: bool = False,
    ):
        """
        Generate pandas DataFrame (if pandas available).

        Columns include all extracted feature names (bias_*, frame_*,
        ideology_*, emotion_*, etc.) plus an optional 'label' column.
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

    def get_bias_module_feature_names(self) -> Dict[str, List[str]]:
        """
        Return the output feature names from each bias module extractor.

        Returns
        -------
        Dict with keys 'bias', 'framing', 'ideology' mapping to
        their respective feature name lists.
        """
        return {
            "bias": BIAS_FEATURE_NAMES,
            "framing": FRAMING_FEATURE_NAMES,
            "ideology": IDEOLOGICAL_FEATURE_NAMES,
        }
