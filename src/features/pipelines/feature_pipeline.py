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

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.features.base.base_feature import BaseFeature, FeatureContext
from src.features.base.feature_registry import FeatureRegistry
from src.features.feature_bootstrap import bootstrap_feature_registry
from src.features.fusion.feature_fusion import FeatureFusion
from src.features.fusion.feature_scaling import FeatureScalingPipeline
from src.features.fusion.feature_selection import FeatureSelectionPipeline
from src.graph.graph_pipeline import GraphPipeline

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
    graph_pipeline: GraphPipeline | None = field(default=None, init=False, repr=False)

    def initialize(self) -> None:
        """
        Initialize feature extractors using FeatureRegistry.
        """
        bootstrap_feature_registry()

        if self.feature_names is None:
            self.feature_names = FeatureRegistry.list_features()

        self.features = [
            FeatureRegistry.create_feature(name) for name in self.feature_names
        ]

        self.fusion = FeatureFusion(self.features)
        try:
            self.graph_pipeline = GraphPipeline()
        except Exception as exc:  # noqa: BLE001
            logger.warning("GraphPipeline unavailable in FeaturePipeline: %s", exc)
            self.graph_pipeline = None

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

        if self.graph_pipeline is not None:
            try:
                graph_output = self.graph_pipeline.run(context.text)
                context.cache["graph_pipeline_output"] = graph_output

                graph_features = graph_output.get("graph_features", {})
                if isinstance(graph_features, dict):
                    for key, value in graph_features.items():
                        if isinstance(value, (int, float)):
                            features.setdefault(key, float(value))

                for section_name in ("entity_graph_metrics", "narrative_graph_metrics"):
                    section = graph_output.get(section_name, {})
                    if isinstance(section, dict):
                        for key, value in section.items():
                            if isinstance(value, (int, float)):
                                features.setdefault(f"graph_pipeline_{key}", float(value))
            except Exception as exc:  # noqa: BLE001
                logger.warning("GraphPipeline feature merge skipped: %s", exc)

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


def apply_feature_engineering(
    df: pd.DataFrame,
    *,
    text_column: str = "text",
    tfidf_max_features: int = 5000,
    top_terms_per_doc: int = 5,
    vectorizer: TfidfVectorizer | None = None,
) -> tuple[pd.DataFrame, TfidfVectorizer]:
    """
    Build engineered text from top TF-IDF terms per document.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    if text_column not in df.columns:
        raise ValueError(f"Missing text column: {text_column}")

    if top_terms_per_doc < 1:
        raise ValueError("top_terms_per_doc must be >= 1")

    texts = df[text_column].fillna("").astype(str).tolist()

    if vectorizer is None:
        vectorizer = TfidfVectorizer(max_features=tfidf_max_features)
        matrix = vectorizer.fit_transform(texts)
    else:
        matrix = vectorizer.transform(texts)

    feature_names = vectorizer.get_feature_names_out()

    engineered: list[str] = []

    for i in range(matrix.shape[0]):
        row = matrix.getrow(i)
        if row.nnz == 0:
            engineered.append("")
            continue

        order = row.data.argsort()[::-1][:top_terms_per_doc]
        top_indices = row.indices[order]
        terms = [str(feature_names[idx]) for idx in top_indices]
        engineered.append(" ".join(terms))

    output_df = df.copy()
    output_df["engineered_text"] = engineered

    return output_df, vectorizer


def transform_feature_pipeline(
    df: pd.DataFrame,
    *,
    vectorizer: TfidfVectorizer,
    text_column: str = "text",
    top_terms_per_doc: int = 5,
) -> pd.DataFrame:
    """
    Transform text data using an existing TF-IDF vectorizer.
    """

    transformed_df, _ = apply_feature_engineering(
        df,
        text_column=text_column,
        top_terms_per_doc=top_terms_per_doc,
        vectorizer=vectorizer,
    )
    return transformed_df
