"""
File Name: batch_feature_pipeline.py
Module: Feature Engineering - Batch Feature Pipeline
Description:
    Implements a high-throughput batch feature extraction pipeline used for
    dataset-scale processing in the TruthLens system. The pipeline wraps the
    single-instance FeaturePipeline and provides:

        • parallel batch execution
        • deterministic ordering
        • progress-aware logging
        • optional fault tolerance
        • scalable dataset processing

    Integrates all feature extractor modules:

        BiasFeatures            → 10 features (bias_*)
        FramingFeatures         → 10 features (frame_*)
        IdeologicalFeatures     →  8 features (ideology_*)
        ArgumentStructureFeatures →  7 features (argument_*)
        DiscourseFeatures       →  7 features (discourse_*)
        EntityGraphFeatures     →  5 features (entity_*)
        InteractionGraphFeatures →  6 features (interaction_*)
        ConflictFeatures        →  9 features (conflict_*)
        NarrativeFeatures       → 11 features (narrative_*)
        NarrativeFrameFeatures  →  9 features (narrative_frame_*)
        NarrativeRoleFeatures   →  7 features (narrative_role_*)
        ManipulationPatterns    → 13 features (manipulation_*)
        PropagandaFeatures      → 11 features (propaganda_*)
        PropagandaLexiconFeatures → 11 features (propaganda_*)
        LexicalFeatures         →  5 features (vocabulary_/hapax_)
        SemanticFeatures        →  5 features (embedding_*)
        SyntacticFeatures       →  7 features (sentence_/pos_*)
        TokenFeatures           →  6 features (token_*)

    All extractors are auto-discovered via FeatureRegistry at initialization.
    The extract_by_section() method partitions each sample's output into
    named sections using partition_feature_sections() from feature_pipeline:
        bias, framing, ideology, emotion, discourse, graph,
        narrative, propaganda, text, other

    Designed for research experiments and production preprocessing jobs.

Dependencies:
    dataclasses
    typing
    logging
    multiprocessing
    itertools

Inputs:
    List[FeatureContext]

Outputs:
    List[Dict[str, float]] feature vectors
"""

from __future__ import annotations

import logging
import multiprocessing as mp
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.features.base.base_feature import FeatureContext
from src.features.pipelines.feature_pipeline import (
    FeaturePipeline,
    partition_feature_sections,
    BIAS_FEATURE_NAMES,
    FRAMING_FEATURE_NAMES,
    IDEOLOGICAL_FEATURE_NAMES,
    ARGUMENT_STRUCTURE_FEATURE_NAMES,
    DISCOURSE_FEATURE_NAMES,
    ENTITY_GRAPH_FEATURE_NAMES,
    INTERACTION_GRAPH_FEATURE_NAMES,
    CONFLICT_FEATURE_NAMES,
    NARRATIVE_FEATURE_NAMES,
    NARRATIVE_FRAME_FEATURE_NAMES,
    NARRATIVE_ROLE_FEATURE_NAMES,
    MANIPULATION_FEATURE_NAMES,
    PROPAGANDA_FEATURE_NAMES,
    PROPAGANDA_LEXICON_FEATURE_NAMES,
    LEXICAL_FEATURE_NAMES,
    SEMANTIC_FEATURE_NAMES,
    SYNTACTIC_FEATURE_NAMES,
    TOKEN_FEATURE_NAMES,
    ALL_BIAS_MODULE_FEATURE_NAMES,
    ALL_DISCOURSE_FEATURE_NAMES,
    ALL_GRAPH_FEATURE_NAMES,
    ALL_NARRATIVE_FEATURE_NAMES,
    ALL_PROPAGANDA_FEATURE_NAMES,
    ALL_TEXT_FEATURE_NAMES,
)

from src.features.bias.bias_features import BiasFeatures                        # noqa: F401
from src.features.bias.framing_features import FramingFeatures                  # noqa: F401
from src.features.bias.ideological_features import IdeologicalFeatures          # noqa: F401

from src.features.discourse.argument_structure_features import ArgumentStructureFeatures  # noqa: F401
from src.features.discourse.discourse_features import DiscourseFeatures         # noqa: F401

from src.features.graph.entity_graph_features import EntityGraphFeatures        # noqa: F401
from src.features.graph.interaction_graph_features import InteractionGraphFeatures  # noqa: F401

from src.features.narrative.conflict_features import ConflictFeatures           # noqa: F401
from src.features.narrative.narrative_features import NarrativeFeatures         # noqa: F401
from src.features.narrative.narrative_frame_features import NarrativeFrameFeatures  # noqa: F401
from src.features.narrative.narrative_role_features import NarrativeRoleFeatures    # noqa: F401

from src.features.propaganda.manipulation_patterns import ManipulationPatterns  # noqa: F401
from src.features.propaganda.propaganda_features import PropagandaFeatures      # noqa: F401
from src.features.propaganda.propaganda_lexicon_features import PropagandaLexiconFeatures  # noqa: F401

from src.features.text.lexical_features import LexicalFeatures                  # noqa: F401
from src.features.text.semantic_features import SemanticFeatures                # noqa: F401
from src.features.text.syntactic_features import SyntacticFeatures              # noqa: F401
from src.features.text.token_features import TokenFeatures                      # noqa: F401

logger = logging.getLogger(__name__)


def _worker_extract(args: tuple[FeaturePipeline, FeatureContext]) -> Dict[str, float]:
    """
    Worker function used for multiprocessing feature extraction.
    """
    pipeline, context = args
    return pipeline.extract(context)


@dataclass
class BatchFeaturePipeline:
    """
    High-throughput batch feature extraction system.

    Wraps FeaturePipeline to provide parallel extraction, fault-tolerance,
    and section-partitioned output. All registered feature modules —
    bias, framing, ideology, discourse, graph, narrative, propaganda,
    and text — are extracted as part of the normal pipeline run and are
    accessible via extract_by_section().

    Feature name constants for all modules are re-exported from this
    module for downstream convenience:
        BIAS_FEATURE_NAMES, FRAMING_FEATURE_NAMES, IDEOLOGICAL_FEATURE_NAMES,
        ARGUMENT_STRUCTURE_FEATURE_NAMES, DISCOURSE_FEATURE_NAMES,
        ENTITY_GRAPH_FEATURE_NAMES, INTERACTION_GRAPH_FEATURE_NAMES,
        CONFLICT_FEATURE_NAMES, NARRATIVE_FEATURE_NAMES,
        NARRATIVE_FRAME_FEATURE_NAMES, NARRATIVE_ROLE_FEATURE_NAMES,
        MANIPULATION_FEATURE_NAMES, PROPAGANDA_FEATURE_NAMES,
        PROPAGANDA_LEXICON_FEATURE_NAMES, LEXICAL_FEATURE_NAMES,
        SEMANTIC_FEATURE_NAMES, SYNTACTIC_FEATURE_NAMES, TOKEN_FEATURE_NAMES
    """

    pipeline: FeaturePipeline
    num_workers: int = 1
    chunk_size: int = 32
    fail_fast: bool = True

    _initialized: bool = field(default=False, init=False)

    def initialize(self) -> None:
        """
        Initialize underlying feature pipeline.
        """

        if not self._initialized:
            self.pipeline.initialize()
            self._initialized = True

            logger.info(
                "BatchFeaturePipeline initialized | workers=%d",
                self.num_workers,
            )

    def _sequential_extract(
        self,
        contexts: List[FeatureContext],
    ) -> List[Dict[str, float]]:
        """
        Sequential feature extraction.
        """

        results: List[Dict[str, float]] = []

        for ctx in contexts:
            try:
                features = self.pipeline.extract(ctx)
                results.append(features)

            except Exception:  # noqa: BLE001
                logger.exception("Feature extraction failed")

                if self.fail_fast:
                    raise

                results.append({})

        return results

    def _parallel_extract(
        self,
        contexts: List[FeatureContext],
    ) -> List[Dict[str, float]]:
        """
        Parallel feature extraction using multiprocessing.
        """

        logger.info(
            "Starting parallel feature extraction | samples=%d workers=%d",
            len(contexts),
            self.num_workers,
        )

        tasks = [(self.pipeline, ctx) for ctx in contexts]

        with mp.Pool(self.num_workers) as pool:
            results = pool.map(
                _worker_extract,
                tasks,
                chunksize=self.chunk_size,
            )

        return results

    def extract(
        self,
        contexts: List[FeatureContext],
    ) -> List[Dict[str, float]]:
        """
        Extract features for a dataset.

        Output dicts include contributions from all registered feature
        modules across bias, framing, ideology, discourse, graph,
        narrative, propaganda, and text groups.

        Parameters
        ----------
        contexts : List[FeatureContext]

        Returns
        -------
        List[Dict[str, float]]
        """

        if not contexts:
            raise ValueError("Input contexts cannot be empty")

        if not self._initialized:
            self.initialize()

        logger.info(
            "Batch feature extraction started | samples=%d",
            len(contexts),
        )

        if self.num_workers <= 1:
            results = self._sequential_extract(contexts)
        else:
            results = self._parallel_extract(contexts)

        logger.info(
            "Batch feature extraction completed | samples=%d",
            len(results),
        )

        return results

    def extract_by_section(
        self,
        contexts: List[FeatureContext],
    ) -> List[Dict[str, Dict[str, float]]]:
        """
        Extract features and partition each sample's output by module section.

        Returns one dict per sample with keys:
            bias, framing, ideology, emotion, discourse, graph,
            narrative, propaganda, text, other

        Parameters
        ----------
        contexts : List[FeatureContext]

        Returns
        -------
        List[Dict[str, Dict[str, float]]]
        """
        flat_results = self.extract(contexts)
        return [partition_feature_sections(f) for f in flat_results]

    def extract_with_labels(
        self,
        contexts: List[FeatureContext],
        labels: Optional[List[int]] = None,
        fit: bool = False,
    ) -> List[Dict[str, float]]:
        """
        Execute full pipeline including scaling and selection.

        All feature module outputs pass through the scaler and selector
        if configured.
        """

        if fit:
            features = self.pipeline.process(contexts, labels=labels, fit=True)
        else:
            features = self.pipeline.process(contexts, labels=labels, fit=False)

        return features
