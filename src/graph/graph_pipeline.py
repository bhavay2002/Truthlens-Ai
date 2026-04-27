from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from typing import Dict, Any, Optional

import numpy as np

from src.graph.entity_graph import EntityGraphBuilder
from src.graph.narrative_graph_builder import NarrativeGraphBuilder
from src.graph.graph_analysis import GraphAnalyzer
from src.graph.graph_features import GraphFeatureExtractor, GraphFeatureExtractorConfig
from src.graph.temporal_graph import TemporalGraphAnalyzer
from src.graph.graph_explainer import GraphExplainer
from src.graph.graph_schema import GraphOutput

from src.analysis.integration_runner import AnalysisIntegrationRunner

logger = logging.getLogger(__name__)


# =========================================================
# CONFIG
# =========================================================

@dataclass(slots=True)
class GraphPipelineConfig:

    enable_entity_graph: bool = True
    enable_narrative_graph: bool = True
    enable_temporal_graph: bool = True

    enable_graph_explainer: bool = True

    return_vector: bool = True
    run_analysis_modules: bool = True


# =========================================================
# PIPELINE
# =========================================================

class GraphPipeline:

    def __init__(self, config: Optional[GraphPipelineConfig] = None):

        self.config = config or GraphPipelineConfig()

        # -------------------------
        # Builders
        # -------------------------
        self.entity_graph_builder = (
            EntityGraphBuilder() if self.config.enable_entity_graph else None
        )

        self.narrative_graph_builder = (
            NarrativeGraphBuilder() if self.config.enable_narrative_graph else None
        )

        self.temporal_analyzer = (
            TemporalGraphAnalyzer() if self.config.enable_temporal_graph else None
        )

        self.graph_analyzer = GraphAnalyzer()

        self.graph_feature_extractor = GraphFeatureExtractor(
            GraphFeatureExtractorConfig(
                enable_entity_graph=self.config.enable_entity_graph,
                enable_narrative_graph=self.config.enable_narrative_graph,
            )
        )

        self.graph_explainer = (
            GraphExplainer() if self.config.enable_graph_explainer else None
        )

        self.analysis_runner = (
            AnalysisIntegrationRunner()
            if self.config.run_analysis_modules
            else None
        )

        logger.info("GraphPipeline initialized")

    # =====================================================
    # CONFIG FINGERPRINT (audit fix #1.2)
    #
    # Stable short hash over every public dataclass field of
    # GraphPipelineConfig.  Embedded in graph cache keys so flipping
    # any toggle (entity / narrative / temporal / vector / explainer /
    # analysis-modules) automatically invalidates stale cache entries
    # — otherwise switching the entity NER model or narrative lexicon
    # silently returned yesterday's graph features.
    # =====================================================

    def config_fingerprint(self) -> str:

        try:
            payload = asdict(self.config)
        except TypeError:
            # Defensive: should not happen because GraphPipelineConfig
            # is a @dataclass, but an out-of-band override could swap
            # in a non-dataclass.  Fall back to the public attribute
            # dict so the fingerprint is still deterministic.
            payload = {
                k: getattr(self.config, k)
                for k in dir(self.config)
                if not k.startswith("_")
                and not callable(getattr(self.config, k))
            }

        raw = json.dumps(payload, sort_keys=True, default=str).encode()
        return hashlib.sha256(raw).hexdigest()[:16]

    # =====================================================
    # VALIDATION
    # =====================================================

    def _validate_text(self, text: str):

        if not isinstance(text, str):
            raise TypeError("text must be string")

        if not text.strip():
            raise ValueError("text must be non-empty")

    # =====================================================
    # MAIN
    # =====================================================

    def run(self, text: str) -> Dict[str, Any]:

        self._validate_text(text)

        entity_graph = None
        narrative_graph = None
        temporal_features = None

        # -------------------------------------------
        # ENTITY GRAPH
        # -------------------------------------------
        if self.entity_graph_builder:
            entity_graph = self.entity_graph_builder.build_graph(text)

        # -------------------------------------------
        # NARRATIVE GRAPH
        # -------------------------------------------
        if self.narrative_graph_builder:
            narrative_graph = self.narrative_graph_builder.build_graph(text)

        # -------------------------------------------
        # TEMPORAL GRAPH
        # -------------------------------------------
        if self.temporal_analyzer:
            temporal_features = self.temporal_analyzer.analyze(text).to_dict()

        # -------------------------------------------
        # GRAPH METRICS
        # -------------------------------------------
        entity_metrics = (
            self.graph_analyzer.analyze(entity_graph).to_dict()
            if entity_graph
            else {}
        )

        narrative_metrics = (
            self.graph_analyzer.analyze(narrative_graph).to_dict()
            if narrative_graph
            else {}
        )

        # -------------------------------------------
        # GRAPH FEATURES
        # -------------------------------------------
        features = self.graph_feature_extractor.extract_from_graphs(
            entity_graph=entity_graph,
            narrative_graph=narrative_graph,
        )

        # merge temporal features
        if temporal_features:
            features.update(temporal_features)

        # -------------------------------------------
        # GRAPH EXPLANATION 🔥
        # -------------------------------------------
        explanation = None

        if self.graph_explainer:
            explanation = self.graph_explainer.explain(
                entity_graph=entity_graph,
                narrative_graph=narrative_graph,
                temporal_features=temporal_features,
            )

        # -------------------------------------------
        # FEATURE VECTOR
        # -------------------------------------------
        vector = None

        if self.config.return_vector:
            try:
                vector = self.graph_feature_extractor.extract_feature_vector_from_features(
                    features
                )
            except Exception as exc:
                logger.exception("Vector creation failed")
                raise RuntimeError("Graph vector failed") from exc

        # -------------------------------------------
        # ANALYSIS MODULES
        # -------------------------------------------
        analysis_modules = None

        if self.analysis_runner:
            analysis_modules = self.analysis_runner.analyze_text(text)

        # -------------------------------------------
        # 🔥 FINAL WRAP (GraphOutput)
        # -------------------------------------------
        graph_output = GraphOutput(
            entity_graph=entity_graph,
            narrative_graph=narrative_graph,
            temporal_features=temporal_features,
            entity_metrics=entity_metrics,
            narrative_metrics=narrative_metrics,
            features=features,
            embeddings=None,  # handled inside features if enabled
            explanation=explanation,
        )

        result = {
            "graph_output": graph_output,
            "graph_features": features,
        }

        if vector is not None:
            result["graph_feature_vector"] = vector

        if analysis_modules is not None:
            result["analysis_modules"] = analysis_modules

        if explanation is not None:
            result["graph_explanation"] = explanation

        logger.debug(
            "GraphPipeline completed: %d features",
            len(features),
        )

        return result