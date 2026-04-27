from __future__ import annotations

import hashlib
import json
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from src.graph.entity_graph import EntityGraphBuilder
from src.graph.narrative_graph_builder import NarrativeGraphBuilder
from src.graph.graph_analysis import GraphAnalyzer, canonicalize_weighted
from src.graph.graph_features import GraphFeatureExtractor, GraphFeatureExtractorConfig
from src.graph.temporal_graph import TemporalGraphAnalyzer
from src.graph.graph_explainer import GraphExplainer
from src.graph.graph_schema import GraphOutput, GraphStructure

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

    # G-P3: batch size used by ``run_batch`` when calling ``nlp.pipe``.
    batch_size: int = 32


# =========================================================
# SCHEMA HELPERS  (G-C2)
# =========================================================

def _to_graph_structure(
    graph: Optional[Dict[str, Dict[str, float]]],
) -> Optional[GraphStructure]:
    """Adapt a weighted adjacency dict to the ``GraphStructure`` schema.

    Used only at the boundary where ``GraphOutput`` is constructed.
    Returns ``None`` for empty input so the optional field stays absent
    rather than carrying a dummy node.
    """
    if not graph:
        return None

    nodes = sorted(
        set(graph.keys())
        | {n for nbrs in graph.values() for n in (nbrs.keys() if isinstance(nbrs, dict) else nbrs)}
    )

    if not nodes:
        return None

    edges: Dict[str, List[str]] = {}
    for n in nodes:
        nbrs = graph.get(n, {})
        if isinstance(nbrs, dict):
            edges[n] = sorted(nbrs.keys())
        else:
            edges[n] = sorted(nbrs)

    return GraphStructure(nodes=nodes, edges=edges)


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
    # =====================================================

    def config_fingerprint(self) -> str:

        try:
            payload = asdict(self.config)
        except TypeError:
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
    # ENTITY GRAPH (factored so run / run_batch share code)
    # =====================================================

    def _entity_graph_from_doc(self, doc) -> Dict[str, Dict[str, float]]:
        """Build the weighted entity graph from a pre-parsed spaCy ``Doc``.

        Mirrors :meth:`EntityGraphBuilder.build_graph` but skips the
        ``self.nlp(text)`` call so a batch of documents parsed once via
        ``nlp.pipe`` can share a single parser pass (G-P3).
        """

        graph: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

        for sent in doc.sents:
            ents = [
                ent.text.lower().strip()
                for ent in sent.ents
                if ent.text.strip()
            ]
            ents = list(dict.fromkeys(ents))

            for i, a in enumerate(ents):
                for b in ents[i + 1:]:
                    graph[a][b] += 1.0

        return canonicalize_weighted(graph)

    # =====================================================
    # MAIN
    # =====================================================

    def run(self, text: str) -> Dict[str, Any]:

        self._validate_text(text)

        if self.entity_graph_builder is not None:
            doc = self.entity_graph_builder.nlp(text)
        else:
            doc = None

        return self._run_with_doc(text, doc)

    # =====================================================
    # G-P3: batched variant
    # =====================================================

    def run_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Vectorised counterpart of :meth:`run`.

        Uses ``spacy.Language.pipe`` to parse the full batch in a single
        call instead of one-by-one; meaningful speedup for batch
        inference (`batch_inference.py`, `feature_pipeline.batch_extract`)
        which previously paid the per-doc spaCy overhead N times.
        Narrative / temporal stages still run per-doc — they're pure
        Python regex and dominated by entity-graph parsing in profile.
        """

        if not texts:
            return []

        for t in texts:
            self._validate_text(t)

        if self.entity_graph_builder is not None:
            docs = list(
                self.entity_graph_builder.nlp.pipe(
                    texts, batch_size=self.config.batch_size
                )
            )
        else:
            docs = [None] * len(texts)

        return [self._run_with_doc(t, d) for t, d in zip(texts, docs)]

    # =====================================================
    # SHARED IMPL
    # =====================================================

    def _run_with_doc(self, text: str, doc) -> Dict[str, Any]:

        entity_graph: Optional[Dict[str, Dict[str, float]]] = None
        narrative_graph: Optional[Dict[str, Dict[str, float]]] = None
        temporal_features: Optional[Dict[str, float]] = None

        # -------------------------------------------
        # ENTITY GRAPH  (already parsed)
        # -------------------------------------------
        if self.entity_graph_builder is not None and doc is not None:
            entity_graph = self._entity_graph_from_doc(doc)

        # -------------------------------------------
        # NARRATIVE GRAPH
        # -------------------------------------------
        if self.narrative_graph_builder is not None:
            narrative_graph = self.narrative_graph_builder.build_graph(text)
            # G-P1: canonicalize once at the top so every downstream
            # consumer (analyzer, embedding, explainer) skips repeating
            # the symmetrise / normalise pass.
            if narrative_graph:
                narrative_graph = canonicalize_weighted(narrative_graph)

        # -------------------------------------------
        # TEMPORAL GRAPH
        # -------------------------------------------
        if self.temporal_analyzer is not None:
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
        # GRAPH EXPLANATION  (G-C3)
        # -------------------------------------------
        explanation = None

        if self.graph_explainer is not None:
            try:
                explanation = self.graph_explainer.explain(
                    entity_graph=entity_graph,
                    narrative_graph=narrative_graph,
                    temporal_features=temporal_features,
                )
            except Exception:
                logger.exception("Graph explanation failed; continuing without it")
                explanation = None

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

        if self.analysis_runner is not None:
            analysis_modules = self.analysis_runner.analyze_text(text)

        # -------------------------------------------
        # FINAL WRAP (GraphOutput)  — G-C2
        # -------------------------------------------
        explanation_dict = explanation.to_dict() if explanation is not None else None

        try:
            graph_output = GraphOutput(
                # G-C2: previously passed raw weighted dicts as
                # ``entity_graph=``/``narrative_graph=`` and the schema
                # expected ``GraphStructure(nodes, edges)`` — pydantic
                # raised ``ValidationError`` on every request. Now
                # adapted at the boundary.
                entity_graph=_to_graph_structure(entity_graph),
                narrative_graph=_to_graph_structure(narrative_graph),
                temporal_features=temporal_features,
                entity_metrics=entity_metrics or None,
                narrative_metrics=narrative_metrics or None,
                features=features,
                embeddings=None,
                explanation=explanation_dict,
            )
        except Exception:
            # Defensive: never let a schema mismatch take down the
            # entire request — the rest of the result dict is still
            # useful even if the typed envelope failed to build.
            logger.exception("GraphOutput construction failed; returning raw dicts")
            graph_output = None

        result: Dict[str, Any] = {
            "graph_output": graph_output,
            "graph_features": features,
            # G-C4: previously the consumer
            # (`feature_pipeline._merge_graph_features`) read
            # ``entity_graph_metrics`` / ``narrative_graph_metrics`` but
            # the producer never emitted those keys, so per-graph metrics
            # were silently dropped before reaching the model. Now
            # surfaced as first-class result keys with the names the
            # consumer already expects.
            "entity_graph_metrics": entity_metrics,
            "narrative_graph_metrics": narrative_metrics,
        }

        if vector is not None:
            result["graph_feature_vector"] = vector

        if analysis_modules is not None:
            result["analysis_modules"] = analysis_modules

        if explanation is not None:
            result["graph_explanation"] = explanation_dict

        logger.debug(
            "GraphPipeline completed: %d features",
            len(features),
        )

        return result
