from __future__ import annotations

import hashlib
import json
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.graph.entity_graph import EntityGraphBuilder
from src.graph.narrative_graph_builder import NarrativeGraphBuilder
from src.graph.graph_analysis import GraphAnalyzer, canonicalize_weighted
from src.graph.graph_features import GraphFeatureExtractor, GraphFeatureExtractorConfig
from src.graph.temporal_graph import TemporalGraphAnalyzer
from src.graph.graph_explainer import GraphExplainer
from src.graph.graph_schema import GraphOutput, GraphStructure
from src.graph.graph_embeddings import GraphEmbeddingConfig
from src.graph.graph_config import GraphConfig, load_default_graph_config

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

    # G-CFG2: tunables previously baked into ``__init__`` defaults of
    # the various builders. Surfaced here so a YAML ``graph:`` block
    # can drive them end-to-end without monkey-patching.
    min_keyword_length: int = 4
    max_keywords_per_sentence: int = 4
    temporal_min_token_length: int = 4

    enable_graph_embeddings: bool = False
    embedding_type: str = "hybrid"
    spectral_dim: int = 8
    embedding_dim: int = 16
    walk_length: int = 10
    num_walks: int = 10

    explainer_node_weight: float = 0.4
    explainer_edge_weight: float = 0.3
    explainer_temporal_weight: float = 0.3

    # =====================================================
    # G-CFG1: translate the YAML-aware ``GraphConfig`` into
    # the runtime config the pipeline actually consumes.
    # =====================================================
    @classmethod
    def from_graph_config(cls, cfg: "GraphConfig") -> "GraphPipelineConfig":
        return cls(
            enable_entity_graph=cfg.enable_entity_graph,
            enable_narrative_graph=cfg.enable_narrative_graph,
            enable_temporal_graph=cfg.enable_temporal_graph,
            enable_graph_explainer=cfg.enable_graph_explainer,
            return_vector=cfg.return_vector,
            run_analysis_modules=cfg.run_analysis_modules,
            batch_size=cfg.batch_size,
            min_keyword_length=cfg.min_keyword_length,
            max_keywords_per_sentence=cfg.max_keywords_per_sentence,
            temporal_min_token_length=cfg.temporal_min_token_length,
            enable_graph_embeddings=cfg.enable_graph_embeddings,
            embedding_type=cfg.embedding_type,
            spectral_dim=cfg.spectral_dim,
            embedding_dim=cfg.embedding_dim,
            walk_length=cfg.walk_length,
            num_walks=cfg.num_walks,
            explainer_node_weight=cfg.explainer_node_weight,
            explainer_edge_weight=cfg.explainer_edge_weight,
            explainer_temporal_weight=cfg.explainer_temporal_weight,
        )

    @classmethod
    def from_yaml(cls, path: str | None = None) -> "GraphPipelineConfig":
        return cls.from_graph_config(load_default_graph_config(path))


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

        # G-CFG1: when no explicit config is supplied, hydrate from the
        # YAML ``graph:`` block so a single edit in ``config/config.yaml``
        # actually drives runtime behaviour. ``load_default_graph_config``
        # falls back to dataclass defaults if the YAML is missing.
        if config is None:
            config = GraphPipelineConfig.from_yaml()

        self.config = config

        # -------------------------
        # Builders — G-CFG2: every tunable now flows from self.config.
        # -------------------------
        self.entity_graph_builder = (
            EntityGraphBuilder() if self.config.enable_entity_graph else None
        )

        self.narrative_graph_builder = (
            NarrativeGraphBuilder(
                min_token_length=self.config.min_keyword_length,
                max_keywords_per_sentence=self.config.max_keywords_per_sentence,
            )
            if self.config.enable_narrative_graph
            else None
        )

        self.temporal_analyzer = (
            TemporalGraphAnalyzer(
                min_token_length=self.config.temporal_min_token_length,
            )
            if self.config.enable_temporal_graph
            else None
        )

        self.graph_analyzer = GraphAnalyzer()

        self.graph_feature_extractor = GraphFeatureExtractor(
            GraphFeatureExtractorConfig(
                enable_entity_graph=self.config.enable_entity_graph,
                enable_narrative_graph=self.config.enable_narrative_graph,
                enable_embeddings=self.config.enable_graph_embeddings,
                embedding_config=GraphEmbeddingConfig(
                    embedding_type=self.config.embedding_type,
                    spectral_dim=self.config.spectral_dim,
                    embedding_dim=self.config.embedding_dim,
                    walk_length=self.config.walk_length,
                    num_walks=self.config.num_walks,
                ),
            )
        )

        self.graph_explainer = (
            GraphExplainer(
                node_weight=self.config.explainer_node_weight,
                edge_weight=self.config.explainer_edge_weight,
                temporal_weight=self.config.explainer_temporal_weight,
            )
            if self.config.enable_graph_explainer
            else None
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

    def _entity_graph_from_doc(
        self,
        doc,
    ) -> Tuple[Dict[str, Dict[str, float]], List[Dict[str, Any]]]:
        """Build the weighted entity graph from a pre-parsed spaCy ``Doc``.

        Mirrors :meth:`EntityGraphBuilder.build_graph_with_spans` but
        skips the ``self.nlp(text)`` call so a batch of documents
        parsed once via ``nlp.pipe`` can share a single parser pass
        (G-P3). Also returns per-mention character spans (G-T1).
        """

        graph: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        spans: List[Dict[str, Any]] = []

        for s_idx, sent in enumerate(doc.sents):

            seen: set = set()
            ents: List[str] = []

            for ent in sent.ents:
                key = ent.text.lower().strip()
                if not key:
                    continue

                spans.append(
                    {
                        "entity": key,
                        "raw_text": ent.text,
                        "start_char": int(ent.start_char),
                        "end_char": int(ent.end_char),
                        "sentence_index": s_idx,
                        "label": getattr(ent, "label_", "") or "",
                    }
                )

                if key not in seen:
                    ents.append(key)
                    seen.add(key)

            for i, a in enumerate(ents):
                for b in ents[i + 1:]:
                    graph[a][b] += 1.0

        return canonicalize_weighted(graph), spans

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

        # G-T1 / G-T2: per-mention character spans for entity & narrative
        # nodes, surfaced through the result dict so the API / explainer
        # layer can map node IDs back to highlightable text regions.
        entity_spans: List[Dict[str, Any]] = []
        narrative_spans: List[Dict[str, Any]] = []
        narrative_tokenizer: Optional[str] = None

        # -------------------------------------------
        # ENTITY GRAPH  (already parsed) — G-T1: spans surfaced
        # -------------------------------------------
        if self.entity_graph_builder is not None and doc is not None:
            entity_graph, entity_spans = self._entity_graph_from_doc(doc)

        # -------------------------------------------
        # NARRATIVE GRAPH — G-S1 / G-T2: spaCy-aligned, span-aware
        # -------------------------------------------
        if self.narrative_graph_builder is not None:
            narrative_payload = (
                self.narrative_graph_builder.build_graph_with_spans(text)
            )
            narrative_graph = narrative_payload["graph"]
            narrative_spans = narrative_payload.get("spans", [])
            narrative_tokenizer = narrative_payload.get("tokenizer")

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
        # GRAPH FEATURES — G-R2: pass through pre-computed metrics so
        # ``extract_from_graphs`` does not run ``GraphAnalyzer.analyze``
        # a second time on the same graph.
        # -------------------------------------------
        features = self.graph_feature_extractor.extract_from_graphs(
            entity_graph=entity_graph,
            narrative_graph=narrative_graph,
            entity_metrics=entity_metrics,
            narrative_metrics=narrative_metrics,
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
            # G-T1 / G-T2: per-mention character spans so the API /
            # explainer can highlight node IDs back into the source text.
            "entity_spans": entity_spans,
            "narrative_spans": narrative_spans,
            "narrative_tokenizer": narrative_tokenizer,
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


# =========================================================
# SINGLETON  (G-R1)
# =========================================================
#
# ``GraphPipeline`` instantiation pulls in 6 builders + 15 analysis
# modules (``AnalysisIntegrationRunner``). The audit found 7 callsites
# across the codebase, each holding its own copy — same parsers, same
# spaCy reference, same module registry. Switching the default to a
# singleton collapses that to one per process while still allowing
# tests / advanced callers to inject their own ``GraphPipeline`` for
# isolation. Reset via ``reset_default_pipeline()`` between tests.

_DEFAULT_PIPELINE: Optional[GraphPipeline] = None


def get_default_pipeline() -> GraphPipeline:
    """Return the process-wide ``GraphPipeline`` singleton.

    Lazily constructed on first call so import order does not force
    ``AnalysisIntegrationRunner`` to load before its dependencies are
    ready. Safe to call from many threads — Python's GIL makes the
    None-check + assignment effectively atomic for our use here.
    """
    global _DEFAULT_PIPELINE
    if _DEFAULT_PIPELINE is None:
        _DEFAULT_PIPELINE = GraphPipeline()
    return _DEFAULT_PIPELINE


def reset_default_pipeline() -> None:
    """Drop the cached singleton — used by tests."""
    global _DEFAULT_PIPELINE
    _DEFAULT_PIPELINE = None
