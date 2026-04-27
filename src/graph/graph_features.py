from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from src.graph.entity_graph import EntityGraphBuilder, ordered_entity_graph_vector
from src.graph.graph_analysis import GraphAnalyzer, ordered_graph_metrics_vector
from src.graph.narrative_graph_builder import NarrativeGraphBuilder
from src.graph.graph_embeddings import graph_embedding_vector, GraphEmbeddingConfig

logger = logging.getLogger(__name__)
EPS = 1e-12


# =========================================================
# DEVICE TRANSFER HELPER  (G-G1)
# =========================================================
#
# The graph layer is intentionally CPU-only (good — spaCy parsing is
# the bottleneck, not the linear algebra) but the resulting feature
# vectors used to be returned as plain ``np.float32`` arrays. The
# downstream consumer (``HybridTruthLensModel.forward``) then did one
# host→device copy *per sample* inside the forward pass, which on
# batched inference is N independent CUDA syncs.
#
# ``to_pinned_tensor`` lets the batch-inference / feature-preparer
# layer:
#   1. accumulate per-sample numpy vectors on the CPU,
#   2. ``np.stack`` them once,
#   3. wrap the stacked array in a pinned-memory tensor here,
#   4. do a single ``.to(device, non_blocking=True)`` at the model
#      boundary.
#
# This collapses N syncs into 1 and overlaps the copy with kernel
# launches when the model is GPU-resident. Pinning is a no-op on
# CPU-only deployments, so this is safe to call unconditionally.
# ``torch`` import is deferred so ``src.graph`` keeps loading on
# torch-less environments (e.g. data-only utility scripts).


def to_pinned_tensor(vec):
    """Wrap a numpy array (or list of arrays) in a pinned CPU tensor.

    Accepts a single ``np.ndarray`` or a list/tuple of equally shaped
    arrays — in the second case the arrays are stacked along axis 0
    so the caller gets a single ``(N, D)`` tensor ready for one
    ``.to(device, non_blocking=True)`` copy at the model edge.

    Returns the tensor unchanged on platforms without CUDA support
    (``pin_memory`` raises on those — we swallow it and return a
    plain CPU tensor instead).
    """
    import torch  # local import keeps ``src.graph`` torch-optional
    import numpy as _np

    if isinstance(vec, (list, tuple)):
        if not vec:
            return torch.empty(0, dtype=torch.float32)
        arr = _np.stack([_np.asarray(v, dtype=_np.float32) for v in vec], axis=0)
    else:
        arr = _np.asarray(vec, dtype=_np.float32)

    t = torch.from_numpy(arr)

    # ``pin_memory`` requires CUDA *or* a CPU-pinned allocator; on
    # pure-CPU builds it raises ``RuntimeError``. Treat that as a
    # no-op so callers can use this helper unconditionally.
    if torch.cuda.is_available():
        try:
            t = t.pin_memory()
        except RuntimeError:
            pass

    return t


# =========================================================
# CONFIG
# =========================================================

@dataclass(slots=True)
class GraphFeatureExtractorConfig:
    enable_entity_graph: bool = True
    enable_narrative_graph: bool = True
    enable_embeddings: bool = True
    embedding_config: Optional[GraphEmbeddingConfig] = None
    normalize_features: bool = True


# =========================================================
# UTIL
# =========================================================

def _normalize_vector(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec) + EPS
    return vec / norm


def merge_feature_blocks_strict(*blocks: Dict[str, float]) -> Dict[str, float]:
    merged: Dict[str, float] = {}

    for block in blocks:
        for k, v in block.items():
            if k in merged:
                raise ValueError(f"Duplicate feature key: {k}")
            merged[k] = float(v)

    return merged


# =========================================================
# MAIN
# =========================================================

class GraphFeatureExtractor:

    def __init__(self, config: Optional[GraphFeatureExtractorConfig] = None):

        self.config = config or GraphFeatureExtractorConfig()

        self.entity_builder = (
            EntityGraphBuilder() if self.config.enable_entity_graph else None
        )

        self.narrative_builder = (
            NarrativeGraphBuilder() if self.config.enable_narrative_graph else None
        )

        self.analyzer = GraphAnalyzer()

        logger.info("GraphFeatureExtractor initialized")

    # =====================================================
    # FULL PIPELINE
    # =====================================================

    def extract_features(self, text: str) -> Dict[str, float]:

        if not isinstance(text, str) or not text.strip():
            raise ValueError("Invalid text")

        entity_graph = None
        narrative_graph = None

        if self.entity_builder:
            entity_graph = self.entity_builder.build_graph(text)

        if self.narrative_builder:
            narrative_graph = self.narrative_builder.build_graph(text)

        return self.extract_from_graphs(entity_graph, narrative_graph)

    # =====================================================
    # CORE LOGIC
    # =====================================================

    def extract_from_graphs(
        self,
        entity_graph: Optional[Dict[str, List[str]]] = None,
        narrative_graph: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, float]:

        blocks: List[Dict[str, float]] = []

        # -------------------------
        # ENTITY GRAPH
        # -------------------------
        if entity_graph and self.entity_builder:

            entity_features = (
                self.entity_builder.extract_graph_features(entity_graph).to_dict()
            )

            metrics = self.analyzer.analyze(entity_graph).to_dict()

            blocks.append(entity_features)
            blocks.append(metrics)

            # 🔥 embeddings
            if self.config.enable_embeddings:
                emb = graph_embedding_vector(
                    entity_graph,
                    self.config.embedding_config,
                )
                for i, val in enumerate(emb):
                    blocks.append({f"graph_embedding_{i}": float(val)})

        # -------------------------
        # NARRATIVE GRAPH
        # -------------------------
        if narrative_graph and self.narrative_builder:

            narrative_features = (
                self.narrative_builder.extract_graph_features(narrative_graph).to_dict()
            )

            blocks.append(narrative_features)

        if not blocks:
            return {}

        return merge_feature_blocks_strict(*blocks)

    # =====================================================
    # VECTOR
    # =====================================================

    def extract_feature_vector(self, text: str) -> np.ndarray:

        features = self.extract_features(text)
        return self.extract_feature_vector_from_features(features)

    def extract_feature_vector_from_features(
        self,
        features: Dict[str, float],
    ) -> np.ndarray:

        if not features:
            return np.zeros(0, dtype=np.float32)

        vectors: List[np.ndarray] = []

        # -------------------------
        # ENTITY + METRICS
        # -------------------------
        try:
            vectors.append(ordered_entity_graph_vector(features))
            vectors.append(ordered_graph_metrics_vector(features))
        except Exception:
            logger.warning("Skipping entity/metrics vector")

        # -------------------------
        # EMBEDDINGS
        # -------------------------
        emb_keys = sorted(
            [k for k in features if k.startswith("graph_embedding_")]
        )

        if emb_keys:
            emb_vec = np.array(
                [features[k] for k in emb_keys],
                dtype=np.float32,
            )
            vectors.append(emb_vec)

        # -------------------------
        # NARRATIVE
        # -------------------------
        narrative_keys = [
            "narrative_graph_nodes",
            "narrative_graph_edges",
            "narrative_graph_avg_degree",
            "narrative_graph_density",
            "narrative_graph_isolated_nodes",
            "narrative_graph_components",
        ]

        if all(k in features for k in narrative_keys):
            vectors.append(
                np.array([features[k] for k in narrative_keys], dtype=np.float32)
            )

        if not vectors:
            return np.zeros(0, dtype=np.float32)

        vec = np.concatenate(vectors).astype(np.float32)

        # 🔥 normalization
        if self.config.normalize_features:
            vec = _normalize_vector(vec)

        return vec