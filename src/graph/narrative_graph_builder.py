from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Set

import numpy as np

logger = logging.getLogger(__name__)
EPS = 1e-12


# =========================================================
# FEATURES
# =========================================================

@dataclass(slots=True)
class NarrativeGraphFeatures:

    narrative_graph_nodes: float
    narrative_graph_edges: float
    narrative_graph_avg_degree: float
    narrative_graph_density: float
    narrative_graph_isolated_nodes: float
    narrative_graph_components: float

    # 🔥 NEW
    narrative_graph_entropy: float
    narrative_graph_centralization: float
    narrative_graph_flow_strength: float

    def to_dict(self) -> Dict[str, float]:
        # ``slots=True`` strips ``__dict__``; build via ``__slots__``.
        return {f: getattr(self, f) for f in self.__slots__}


# =========================================================
# HELPERS
# =========================================================

def _split_sentences(text: str) -> List[str]:
    sentences = re.split(r"[.!?]+", text)
    return [s.strip() for s in sentences if s.strip()]


def _extract_keywords(sentence: str, min_len: int) -> List[str]:

    tokens = re.findall(r"\b[a-zA-Z]+\b", sentence.lower())

    filtered = [t for t in tokens if len(t) >= min_len]

    if not filtered:
        return []

    counts = Counter(filtered)

    ranked = sorted(
        counts.items(),
        key=lambda x: (-x[1], x[0]),
    )

    return [t for t, _ in ranked]


# =========================================================
# BUILDER
# =========================================================

class NarrativeGraphBuilder:

    def __init__(
        self,
        min_token_length: int = 4,
        max_keywords_per_sentence: int = 4,
    ):

        if min_token_length < 1:
            raise ValueError("min_token_length must be >= 1")

        if max_keywords_per_sentence < 1:
            raise ValueError("max_keywords_per_sentence must be >= 1")

        self.min_token_length = min_token_length
        self.max_keywords_per_sentence = max_keywords_per_sentence

        logger.info("NarrativeGraphBuilder initialized")

    # =====================================================
    # 🔥 BUILD GRAPH (WEIGHTED)
    # =====================================================

    def build_graph(self, text: str) -> Dict[str, Dict[str, float]]:

        if not isinstance(text, str) or not text.strip():
            raise ValueError("Invalid text")

        sentences = _split_sentences(text)

        graph: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

        prev_keywords: List[str] = []

        for idx, sentence in enumerate(sentences):

            ranked = _extract_keywords(sentence, self.min_token_length)
            keywords = ranked[: self.max_keywords_per_sentence]

            if not keywords:
                continue

            # ensure nodes — use ``defaultdict(float)`` so the
            # ``graph[src][tgt] += 1.0`` accumulator below doesn't
            # KeyError on the first transition (the previous
            # ``setdefault(k, {})`` overrode the inner factory with a
            # plain dict, breaking the augmented-assignment).
            for k in keywords:
                if k not in graph:
                    graph[k] = defaultdict(float)

            # 🔥 weighted transitions
            if prev_keywords:
                for src in prev_keywords:
                    for tgt in keywords:
                        if src != tgt:
                            graph[src][tgt] += 1.0

            prev_keywords = keywords

        return {k: dict(v) for k, v in graph.items()}

    # =====================================================
    # FEATURES
    # =====================================================

    def extract_graph_features(
        self,
        graph: Dict[str, Dict[str, float]],
    ) -> NarrativeGraphFeatures:

        if not isinstance(graph, dict):
            raise ValueError("graph must be dict")

        nodes = set(graph.keys())

        edges = set()
        degrees = []

        weights = []

        for src, nbrs in graph.items():

            for tgt, w in nbrs.items():
                if src != tgt:
                    edges.add((src, tgt))
                    weights.append(w)

            degrees.append(len(nbrs))

            nodes.update(nbrs.keys())

        n = len(nodes)
        e = len(edges)

        degrees_arr = np.array(degrees, dtype=float) if degrees else np.array([])

        avg_degree = float(np.mean(degrees_arr)) if degrees_arr.size else 0.0
        density = float(e / (n * (n - 1) + EPS)) if n > 1 else 0.0

        isolated = sum(1 for d in degrees if d == 0)

        components = self._weak_components(graph)

        # =================================================
        # 🔥 NEW METRICS
        # =================================================

        # entropy (distribution of edges)
        if weights:
            w = np.array(weights, dtype=float)
            p = w / (np.sum(w) + EPS)
            entropy = float(-np.sum(p * np.log(p + EPS)))
        else:
            entropy = 0.0

        # centralization
        if degrees_arr.size:
            centralization = float(
                (np.max(degrees_arr) - np.mean(degrees_arr)) / (n - 1 + EPS)
            )
        else:
            centralization = 0.0

        # flow strength (temporal continuity)
        flow_strength = float(np.mean(weights)) if weights else 0.0

        return NarrativeGraphFeatures(
            narrative_graph_nodes=float(n),
            narrative_graph_edges=float(e),
            narrative_graph_avg_degree=avg_degree,
            narrative_graph_density=density,
            narrative_graph_isolated_nodes=float(isolated),
            narrative_graph_components=float(components),
            narrative_graph_entropy=entropy,
            narrative_graph_centralization=centralization,
            narrative_graph_flow_strength=flow_strength,
        )

    # =====================================================
    # COMPONENTS
    # =====================================================

    def _weak_components(self, graph: Dict[str, Dict[str, float]]) -> int:

        undirected: Dict[str, Set[str]] = defaultdict(set)

        for u, nbrs in graph.items():
            for v in nbrs:
                undirected[u].add(v)
                undirected[v].add(u)

        visited: Set[str] = set()
        count = 0

        for start in undirected:

            if start in visited:
                continue

            count += 1

            q = deque([start])
            visited.add(start)

            while q:
                node = q.popleft()
                for nbr in undirected[node]:
                    if nbr not in visited:
                        visited.add(nbr)
                        q.append(nbr)

        return count


# =========================================================
# VECTOR
# =========================================================

def narrative_graph_vector(features: Dict[str, float]) -> np.ndarray:

    keys: Iterable[str] = (
        "narrative_graph_nodes",
        "narrative_graph_edges",
        "narrative_graph_avg_degree",
        "narrative_graph_density",
        "narrative_graph_isolated_nodes",
        "narrative_graph_components",
        "narrative_graph_entropy",
        "narrative_graph_centralization",
        "narrative_graph_flow_strength",
    )

    return np.array(
        [float(features.get(k, 0.0)) for k in keys],
        dtype=np.float32,
    )