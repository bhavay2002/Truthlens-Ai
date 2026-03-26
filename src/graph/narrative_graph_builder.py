"""
File Name: narrative_graph_builder.py
Module: Graph Analysis - Narrative Graph Construction
Description:
    Builds lightweight narrative transition graphs from article text.
    The graph captures how salient keywords move from one sentence to
    the next, enabling discourse-flow analysis features.
"""

from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict, deque
from typing import Dict, Iterable, List

import numpy as np


logger = logging.getLogger(__name__)


def _split_sentences(text: str) -> List[str]:
    sentences = re.split(r"[.!?]+", text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def _extract_keywords(sentence: str, min_token_length: int) -> List[str]:
    tokens = re.findall(r"\b[a-zA-Z]+\b", sentence.lower())
    filtered = [
        token
        for token in tokens
        if len(token) >= min_token_length
    ]
    if not filtered:
        return []

    token_counts = Counter(filtered)
    ranked = sorted(
        token_counts.items(),
        key=lambda item: (-item[1], item[0]),
    )
    return [token for token, _ in ranked]


class NarrativeGraphBuilder:
    """Build sentence-transition narrative graphs and summary features."""

    def __init__(
        self,
        min_token_length: int = 4,
        max_keywords_per_sentence: int = 4,
    ) -> None:
        if min_token_length < 1:
            raise ValueError("min_token_length must be >= 1")
        if max_keywords_per_sentence < 1:
            raise ValueError("max_keywords_per_sentence must be >= 1")

        self.min_token_length = min_token_length
        self.max_keywords_per_sentence = max_keywords_per_sentence

    def build_graph(self, text: str) -> Dict[str, List[str]]:
        """Build directed narrative graph from sentence keyword transitions."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        sentences = _split_sentences(text)
        graph: Dict[str, List[str]] = defaultdict(list)
        previous_keywords: list[str] = []

        for sentence in sentences:
            ranked_keywords = _extract_keywords(
                sentence,
                min_token_length=self.min_token_length,
            )
            keywords = ranked_keywords[: self.max_keywords_per_sentence]
            if not keywords:
                continue

            for keyword in keywords:
                graph.setdefault(keyword, [])

            if previous_keywords:
                for source in previous_keywords:
                    for target in keywords:
                        if source != target:
                            graph[source].append(target)

            previous_keywords = keywords

        return dict(graph)

    def extract_graph_features(
        self,
        graph: Dict[str, List[str]],
    ) -> Dict[str, float]:
        """Extract structural features from a narrative graph."""
        if not isinstance(graph, dict):
            raise ValueError("graph must be a dictionary")

        adjacency: dict[str, set[str]] = {}
        all_nodes: set[str] = set()

        for node, neighbors in graph.items():
            if not isinstance(node, str):
                raise ValueError("graph keys must be strings")
            if not isinstance(neighbors, list):
                raise ValueError("graph values must be lists")

            node_key = node.strip().lower()
            neighbor_set = {
                str(neighbor).strip().lower()
                for neighbor in neighbors
                if isinstance(neighbor, str)
                and neighbor.strip()
                and str(neighbor).strip().lower() != node_key
            }
            adjacency[node_key] = neighbor_set
            all_nodes.add(node_key)
            all_nodes.update(neighbor_set)

        for node in all_nodes:
            adjacency.setdefault(node, set())

        edge_pairs = {
            (source, target)
            for source, neighbors in adjacency.items()
            for target in neighbors
            if source != target
        }

        node_count = len(all_nodes)
        edge_count = len(edge_pairs)
        avg_degree = edge_count / max(node_count, 1)
        density = edge_count / max(node_count * (node_count - 1), 1)
        isolated_nodes = sum(
            1 for node, neighbors in adjacency.items() if len(neighbors) == 0
        )
        component_count = self._weak_component_count(adjacency)

        return {
            "narrative_graph_nodes": float(node_count),
            "narrative_graph_edges": float(edge_count),
            "narrative_graph_avg_degree": float(avg_degree),
            "narrative_graph_density": float(density),
            "narrative_graph_isolated_nodes": float(isolated_nodes),
            "narrative_graph_components": float(component_count),
        }

    def _weak_component_count(self, adjacency: Dict[str, set[str]]) -> int:
        if not adjacency:
            return 0

        undirected = {node: set(neighbors) for node, neighbors in adjacency.items()}
        for node, neighbors in list(undirected.items()):
            for neighbor in neighbors:
                undirected.setdefault(neighbor, set()).add(node)

        visited: set[str] = set()
        components = 0

        for start in undirected:
            if start in visited:
                continue

            components += 1
            queue: deque[str] = deque([start])
            visited.add(start)

            while queue:
                node = queue.popleft()
                for neighbor in undirected[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

        return components


def narrative_graph_vector(features: Dict[str, float]) -> np.ndarray:
    """Convert narrative graph feature dictionary to fixed-order vector."""
    if not isinstance(features, dict) or not features:
        raise ValueError("features must be a non-empty dictionary")

    ordered_keys: Iterable[str] = (
        "narrative_graph_nodes",
        "narrative_graph_edges",
        "narrative_graph_avg_degree",
        "narrative_graph_density",
        "narrative_graph_isolated_nodes",
        "narrative_graph_components",
    )

    return np.array(
        [float(features.get(key, 0.0)) for key in ordered_keys],
        dtype=np.float32,
    )
