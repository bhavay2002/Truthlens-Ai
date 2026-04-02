"""
File Name: graph_utils.py
Module: Graph Analysis - Graph Utilities
Description:
    Provides reusable low-level graph operations used across the TruthLens
    graph subsystem. These utilities normalize adjacency structures,
    convert graphs to undirected representations, compute degree
    distributions, remove invalid/self-loop edges, and extract unique
    edge pairs. Centralizing these utilities prevents duplicated logic
    across graph builders, analyzers, and pipelines.

Dependencies:
    logging
    typing
    collections

Inputs:
    Graph adjacency dictionary

Outputs:
    Normalized graphs, statistics, and edge sets
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Dict, Iterable, List, Set, Tuple


logger = logging.getLogger(__name__)


Graph = Dict[str, List[str]]
AdjacencySet = Dict[str, Set[str]]
EdgePair = Tuple[str, str]


def normalize_adjacency(graph: Graph) -> AdjacencySet:
    """
    Normalize adjacency list into a clean dictionary of sets.

    - lowercases nodes
    - strips whitespace
    - removes invalid neighbors
    """

    if not isinstance(graph, dict):
        raise ValueError("graph must be a dictionary")

    adjacency: AdjacencySet = {}

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

    logger.debug("Adjacency normalized with %d nodes", len(adjacency))

    return adjacency


def to_undirected_graph(adjacency: AdjacencySet) -> AdjacencySet:
    """
    Convert directed adjacency representation to undirected graph.
    """

    if not isinstance(adjacency, dict):
        raise ValueError("adjacency must be a dictionary")

    undirected: AdjacencySet = {
        node: set(neighbors) for node, neighbors in adjacency.items()
    }

    for node, neighbors in list(undirected.items()):

        for neighbor in neighbors:

            undirected.setdefault(neighbor, set()).add(node)

    logger.debug("Converted graph to undirected representation")

    return undirected


def degree_distribution(adjacency: AdjacencySet) -> Dict[int, int]:
    """
    Compute degree distribution of graph.

    Returns
    -------
    Dict[int, int]
        degree -> frequency
    """

    if not isinstance(adjacency, dict):
        raise ValueError("adjacency must be a dictionary")

    degrees = [len(neighbors) for neighbors in adjacency.values()]

    distribution = dict(Counter(degrees))

    logger.debug("Computed degree distribution")

    return distribution


def remove_self_loops(adjacency: AdjacencySet) -> AdjacencySet:
    """
    Remove self-loop edges from graph.
    """

    if not isinstance(adjacency, dict):
        raise ValueError("adjacency must be a dictionary")

    cleaned: AdjacencySet = {}

    for node, neighbors in adjacency.items():

        cleaned[node] = {
            neighbor
            for neighbor in neighbors
            if neighbor != node
        }

    logger.debug("Self-loops removed")

    return cleaned


def unique_edge_pairs(adjacency: AdjacencySet) -> Set[EdgePair]:
    """
    Extract unique edge pairs from graph.

    Returns
    -------
    Set[Tuple[str, str]]
        Unique directed edges
    """

    if not isinstance(adjacency, dict):
        raise ValueError("adjacency must be a dictionary")

    edges: Set[EdgePair] = set()

    for source, neighbors in adjacency.items():

        for target in neighbors:

            if source != target:

                edges.add((source, target))

    logger.debug("Extracted %d unique edges", len(edges))

    return edges


def node_set(adjacency: AdjacencySet) -> Set[str]:
    """
    Return set of all nodes present in graph.
    """

    nodes: Set[str] = set(adjacency.keys())

    for neighbors in adjacency.values():
        nodes.update(neighbors)

    return nodes


def edge_count(adjacency: AdjacencySet) -> int:
    """
    Count edges in adjacency graph.
    """

    return sum(len(neighbors) for neighbors in adjacency.values())


def graph_summary(adjacency: AdjacencySet) -> Dict[str, int]:
    """
    Produce basic graph summary statistics.

    Returns
    -------
    Dict[str, int]
    """

    nodes = node_set(adjacency)
    edges = edge_count(adjacency)

    summary = {
        "nodes": len(nodes),
        "edges": edges,
    }

    return summary