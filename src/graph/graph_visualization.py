"""
File Name: graph_visualization.py
Module: Graph Analysis - Graph Visualization
Description:
    Provides visualization utilities for graph structures used in the
    TruthLens AI system. This module supports rendering entity graphs
    and narrative graphs for debugging, research analysis, and
    explainability dashboards. Graphs are converted into NetworkX
    objects and exported as PNG images using Matplotlib.

Dependencies:
    logging
    typing
    pathlib
    networkx
    matplotlib

Inputs:
    Graph adjacency dictionary

Outputs:
    PNG visualization files (entity_graph.png, narrative_graph.png)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

from graph_hardening_patch import ensure_headless_matplotlib_backend

ensure_headless_matplotlib_backend()

import matplotlib.pyplot as plt
import networkx as nx


logger = logging.getLogger(__name__)


class GraphVisualizer:
    """
    Utility class for visualizing graphs used in the TruthLens system.
    """

    def __init__(self, output_dir: str | Path = "reports/graphs") -> None:
        """
        Initialize visualization output directory.

        Parameters
        ----------
        output_dir : str | Path
            Directory where graph images will be saved.
        """

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("GraphVisualizer initialized (output_dir=%s)", self.output_dir)

    def _validate_graph(self, graph: Dict[str, List[str]]) -> None:
        """Validate adjacency graph structure."""

        if not isinstance(graph, dict):
            raise ValueError("graph must be a dictionary")

        for node, neighbors in graph.items():
            if not isinstance(node, str):
                raise ValueError("graph keys must be strings")
            if not isinstance(neighbors, list):
                raise ValueError("graph values must be lists")

    def _build_networkx_graph(self, graph: Dict[str, List[str]]) -> nx.DiGraph:
        """
        Convert adjacency dictionary into NetworkX graph.
        """

        G = nx.DiGraph()

        for node, neighbors in graph.items():
            node_key = node.strip().lower()
            G.add_node(node_key)

            for neighbor in neighbors:
                if isinstance(neighbor, str) and neighbor.strip():
                    neighbor_key = neighbor.strip().lower()
                    if neighbor_key != node_key:
                        G.add_edge(node_key, neighbor_key)

        return G

    def visualize_entity_graph(
        self,
        graph: Dict[str, List[str]],
        filename: str = "entity_graph.png",
    ) -> Path:
        """
        Visualize entity interaction graph.

        Parameters
        ----------
        graph : Dict[str, List[str]]
        filename : str

        Returns
        -------
        Path
            Path to saved image.
        """

        self._validate_graph(graph)

        G = self._build_networkx_graph(graph)

        output_path = self.output_dir / filename

        plt.figure(figsize=(10, 8))

        pos = nx.spring_layout(G)

        nx.draw_networkx_nodes(G, pos)
        nx.draw_networkx_edges(G, pos)
        nx.draw_networkx_labels(G, pos, font_size=8)

        plt.title("Entity Interaction Graph")
        plt.axis("off")

        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()

        logger.info("Entity graph saved: %s", output_path)

        return output_path

    def visualize_narrative_graph(
        self,
        graph: Dict[str, List[str]],
        filename: str = "narrative_graph.png",
    ) -> Path:
        """
        Visualize narrative transition graph.

        Parameters
        ----------
        graph : Dict[str, List[str]]
        filename : str

        Returns
        -------
        Path
            Path to saved image.
        """

        self._validate_graph(graph)

        G = self._build_networkx_graph(graph)

        output_path = self.output_dir / filename

        plt.figure(figsize=(10, 8))

        pos = nx.spring_layout(G)

        nx.draw_networkx_nodes(G, pos)
        nx.draw_networkx_edges(G, pos)
        nx.draw_networkx_labels(G, pos, font_size=8)

        plt.title("Narrative Transition Graph")
        plt.axis("off")

        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()

        logger.info("Narrative graph saved: %s", output_path)

        return output_path