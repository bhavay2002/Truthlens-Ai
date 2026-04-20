"""
File Name: graph_config.py
Module: Graph Analysis - Graph Configuration Management
Description:
    Defines configuration structures and utilities for the graph subsystem
    in the TruthLens AI system. The module integrates with the central YAML
    configuration system and converts graph-related configuration blocks
    into strongly-typed dataclasses used by graph builders and pipelines.

Dependencies:
    logging
    typing
    dataclasses
    pathlib
    yaml

Inputs:
    YAML configuration file or configuration dictionary

Outputs:
    Graph configuration dataclass instances
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from graph_hardening_patch import load_yaml_as_dict, parse_graph_config


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class GraphConfig:
    """
    Configuration for graph subsystem.

    Attributes
    ----------
    enable_entity_graph : bool
        Enable entity interaction graph.
    enable_narrative_graph : bool
        Enable narrative transition graph.
    min_keyword_length : int
        Minimum token length used in keyword extraction.
    max_keywords_per_sentence : int
        Maximum keywords extracted per sentence.
    """

    enable_entity_graph: bool = True
    enable_narrative_graph: bool = True
    min_keyword_length: int = 4
    max_keywords_per_sentence: int = 4


class GraphConfigLoader:
    """
    Loader for graph configuration from YAML or dictionary sources.
    """

    def __init__(self) -> None:
        logger.info("GraphConfigLoader initialized")

    def load_from_yaml(self, config_path: str | Path) -> GraphConfig:
        """
        Load graph configuration from YAML file.
        """

        path = Path(config_path)

        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        try:
            config_data = load_yaml_as_dict(path)
        except Exception as exc:
            logger.exception("Failed to load YAML configuration")
            raise RuntimeError("YAML configuration loading failed") from exc

        return self._parse_config(config_data)

    def load_from_dict(self, config_dict: Dict[str, Any]) -> GraphConfig:
        """
        Load graph configuration from dictionary.
        """

        if not isinstance(config_dict, dict):
            raise ValueError("config_dict must be a dictionary")

        return self._parse_config(config_dict)

    def _parse_config(self, config_data: Dict[str, Any]) -> GraphConfig:
        """
        Parse configuration dictionary into GraphConfig.
        """

        hardened = parse_graph_config(config_data)

        config = GraphConfig(
            enable_entity_graph=hardened.enable_entity_graph,
            enable_narrative_graph=hardened.enable_narrative_graph,
            min_keyword_length=hardened.min_keyword_length,
            max_keywords_per_sentence=hardened.max_keywords_per_sentence,
        )

        self._validate_config(config)

        logger.info("Graph configuration loaded successfully")

        return config

    def _validate_config(self, config: GraphConfig) -> None:
        """
        Validate graph configuration values.
        """

        if config.min_keyword_length < 1:
            raise ValueError("min_keyword_length must be >= 1")

        if config.max_keywords_per_sentence < 1:
            raise ValueError("max_keywords_per_sentence must be >= 1")