from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)


# =========================================================
# YAML LOADER
# =========================================================

def load_yaml_as_dict(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML dict at {path}")

    return data


# =========================================================
# PARSER
# =========================================================

def parse_graph_config(config_data: Dict[str, Any]) -> Dict[str, Any]:

    graph = config_data.get("graph", config_data)

    return {
        # core toggles
        "enable_entity_graph": bool(graph.get("enable_entity_graph", True)),
        "enable_narrative_graph": bool(graph.get("enable_narrative_graph", True)),

        #  NEW
        "enable_temporal_graph": bool(graph.get("enable_temporal_graph", True)),
        "enable_graph_explainer": bool(graph.get("enable_graph_explainer", True)),

        # extraction
        "min_keyword_length": int(graph.get("min_keyword_length", 4)),
        "max_keywords_per_sentence": int(graph.get("max_keywords_per_sentence", 4)),

        # graph behavior
        "use_weighted_edges": bool(graph.get("use_weighted_edges", True)),
        "normalize_graph": bool(graph.get("normalize_graph", True)),

        # thresholds
        "min_edge_weight": float(graph.get("min_edge_weight", 0.0)),
        "max_edge_weight": float(graph.get("max_edge_weight", 10.0)),

        # scaling
        "feature_scale": float(graph.get("feature_scale", 1.0)),

        # advanced
        "enable_graph_embeddings": bool(graph.get("enable_graph_embeddings", False)),
    }


# =========================================================
# DATACLASS
# =========================================================

@dataclass(slots=True)
class GraphConfig:

    # toggles
    enable_entity_graph: bool = True
    enable_narrative_graph: bool = True

    # 🔥 NEW
    enable_temporal_graph: bool = True
    enable_graph_explainer: bool = True

    # extraction
    min_keyword_length: int = 4
    max_keywords_per_sentence: int = 4

    # graph behavior
    use_weighted_edges: bool = True
    normalize_graph: bool = True

    # thresholds
    min_edge_weight: float = 0.0
    max_edge_weight: float = 10.0

    # scaling
    feature_scale: float = 1.0

    # advanced
    enable_graph_embeddings: bool = False


# =========================================================
# LOADER
# =========================================================

class GraphConfigLoader:

    def __init__(self):
        logger.info("GraphConfigLoader initialized")

    def load_from_yaml(self, path: str | Path) -> GraphConfig:

        p = Path(path)

        if not p.exists():
            raise FileNotFoundError(f"Config not found: {p}")

        data = load_yaml_as_dict(p)
        return self._parse(data)

    def load_from_dict(self, config: Dict[str, Any]) -> GraphConfig:

        if not isinstance(config, dict):
            raise TypeError("config must be dict")

        return self._parse(config)

    # =====================================================
    # INTERNAL
    # =====================================================

    def _parse(self, config_data: Dict[str, Any]) -> GraphConfig:

        parsed = parse_graph_config(config_data)

        cfg = GraphConfig(**parsed)

        self._validate(cfg)

        logger.info("GraphConfig loaded")

        return cfg

    # =====================================================
    # VALIDATION
    # =====================================================

    def _validate(self, cfg: GraphConfig) -> None:

        if cfg.min_keyword_length < 1:
            raise ValueError("min_keyword_length must be >= 1")

        if cfg.max_keywords_per_sentence < 1:
            raise ValueError("max_keywords_per_sentence must be >= 1")

        if cfg.min_edge_weight < 0:
            raise ValueError("min_edge_weight must be >= 0")

        if cfg.max_edge_weight <= cfg.min_edge_weight:
            raise ValueError("max_edge_weight must be > min_edge_weight")

        if not (0.0 < cfg.feature_scale <= 10.0):
            raise ValueError("feature_scale must be in (0, 10]")

        if not isinstance(cfg.enable_entity_graph, bool):
            raise TypeError("enable_entity_graph must be bool")

        if not isinstance(cfg.enable_narrative_graph, bool):
            raise TypeError("enable_narrative_graph must be bool")

        #  NEW VALIDATION
        if not isinstance(cfg.enable_temporal_graph, bool):
            raise TypeError("enable_temporal_graph must be bool")

        if not isinstance(cfg.enable_graph_explainer, bool):
            raise TypeError("enable_graph_explainer must be bool")


# =========================================================
# UTILITIES
# =========================================================

def clip_edge_weight(value: float, cfg: GraphConfig) -> float:
    return float(max(cfg.min_edge_weight, min(value, cfg.max_edge_weight)))


def scale_feature(value: float, cfg: GraphConfig) -> float:
    return float(value * cfg.feature_scale)