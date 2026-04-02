"""
File Name: entity_graph_features.py
Module: Feature Engineering - Graph Features
Description:
    Builds an entity co-occurrence graph from text and extracts structural
    graph statistics describing relationships between entities mentioned in
    the document. The graph is constructed by connecting entities that appear
    within the same sentence or context window.

    The extracted features characterize entity interaction complexity,
    narrative centralization, and connectivity patterns which are useful
    for narrative analysis, bias detection, and discourse structure modeling.

    The module optionally uses spaCy for entity recognition. If spaCy is not
    available, a heuristic fallback based on capitalized tokens is used.

Dependencies:
    dataclasses
    typing
    logging
    re
    itertools
    networkx (optional)
    spacy (optional)

Inputs:
    FeatureContext containing input text

Outputs:
    Dict[str, float] representing entity graph statistics
"""

from __future__ import annotations

import logging
import re
import itertools
from dataclasses import dataclass
from typing import Dict, List

from src.features.base.base_feature import BaseFeature, FeatureContext
from src.features.base.feature_registry import register_feature

logger = logging.getLogger(__name__)

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except Exception:
    NETWORKX_AVAILABLE = False
    logger.warning("networkx not available. Graph metrics will be approximated.")

try:
    import spacy
    _NLP = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except Exception:
    _NLP = None
    SPACY_AVAILABLE = False
    logger.warning("spaCy not available. Using heuristic entity detection.")


def _sentence_split(text: str) -> List[str]:
    """Basic sentence splitter."""
    return [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]


def _heuristic_entities(sentence: str) -> List[str]:
    """
    Simple fallback entity detection using capitalized tokens.
    """
    tokens = re.findall(r"\b[A-Z][a-zA-Z]+\b", sentence)
    return list(set(tokens))


def _extract_entities(sentence: str) -> List[str]:
    """Extract named entities using spaCy if available."""
    if SPACY_AVAILABLE:
        doc = _NLP(sentence)
        return list({ent.text for ent in doc.ents})
    return _heuristic_entities(sentence)


@dataclass
@register_feature
class EntityGraphFeatures(BaseFeature):
    """
    Extracts entity graph statistics.

    Output Features
    ---------------
    entity_count
    entity_edge_count
    entity_avg_degree
    entity_density
    entity_centralization
    """

    name: str = "entity_graph_features"
    description: str = "Entity interaction graph statistics"

    def extract(self, context: FeatureContext) -> Dict[str, float]:

        if not context.text:
            raise ValueError("FeatureContext.text cannot be empty")

        sentences = _sentence_split(context.text)

        entity_pairs = []

        entities = set()

        for sent in sentences:
            sent_entities = _extract_entities(sent)

            entities.update(sent_entities)

            for pair in itertools.combinations(sent_entities, 2):
                entity_pairs.append(pair)

        entity_count = len(entities)
        edge_count = len(entity_pairs)

        if not NETWORKX_AVAILABLE or entity_count == 0:
            return {
                "entity_count": float(entity_count),
                "entity_edge_count": float(edge_count),
                "entity_avg_degree": 0.0,
                "entity_density": 0.0,
                "entity_centralization": 0.0,
            }

        G = nx.Graph()

        for e in entities:
            G.add_node(e)

        for u, v in entity_pairs:
            G.add_edge(u, v)

        degrees = [deg for _, deg in G.degree()]

        avg_degree = sum(degrees) / len(degrees) if degrees else 0.0

        density = nx.density(G)

        max_deg = max(degrees) if degrees else 0
        centralization = max_deg / max(len(entities) - 1, 1)

        features: Dict[str, float] = {
            "entity_count": float(entity_count),
            "entity_edge_count": float(edge_count),
            "entity_avg_degree": float(avg_degree),
            "entity_density": float(density),
            "entity_centralization": float(centralization),
        }

        logger.debug(
            "Entity graph features extracted | nodes=%d edges=%d",
            entity_count,
            edge_count,
        )

        return features