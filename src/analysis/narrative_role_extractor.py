"""
File Name: narrative_role_extractor.py
Module: Narrative Analysis - Narrative Role Extraction
Description:
    Extracts narrative roles (Hero, Villain, Victim) from text for the TruthLens
    AI system. The module analyzes named entities and linguistic context to
    identify actors positioned as helpers/protectors (heroes), aggressors or
    blamed actors (villains), and harmed or suffering groups (victims).

    These roles are commonly used in narrative analysis, propaganda detection,
    and political discourse modeling. The extracted entities help downstream
    modules understand narrative framing and actor dynamics within text.

Dependencies:
    logging
    typing
    dataclasses
    collections
    numpy
    spacy

Inputs:
    Raw text string

Outputs:
    Dictionary containing hero_entities, villain_entities, victim_entities
    and optional numerical vector representation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import spacy
from spacy.language import Language
from spacy.tokens import Doc, Token


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class NarrativeRoleConfig:
    """
    Configuration for NarrativeRoleExtractor.
    """

    spacy_model: str = "en_core_web_sm"


class NarrativeRoleExtractor:
    """
    Extracts Hero / Villain / Victim narrative roles from text.
    """

    HERO_TERMS = {
        "protect",
        "defend",
        "save",
        "support",
        "rescue",
        "help",
        "aid",
        "lead",
        "champion",
    }

    VILLAIN_TERMS = {
        "attack",
        "harm",
        "destroy",
        "blame",
        "threaten",
        "corrupt",
        "abuse",
        "exploit",
    }

    VICTIM_TERMS = {
        "suffer",
        "victim",
        "hurt",
        "killed",
        "injured",
        "affected",
        "targeted",
        "displaced",
    }

    def __init__(self, config: NarrativeRoleConfig | None = None) -> None:
        """
        Initialize NLP pipeline for narrative role extraction.
        """

        self.config = config or NarrativeRoleConfig()

        try:
            self.nlp: Language = spacy.load(self.config.spacy_model)
        except Exception as exc:
            logger.exception("spaCy model loading failed")
            raise RuntimeError(
                f"Failed to load spaCy model: {self.config.spacy_model}"
            ) from exc

        logger.info(
            "NarrativeRoleExtractor initialized with model=%s",
            self.config.spacy_model,
        )

    def analyze(self, text: str) -> Dict[str, List[str]]:
        """
        Extract narrative roles from text.

        Args:
            text: Input text.

        Returns:
            Dictionary containing hero_entities, villain_entities, victim_entities.
        """

        if not isinstance(text, str):
            raise ValueError("Input text must be a string")

        cleaned_text = text.strip()

        if not cleaned_text:
            raise ValueError("Input text must be non-empty")

        try:
            doc: Doc = self.nlp(cleaned_text)
        except Exception as exc:
            logger.exception("spaCy processing failed")
            raise RuntimeError("Text processing failed") from exc

        hero_entities: set[str] = set()
        villain_entities: set[str] = set()
        victim_entities: set[str] = set()

        for token in doc:

            token_lower = token.lemma_.lower()

            if token_lower in self.HERO_TERMS:
                entity = self._resolve_entity(token)
                if entity:
                    hero_entities.add(entity)

            elif token_lower in self.VILLAIN_TERMS:
                entity = self._resolve_entity(token)
                if entity:
                    villain_entities.add(entity)

            elif token_lower in self.VICTIM_TERMS:
                entity = self._resolve_entity(token)
                if entity:
                    victim_entities.add(entity)

        features: Dict[str, List[str]] = {
            "hero_entities": sorted(hero_entities),
            "villain_entities": sorted(villain_entities),
            "victim_entities": sorted(victim_entities),
        }

        logger.debug("Narrative roles extracted")

        return features

    def _resolve_entity(self, token: Token) -> str | None:
        """
        Resolve entity associated with a narrative action.
        """

        head = token.head

        if head.ent_type_:
            return head.text.lower()

        for child in token.children:
            if child.ent_type_:
                return child.text.lower()

        if head.pos_ in {"NOUN", "PROPN"}:
            return head.lemma_.lower()

        return None


def narrative_role_vector(features: Dict[str, List[str]]) -> np.ndarray:
    """
    Convert narrative roles into numeric vector representation.

    Vector structure:
        [num_heroes, num_villains, num_victims]
    """

    if not isinstance(features, dict):
        raise ValueError("features must be a dictionary")

    heroes = features.get("hero_entities", [])
    villains = features.get("villain_entities", [])
    victims = features.get("victim_entities", [])

    try:
        vector = np.array(
            [
                float(len(heroes)),
                float(len(villains)),
                float(len(victims)),
            ],
            dtype=np.float32,
        )
        return vector

    except Exception as exc:
        logger.exception("Narrative role vector conversion failed")
        raise RuntimeError("Failed to convert narrative role features") from exc