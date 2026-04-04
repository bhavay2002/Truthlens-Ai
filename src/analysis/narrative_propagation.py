"""
File Name: narrative_propagation.py
Module: Narrative Analysis - Propagation Dynamics
Description:
    Analyzes how narrative signals propagate within a piece of text. The module
    estimates narrative spread, reinforcement, and persistence by tracking
    repeated narrative frames, thematic continuity, and cross-sentence narrative
    reinforcement. These signals help the TruthLens AI system identify how
    strongly a narrative is being pushed and whether it is repeatedly reinforced
    throughout the discourse.
    The module detects structured narrative conflict patterns using:

    1. Conflict verb ontology
    2. Opposition framing markers
    3. Polarization language detection
    4. Phrase-level narrative conflict patterns
    5. Actor-role narrative modeling (hero / villain / victim)

Dependencies:
    logging
    typing
    collections
    dataclasses
    numpy
    spacy

Inputs:
    Raw text string

Outputs:
    Narrative propagation feature dictionary and optional numerical vector
"""


from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import spacy
from spacy.language import Language
from spacy.tokens import Doc

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

@dataclass(slots=True)
class NarrativeConflictConfig:

    spacy_model: str = "en_core_web_sm"
    normalize_ratios: bool = True


# ---------------------------------------------------------
# Conflict Lexicons
# ---------------------------------------------------------

CONFLICT_VERBS = {

    "violent_conflict": {
        "attack","assault","strike","bomb","invade","raid",
        "kill","destroy","eliminate","retaliate","counterattack",
        "engage","fight","battle","clash","ambush"
    },

    "political_conflict": {
        "oppose","challenge","confront","block","resist",
        "defy","undermine","topple","overthrow","obstruct",
        "contest","counter"
    },

    "discursive_conflict": {
        "accuse","blame","criticize","condemn","denounce",
        "slam","rebuke","mock","dismiss","discredit"
    },

    "institutional_conflict": {
        "sue","investigate","prosecute","charge",
        "sanction","indict","impeach"
    },

    "coercion_conflict": {
        "threaten","warn","pressure","intimidate",
        "coerce","target","force"
    },
}


OPPOSITION_MARKERS = {

    "against",
    "versus",
    "vs",
    "opposed",
    "opposing",

    "conflict",
    "confrontation",
    "showdown",
    "standoff",

    "rival",
    "rivalry",
    "competitor",
    "adversary",

    "struggle",
    "battle",
    "fight",
    "clash",

    "power_struggle",
    "political_fight",
    "ideological_clash",
}


POLARIZATION_TERMS = {

    "us",
    "we",
    "our",
    "ours",

    "them",
    "they",
    "their",
    "others",

    "enemy",
    "opponent",
    "adversary",

    "elite",
    "establishment",
    "globalists",

    "extremists",
    "radicals",

    "the_people",
    "ordinary_people",
    "corrupt_elites",
}


CONFLICT_PHRASES = {

    "war against",
    "fight against",
    "battle against",
    "clash with",
    "conflict with",
    "power struggle",
    "political fight",
    "ideological battle",
    "direct confrontation",
    "rising tensions",
    "growing conflict",
}


QUESTION_PATTERN = re.compile(r"\?")
EXCLAMATION_PATTERN = re.compile(r"!")

# ---------------------------------------------------------
# Analyzer
# ---------------------------------------------------------

class NarrativeConflictAnalyzer:

    """
    Extract adversarial narrative structures from text.
    """

    def __init__(self, config: Optional[NarrativeConflictConfig] = None):

        self.config = config or NarrativeConflictConfig()

        try:
            self.nlp: Language = spacy.load(self.config.spacy_model)
        except Exception as exc:
            logger.exception("spaCy model loading failed")
            raise RuntimeError("Failed to load spaCy model") from exc

        logger.info("NarrativeConflictAnalyzer initialized")


    # -----------------------------------------------------
    # Main analysis
    # -----------------------------------------------------

    def analyze(
        self,
        text: str,
        hero_entities: Optional[List[str]] = None,
        villain_entities: Optional[List[str]] = None,
        victim_entities: Optional[List[str]] = None,
    ) -> Dict[str, float]:

        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input text must be non-empty")

        doc: Doc = self.nlp(text)

        tokens = [t.lemma_.lower() for t in doc if t.is_alpha]

        features: Dict[str, float] = {}

        features.update(self._conflict_verbs(tokens))
        features.update(self._opposition_markers(tokens))
        features.update(self._polarization(tokens))
        features.update(self._conflict_phrases(text.lower()))
        features.update(self._actor_roles(text, hero_entities, villain_entities, victim_entities))
        features.update(self._punctuation_features(text))

        return features


    # -----------------------------------------------------
    # Conflict verbs
    # -----------------------------------------------------

    def _conflict_verbs(self, tokens: List[str]) -> Dict[str, float]:

        features = {}
        total_tokens = max(len(tokens), 1)

        for category, verbs in CONFLICT_VERBS.items():

            count = sum(1 for t in tokens if t in verbs)

            features[f"{category}_ratio"] = count / total_tokens

        return features


    # -----------------------------------------------------
    # Opposition markers
    # -----------------------------------------------------

    def _opposition_markers(self, tokens: List[str]) -> Dict[str, float]:

        total_tokens = max(len(tokens), 1)

        count = sum(1 for t in tokens if t in OPPOSITION_MARKERS)

        return {"opposition_marker_ratio": count / total_tokens}


    # -----------------------------------------------------
    # Polarization
    # -----------------------------------------------------

    def _polarization(self, tokens: List[str]) -> Dict[str, float]:

        total_tokens = max(len(tokens), 1)

        count = sum(1 for t in tokens if t in POLARIZATION_TERMS)

        return {"polarization_ratio": count / total_tokens}


    # -----------------------------------------------------
    # Phrase detection
    # -----------------------------------------------------

    def _conflict_phrases(self, text: str) -> Dict[str, float]:

        count = sum(1 for phrase in CONFLICT_PHRASES if phrase in text)

        return {"conflict_phrase_count": float(count)}


    # -----------------------------------------------------
    # Actor roles
    # -----------------------------------------------------

    def _actor_roles(
        self,
        text: str,
        heroes: Optional[List[str]],
        villains: Optional[List[str]],
        victims: Optional[List[str]],
    ) -> Dict[str, float]:

        text = text.lower()

        heroes = heroes or []
        villains = villains or []
        victims = victims or []

        hero_mentions = sum(text.count(h.lower()) for h in heroes)
        villain_mentions = sum(text.count(v.lower()) for v in villains)
        victim_mentions = sum(text.count(v.lower()) for v in victims)

        hero_villain_conflict = min(hero_mentions, villain_mentions)
        villain_victim_harm = min(villain_mentions, victim_mentions)
        hero_victim_protection = min(hero_mentions, victim_mentions)

        return {

            "hero_mentions": float(hero_mentions),
            "villain_mentions": float(villain_mentions),
            "victim_mentions": float(victim_mentions),

            "hero_villain_conflict_score": float(hero_villain_conflict),
            "villain_victim_harm_score": float(villain_victim_harm),
            "hero_victim_protection_score": float(hero_victim_protection),
        }


    # -----------------------------------------------------
    # Punctuation rhetoric
    # -----------------------------------------------------

    def _punctuation_features(self, text: str):

        length = max(len(text), 1)

        return {

            "conflict_exclamation_ratio":
                len(EXCLAMATION_PATTERN.findall(text)) / length,

            "conflict_question_ratio":
                len(QUESTION_PATTERN.findall(text)) / length,
        }


# ---------------------------------------------------------
# Vector Conversion
# ---------------------------------------------------------

def narrative_conflict_vector(features: Dict[str, float]) -> np.ndarray:

    ordered_keys = [

        "violent_conflict_ratio",
        "political_conflict_ratio",
        "discursive_conflict_ratio",
        "institutional_conflict_ratio",
        "coercion_conflict_ratio",

        "opposition_marker_ratio",
        "polarization_ratio",
        "conflict_phrase_count",

        "hero_mentions",
        "villain_mentions",
        "victim_mentions",

        "hero_villain_conflict_score",
        "villain_victim_harm_score",
        "hero_victim_protection_score",
    ]

    return np.array(
        [float(features.get(k, 0.0)) for k in ordered_keys],
        dtype=np.float32,
    )