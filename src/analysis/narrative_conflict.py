"""
File Name: narrative_conflict.py
Module: Narrative Analysis - Conflict Detection
Description:
    Detects narrative conflict structures within text for the TruthLens AI system.
    The module analyzes linguistic signals that indicate opposing actors,
    ideological clashes, threat framing, and adversarial narrative structures.
    These signals help identify conflict-driven narratives frequently used in
    propaganda, political messaging, and ideological discourse.

    This module detects structured narrative conflict patterns using:
    

    1. Conflict verbs and adversarial actions
    2. Opposition framing markers
    3. Polarization language ("us vs them")
    4. Actor role modeling (hero / villain / victim)
    5. Rhetorical punctuation signals
    
    This implementation aligns with :
    
    - Political narrative analysis
    - Propaganda detection
    - Media framing theory
    - Computational narrative modeling

Dependencies:
    logging
    typing
    collections
    numpy
    spacy
    re

Inputs:
    Raw text string

Outputs:
    Narrative conflict feature dictionary and optional numerical vector
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from collections import Counter
from typing import Dict, List, Optional

import numpy as np
from spacy.language import Language
from spacy.tokens import Doc

from src.analysis._nlp import get_nlp
from src.analysis.feature_schema import NARRATIVE_CONFLICT_KEYS, make_vector


logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

@dataclass(slots=True)
class NarrativeConflictConfig:

    spacy_model: str = "en_core_web_sm"
    normalize_ratios: bool = True


# ---------------------------------------------------------
# Analyzer
# ---------------------------------------------------------

class NarrativeConflictAnalyzer:
    """
    Extract adversarial narrative structures from text.
    """

    CONFLICT_VERBS = {

        "attack","assault","strike","bomb","invade","raid",
        "kill","destroy","eliminate","retaliate","counterattack",
        "fight","battle","clash",

        "oppose","challenge","confront","block","resist",
        "defy","undermine","overthrow","topple",

        "accuse","blame","criticize","condemn","denounce",
        "slam","rebuke","mock","dismiss",

        "threaten","warn","pressure","intimidate","coerce",

        "sue","investigate","prosecute","sanction","charge","impeach"
    }


    OPPOSITION_MARKERS = {

        "against","versus","vs","opposed","opposing",

        "conflict","confrontation","showdown","standoff",

        "rival","rivalry","competitor","adversary",

        "struggle","battle","fight","clash",

        "ideological_clash","power_struggle","political_fight",
    }


    POLARIZATION_TERMS = {

        "us","we","our","ours",

        "them","they","their","others",

        "enemy","opponent","adversary",

        "elite","establishment","globalists",

        "extremists","radicals",

        "the_people","ordinary_people","corrupt_elites",
    }


    QUESTION_PATTERN = re.compile(r"\?")
    EXCLAMATION_PATTERN = re.compile(r"!")

    # -----------------------------------------------------

    def __init__(self, config: Optional[NarrativeConflictConfig] = None):

        self.config = config or NarrativeConflictConfig()

        self.nlp: Language = get_nlp(self.config.spacy_model)

        self._conflict_verbs = {t.replace("_", " ").lower() for t in self.CONFLICT_VERBS}
        self._opposition_markers = {t.replace("_", " ").lower() for t in self.OPPOSITION_MARKERS}
        self._polarization_terms = {t.replace("_", " ").lower() for t in self.POLARIZATION_TERMS}

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
        return_vector: bool = False,
    ) -> Dict[str, float] | tuple[Dict[str, float], np.ndarray]:

        if not isinstance(text, str):
            raise TypeError("text must be a string")

        text = text.strip()

        if not text:
            features = {k: 0.0 for k in NARRATIVE_CONFLICT_KEYS}
            return (
                features,
                make_vector(features, NARRATIVE_CONFLICT_KEYS),
            ) if return_vector else features

        doc: Doc = self.nlp(text)
        return self.analyze_doc(
            doc,
            hero_entities=hero_entities,
            villain_entities=villain_entities,
            victim_entities=victim_entities,
        )

    # -----------------------------------------------------

    def analyze_doc(
        self,
        doc: Doc,
        hero_entities: Optional[List[str]] = None,
        villain_entities: Optional[List[str]] = None,
        victim_entities: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Compute narrative conflict features from a pre-built spaCy Doc.

        Args:
            doc:              A processed spaCy Doc instance.
            hero_entities:    Hero entity strings from narrative role extraction.
            villain_entities: Villain entity strings.
            victim_entities:  Victim entity strings.

        Returns:
            Dictionary of narrative conflict feature names to float values.
        """

        tokens = [t.lemma_.lower() for t in doc if t.is_alpha]

        features: Dict[str, float] = {}

        features.update(self._conflict_verb_features(doc))
        features.update(self._opposition_features(doc.text, tokens))
        features.update(self._polarization_features(doc.text, tokens))

        features.update(
            self._actor_conflict_structure(
                doc,
                hero_entities,
                villain_entities,
                victim_entities,
            )
        )

        features.update(self._punctuation_features(doc.text))

        return features


    # -----------------------------------------------------
    # Conflict verbs
    # -----------------------------------------------------

    def _conflict_verb_features(self, doc: Doc) -> Dict[str, float]:

        verbs = [
            token.lemma_.lower()
            for token in doc
            if token.pos_ == "VERB"
        ]

        if not verbs:
            return {"conflict_verb_ratio": 0.0}

        count = sum(1 for v in verbs if v in self._conflict_verbs)

        return {"conflict_verb_ratio": count / len(verbs)}


    # -----------------------------------------------------
    # Opposition markers
    # -----------------------------------------------------

    def _opposition_features(self, text: str, tokens: List[str]):

        if not tokens:
            return {"opposition_marker_ratio": 0.0}

        count = self._count_terms(text, tokens, self._opposition_markers)

        return {"opposition_marker_ratio": count / len(tokens)}


    # -----------------------------------------------------
    # Polarization
    # -----------------------------------------------------

    def _polarization_features(self, text: str, tokens: List[str]):

        if not tokens:
            return {"polarization_ratio": 0.0}

        count = self._count_terms(text, tokens, self._polarization_terms)

        return {"polarization_ratio": count / len(tokens)}


    # -----------------------------------------------------
    # Actor conflict structure
    # -----------------------------------------------------

    def _actor_conflict_structure(
        self,
        doc: Doc,
        heroes: Optional[List[str]],
        villains: Optional[List[str]],
        victims: Optional[List[str]],
    ) -> Dict[str, float]:

        heroes = heroes or []
        villains = villains or []
        victims = victims or []

        text = doc.text.lower()

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
        length = max(len(text.split()), 1)

        return {
            "conflict_exclamation_ratio":
                len(self.EXCLAMATION_PATTERN.findall(text)) / length,

            "conflict_question_ratio":
                len(self.QUESTION_PATTERN.findall(text)) / length,
        }

    def _count_terms(self, text: str, tokens: List[str], terms: set[str]) -> int:
        text_lower = text.lower()
        token_counts = Counter(tokens)
        hits = 0
        for term in terms:
            if " " in term:
                hits += text_lower.count(term)
            else:
                hits += token_counts.get(term, 0)
        return hits


# ---------------------------------------------------------
# Vector conversion
# ---------------------------------------------------------

def narrative_conflict_vector(features: Dict[str, float]) -> np.ndarray:

    return make_vector(features, NARRATIVE_CONFLICT_KEYS)