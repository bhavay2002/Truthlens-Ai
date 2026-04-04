"""
File Name: narrative_role_extractor.py
Module: Narrative Analysis - Narrative Role Extraction
Description:
    Extracts narrative roles (Hero, Villain, Victim) from text for the TruthLens AI system.
    
    This module detects actor roles using:
    
    1. Narrative action lexicons
    2. Dependency-based actor relations
    3. Passive voice victim detection
    4. Named entity resolution
    
    Roles help downstream modules detect propaganda narratives such as:
    
    Hero → protects → Victim
    Villain → harms → Victim
    Hero ↔ Villain conflict

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
from typing import Dict, List, Set

import numpy as np
import spacy
from spacy.language import Language
from spacy.tokens import Doc, Token

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

@dataclass(slots=True)
class NarrativeRoleConfig:
    spacy_model: str = "en_core_web_sm"


# ---------------------------------------------------------
# Role Extractor
# ---------------------------------------------------------

class NarrativeRoleExtractor:

    """
    Extracts Hero / Villain / Victim narrative roles.
    """

    HERO_TERMS ={

    # Protection / defense
    "protect",
    "defend",
    "shield",
    "guard",
    "secure",
    "rescue",
    "save",

    # Assistance / support
    "help",
    "aid",
    "assist",
    "support",
    "relieve",
    "provide",
    "deliver",

    # Leadership / guidance
    "lead",
    "guide",
    "champion",
    "represent",
    "advocate",

    # Resistance against harm
    "fight_for",
    "stand_for",
    "stand_with",
    "defend_rights",
    "protect_rights",

    # Humanitarian actions
    "rebuild",
    "restore",
    "stabilize",
    "reform",

        
    }

    VILLAIN_TERMS = {
        
    # Physical aggression
    "attack",
    "assault",
    "bomb",
    "invade",
    "raid",
    "strike",
    "kill",
    "destroy",

    # Harm / damage
    "harm",
    "injure",
    "damage",
    "target",

    # Oppression / exploitation
    "exploit",
    "abuse",
    "oppress",
    "suppress",
    "control",
    "dominate",

    # Political aggression
    "corrupt",
    "manipulate",
    "rig",
    "undermine",
    "destabilize",

    # Threat / intimidation
    "threaten",
    "intimidate",
    "coerce",
    "pressure",

    # Blame / accusation
    "blame",
    "accuse",
    "condemn",

    }

    VICTIM_TERMS = {
       
    # Direct harm
    "hurt",
    "injure",
    "kill",
    "attack",
    "harm",

    # Suffering
    "suffer",
    "struggle",
    "endure",
    "experience",

    # Damage or loss
    "lose",
    "damage",
    "destroy",

    # Targeting
    "target",
    "attack",
    "persecute",

    # Displacement / crisis
    "displace",
    "evacuate",
    "flee",
    "escape",

    # Economic / social impact
    "affect",
    "impact",
    "burden",

    }

    def __init__(self, config: NarrativeRoleConfig | None = None):

        self.config = config or NarrativeRoleConfig()

        try:
            self.nlp: Language = spacy.load(self.config.spacy_model)
        except Exception as exc:
            logger.exception("spaCy model loading failed")
            raise RuntimeError(
                f"Failed to load spaCy model: {self.config.spacy_model}"
            ) from exc

        logger.info("NarrativeRoleExtractor initialized")

    # -----------------------------------------------------
    # Main Analysis
    # -----------------------------------------------------

    def analyze(self, text: str) -> Dict[str, List[str]]:

        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input text must be non-empty")

        doc: Doc = self.nlp(text)

        hero_entities: Set[str] = set()
        villain_entities: Set[str] = set()
        victim_entities: Set[str] = set()

        for token in doc:

            lemma = token.lemma_.lower()

            # ---------------- HERO ----------------
            if lemma in self.HERO_TERMS:

                subject = self._get_subject(token)
                obj = self._get_object(token)

                if subject:
                    hero_entities.add(subject)

                if obj:
                    victim_entities.add(obj)

            # ---------------- VILLAIN ----------------
            elif lemma in self.VILLAIN_TERMS:

                subject = self._get_subject(token)
                obj = self._get_object(token)

                if subject:
                    villain_entities.add(subject)

                if obj:
                    victim_entities.add(obj)

            # ---------------- VICTIM ----------------
            elif lemma in self.VICTIM_TERMS:

                obj = self._get_object(token)

                if obj:
                    victim_entities.add(obj)

            # ---------------- PASSIVE VOICE ----------------
            if token.dep_ == "nsubjpass":
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


    # -----------------------------------------------------
    # Dependency helpers
    # -----------------------------------------------------

    def _get_subject(self, token: Token) -> str | None:

        for child in token.children:
            if child.dep_ in {"nsubj", "nsubjpass"}:
                return self._resolve_entity(child)

        return None


    def _get_object(self, token: Token) -> str | None:

        for child in token.children:
            if child.dep_ in {"dobj", "pobj", "obj"}:
                return self._resolve_entity(child)

        return None


    def _resolve_entity(self, token: Token) -> str | None:

        if token.ent_type_:
            return token.text.lower()

        if token.pos_ in {"PROPN", "NOUN"}:
            return token.lemma_.lower()

        head = token.head

        if head.ent_type_:
            return head.text.lower()

        return None


# ---------------------------------------------------------
# Vector Representation
# ---------------------------------------------------------

def narrative_role_vector(features: Dict[str, List[str]]) -> np.ndarray:
    """
    Convert narrative roles into vector form.

    Vector structure:
        [num_heroes, num_villains, num_victims, role_balance]
    """

    heroes = features.get("hero_entities", [])
    villains = features.get("villain_entities", [])
    victims = features.get("victim_entities", [])

    num_heroes = len(heroes)
    num_villains = len(villains)
    num_victims = len(victims)

    role_balance = num_heroes - num_villains

    return np.array(
        [
            float(num_heroes),
            float(num_villains),
            float(num_victims),
            float(role_balance),
        ],
        dtype=np.float32,
    )