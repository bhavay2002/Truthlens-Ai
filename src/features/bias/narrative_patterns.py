"""
TruthLens AI
Narrative Pattern Detection Module

Detects narrative structures used in persuasive or propagandistic text.

Capabilities
------------
• spaCy Named Entity Recognition
• Dependency-based event extraction
• Narrative role classification
• Narrative pattern detection
• Token-level bias heatmap

Narratives detected
-------------------
victim_narrative
enemy_narrative
fear_narrative
hero_narrative
"""

from __future__ import annotations

import re
from typing import Dict, List
from dataclasses import dataclass
from collections import Counter

import spacy

# ---------------------------------------------------------
# Load NLP Model
# ---------------------------------------------------------

NLP_MODEL = "en_core_web_sm"
_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load(NLP_MODEL)
    return _nlp


# ---------------------------------------------------------
# Narrative Lexicons
# ---------------------------------------------------------

VICTIM_WORDS = {
    "victim",
    "suffer",
    "oppressed",
    "abused",
    "targeted",
    "marginalized",
    "hurt",
    "persecuted",
}

ENEMY_WORDS = {
    "corrupt",
    "evil",
    "traitor",
    "enemy",
    "criminal",
    "dangerous",
    "radical",
    "extremist",
    "destructive",
}

FEAR_WORDS = {
    "crisis",
    "threat",
    "danger",
    "disaster",
    "collapse",
    "chaos",
    "panic",
    "catastrophe",
}

HERO_WORDS = {
    "hero",
    "savior",
    "rescued",
    "defended",
    "protect",
    "brave",
    "leader",
    "champion",
}

BIAS_LEXICON = VICTIM_WORDS | ENEMY_WORDS | FEAR_WORDS | HERO_WORDS


# ---------------------------------------------------------
# Data Structures
# ---------------------------------------------------------


@dataclass
class NarrativeResult:
    narrative_type: str
    narrative_score: float
    detected_patterns: List[Dict]
    narrative_events: List[Dict]
    bias_heatmap: List[Dict]


# ---------------------------------------------------------
# Sentence Tokenization
# ---------------------------------------------------------


def split_sentences(text: str) -> List[str]:
    sentences = re.split(r"[.!?]+", text)
    return [s.strip() for s in sentences if s.strip()]


# ---------------------------------------------------------
# Entity Extraction
# ---------------------------------------------------------


def extract_entities(doc) -> List[str]:
    return [ent.text for ent in doc.ents]


# ---------------------------------------------------------
# Event Extraction
# ---------------------------------------------------------


def extract_events(doc) -> List[Dict]:
    """
    Extract simple actor-action-target structures
    using dependency parsing.
    """

    events = []

    for token in doc:

        if token.dep_ == "ROOT" and token.pos_ == "VERB":

            actor = None
            target = None

            for child in token.children:

                if child.dep_ in {"nsubj", "nsubjpass"}:
                    actor = child.text

                if child.dep_ in {"dobj", "pobj"}:
                    target = child.text

            events.append(
                {"actor": actor, "action": token.lemma_, "target": target}
            )

    return events


# ---------------------------------------------------------
# Narrative Role Classification
# ---------------------------------------------------------


def classify_event(event: Dict) -> str:

    action = event["action"]
    target = event["target"]

    words = {action, target} if target else {action}

    if words & HERO_WORDS:
        return "hero_narrative"

    if words & VICTIM_WORDS:
        return "victim_narrative"

    if words & ENEMY_WORDS:
        return "enemy_narrative"

    if words & FEAR_WORDS:
        return "fear_narrative"

    return "neutral"


# ---------------------------------------------------------
# Token-Level Bias Heatmap
# ---------------------------------------------------------


def generate_bias_heatmap(doc) -> List[Dict]:

    heatmap = []

    for token in doc:

        if token.text.lower() in BIAS_LEXICON:

            heatmap.append(
                {
                    "token": token.text,
                    "bias_type": "narrative_bias",
                    "position": token.i,
                }
            )

    return heatmap


# ---------------------------------------------------------
# Narrative Pattern Detection
# ---------------------------------------------------------


def detect_narrative_patterns(text: str) -> Dict:

    sentences = split_sentences(text)

    detected_patterns = []
    narrative_events = []
    pattern_counter = Counter()

    bias_heatmap = []

    for sentence in sentences:

        doc = _get_nlp()(sentence)

        events = extract_events(doc)

        for event in events:

            narrative_type = classify_event(event)

            if narrative_type != "neutral":

                detected_patterns.append(
                    {"sentence": sentence, "pattern": narrative_type}
                )

                pattern_counter[narrative_type] += 1

                narrative_events.append(
                    {
                        "sentence": sentence,
                        "event": event,
                        "narrative": narrative_type,
                    }
                )

        bias_heatmap.extend(generate_bias_heatmap(doc))

    if not pattern_counter:
        return {
            "narrative_type": "neutral",
            "narrative_score": 0.0,
            "detected_patterns": [],
            "narrative_events": [],
            "bias_heatmap": [],
        }

    dominant_narrative = pattern_counter.most_common(1)[0][0]

    narrative_score = round(
        sum(pattern_counter.values()) / max(len(sentences), 1), 4
    )

    return {
        "narrative_type": dominant_narrative,
        "narrative_score": narrative_score,
        "detected_patterns": detected_patterns,
        "narrative_events": narrative_events,
        "bias_heatmap": bias_heatmap,
    }


# ---------------------------------------------------------
# Example Usage
# ---------------------------------------------------------

if __name__ == "__main__":

    text = """
    The corrupt government has become an enemy of the people.
    Citizens are suffering under disastrous policies.
    But the brave leader rescued the nation.
    """

    result = detect_narrative_patterns(text)

    print(result)
