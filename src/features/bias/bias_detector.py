"""
TruthLens AI
Bias Detection Orchestration Module

This module serves as the central aggregation layer for bias analysis.
It integrates multiple bias detection components to produce a
comprehensive bias assessment for a news article.

Signals Integrated
------------------
• Lexicon Bias
• Framing Bias
• Narrative Bias
• Subjectivity Score
• Sentence-Level Bias Detection

Outputs
-------
bias_score
bias_type
biased_sentences
bias_components
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from statistics import mean
from typing import Dict, List

from .bias_lexicon import compute_bias_features
from .framing_detector import detect_framing_bias
from .narrative_patterns import detect_narrative_patterns

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

BIAS_WEIGHTS = {
    "lexicon": 0.35,
    "subjectivity": 0.25,
    "framing": 0.20,
    "narrative": 0.20,
}

SENTENCE_BIAS_THRESHOLD = 0.08


# ---------------------------------------------------------
# Subjectivity Lexicon
# ---------------------------------------------------------

SUBJECTIVE_WORDS = {
    "clearly",
    "obviously",
    "undoubtedly",
    "certainly",
    "terrible",
    "amazing",
    "horrible",
    "fantastic",
    "unbelievable",
    "ridiculous",
    "absurd",
    "disgraceful",
    "shocking",
    "disturbing",
}


# ---------------------------------------------------------
# Result Data Structures
# ---------------------------------------------------------


@dataclass
class BiasDetectionResult:
    bias_score: float
    bias_type: str
    biased_sentences: List[str]
    bias_components: Dict[str, float]
    sentence_bias_scores: List[Dict]


# ---------------------------------------------------------
# Text Processing Utilities
# ---------------------------------------------------------


def tokenize_words(text: str) -> List[str]:
    """
    Tokenize text into lowercase word tokens.
    """
    return re.findall(r"\b[a-z]+\b", text.lower())


def tokenize_sentences(text: str) -> List[str]:
    """
    Lightweight sentence tokenizer.
    """
    sentences = re.split(r"[.!?]+", text)
    return [s.strip() for s in sentences if s.strip()]


# ---------------------------------------------------------
# Subjectivity Detection
# ---------------------------------------------------------


def compute_subjectivity_score(text: str) -> float:
    """
    Estimate subjectivity using lexicon density.
    """

    words = tokenize_words(text)

    if not words:
        return 0.0

    subjective_count = sum(1 for word in words if word in SUBJECTIVE_WORDS)

    score = subjective_count / len(words)

    return round(score, 4)


# ---------------------------------------------------------
# Sentence-Level Bias Detection
# ---------------------------------------------------------


def compute_sentence_bias(text: str) -> List[Dict]:
    """
    Compute bias score per sentence.
    """

    sentences = tokenize_sentences(text)

    sentence_scores = []

    for sentence in sentences:

        lexicon_result = compute_bias_features(sentence)

        score = lexicon_result.bias_score

        sentence_scores.append({"sentence": sentence, "bias_score": score})

    return sentence_scores


# ---------------------------------------------------------
# Bias Type Classification
# ---------------------------------------------------------


def determine_bias_type(
    lexicon_score: float,
    subjectivity_score: float,
    framing_score: float,
    narrative_score: float,
) -> str:
    """
    Determine dominant bias signal.
    """

    scores = {
        "lexical_bias": lexicon_score,
        "subjective_bias": subjectivity_score,
        "framing_bias": framing_score,
        "narrative_bias": narrative_score,
    }

    dominant_bias = max(scores, key=scores.get)

    return dominant_bias


# ---------------------------------------------------------
# Main Bias Detection Pipeline
# ---------------------------------------------------------


class BiasDetector:
    """
    Central bias detection engine for TruthLens AI.
    """

    def analyze(self, text: str) -> BiasDetectionResult:
        """
        Perform full bias analysis.
        """

        # ----------------------------------------------
        # Sentence-level bias
        # ----------------------------------------------

        sentence_scores = compute_sentence_bias(text)

        biased_sentences = [
            s["sentence"]
            for s in sentence_scores
            if s["bias_score"] > SENTENCE_BIAS_THRESHOLD
        ]

        avg_sentence_bias = (
            mean(s["bias_score"] for s in sentence_scores)
            if sentence_scores
            else 0.0
        )

        # ----------------------------------------------
        # Lexicon Bias
        # ----------------------------------------------

        lexicon_result = compute_bias_features(text)
        lexicon_bias = lexicon_result.bias_score

        # ----------------------------------------------
        # Subjectivity
        # ----------------------------------------------

        subjectivity_score = compute_subjectivity_score(text)

        # ----------------------------------------------
        # Framing Bias
        # ----------------------------------------------

        framing_result = detect_framing_bias(text)
        framing_score = framing_result.get("framing_score", 0.0)

        # ----------------------------------------------
        # Narrative Bias
        # ----------------------------------------------

        narrative_result = detect_narrative_patterns(text)
        narrative_score = narrative_result.get("narrative_score", 0.0)

        # ----------------------------------------------
        # Weighted Bias Aggregation
        # ----------------------------------------------

        bias_score = (
            BIAS_WEIGHTS["lexicon"] * lexicon_bias
            + BIAS_WEIGHTS["subjectivity"] * subjectivity_score
            + BIAS_WEIGHTS["framing"] * framing_score
            + BIAS_WEIGHTS["narrative"] * narrative_score
        )

        bias_score = round(bias_score, 4)

        # ----------------------------------------------
        # Bias Type Classification
        # ----------------------------------------------

        bias_type = determine_bias_type(
            lexicon_bias, subjectivity_score, framing_score, narrative_score
        )

        # ----------------------------------------------
        # Component Breakdown
        # ----------------------------------------------

        bias_components = {
            "lexicon_bias": round(lexicon_bias, 4),
            "subjectivity": round(subjectivity_score, 4),
            "framing_bias": round(framing_score, 4),
            "narrative_bias": round(narrative_score, 4),
            "sentence_bias": round(avg_sentence_bias, 4),
        }

        return BiasDetectionResult(
            bias_score=bias_score,
            bias_type=bias_type,
            biased_sentences=biased_sentences,
            bias_components=bias_components,
            sentence_bias_scores=sentence_scores,
        )


def detect_bias(text: str) -> Dict:
    """
    Backward-compatible helper for callers that expect a dict response.
    """

    result = BiasDetector().analyze(text)

    return asdict(result)


# ---------------------------------------------------------
# Example Usage
# ---------------------------------------------------------

if __name__ == "__main__":

    detector = BiasDetector()

    text = """
    The shocking decision by the corrupt regime proves the elite
    are manipulating the public. Obviously this disastrous policy
    will destroy the economy.
    """

    result = detector.analyze(text)

    print("Bias Score:", result.bias_score)
    print("Bias Type:", result.bias_type)
    print("Biased Sentences:", result.biased_sentences)
