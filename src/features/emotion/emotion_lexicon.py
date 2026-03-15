"""
TruthLens AI
Emotion Detection using NRC Emotion Lexicon

Detects emotional signals in text using the NRC Emotion Lexicon.

Capabilities
------------
• Emotion intensity scoring
• Token-level emotion attribution
• Sentence-level emotion analysis
• Dominant emotion detection
• Support for full NRC lexicon loading

Emotions detected
-----------------
fear
anger
joy
sadness
surprise
disgust
"""

from __future__ import annotations

import re
from typing import Dict, List
from dataclasses import dataclass
from collections import defaultdict, Counter
from statistics import mean


# ---------------------------------------------------------
# Emotion Categories
# ---------------------------------------------------------

EMOTIONS = [
    "fear",
    "anger",
    "joy",
    "sadness",
    "surprise",
    "disgust"
]


# ---------------------------------------------------------
# Default Lightweight NRC Subset
# ---------------------------------------------------------

DEFAULT_NRC_LEXICON = {

    "fear": {
        "fear", "threat", "terror", "danger", "panic",
        "risk", "crisis", "afraid", "scared", "fright"
    },

    "anger": {
        "anger", "rage", "furious", "outrage", "hate",
        "hostile", "violent", "attack", "fight"
    },

    "joy": {
        "joy", "happy", "celebrate", "success",
        "victory", "delight", "excited", "pleased"
    },

    "sadness": {
        "sad", "grief", "sorrow", "loss", "tragic",
        "depressed", "cry", "mourning"
    },

    "surprise": {
        "surprise", "unexpected", "suddenly",
        "shocking", "astonishing"
    },

    "disgust": {
        "disgust", "repulsive", "dirty",
        "corrupt", "filthy", "gross"
    }
}


# ---------------------------------------------------------
# Data Structures
# ---------------------------------------------------------

@dataclass
class EmotionResult:
    emotion_scores: Dict[str, float]
    dominant_emotion: str
    emotion_distribution: Dict[str, int]
    sentence_emotions: List[Dict]
    emotion_tokens: List[Dict]


# ---------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------

def tokenize_words(text: str) -> List[str]:
    """
    Tokenize words from text.
    """
    return re.findall(r"\b[a-z]+\b", text.lower())


def tokenize_sentences(text: str) -> List[str]:
    """
    Lightweight sentence tokenizer.
    """
    sentences = re.split(r"[.!?]+", text)
    return [s.strip() for s in sentences if s.strip()]


# ---------------------------------------------------------
# NRC Lexicon Loader
# ---------------------------------------------------------

def load_nrc_lexicon(path: str) -> Dict[str, set]:
    """
    Load NRC Emotion Lexicon from file.

    Expected format:
        word emotion association
    """

    lexicon = defaultdict(set)

    with open(path, "r", encoding="utf-8") as f:

        for line in f:

            word, emotion, association = line.strip().split()

            if emotion in EMOTIONS and association == "1":
                lexicon[emotion].add(word)

    return dict(lexicon)


# ---------------------------------------------------------
# Emotion Analyzer
# ---------------------------------------------------------

class EmotionLexiconAnalyzer:
    """
    Emotion detection engine using NRC lexicon.
    """

    def __init__(self, lexicon: Dict[str, set] | None = None):

        self.lexicon = lexicon or DEFAULT_NRC_LEXICON

    # -----------------------------------------------------

    def _detect_token_emotions(self, tokens: List[str]):

        emotion_counts = Counter()

        emotion_tokens = []

        for idx, token in enumerate(tokens):

            for emotion, words in self.lexicon.items():

                if token in words:

                    emotion_counts[emotion] += 1

                    emotion_tokens.append({
                        "token": token,
                        "emotion": emotion,
                        "position": idx
                    })

        return emotion_counts, emotion_tokens

    # -----------------------------------------------------

    def _compute_scores(self, emotion_counts, total_tokens):

        if total_tokens == 0:
            return {emotion: 0.0 for emotion in EMOTIONS}

        scores = {
            emotion: round(emotion_counts.get(emotion, 0) / total_tokens, 4)
            for emotion in EMOTIONS
        }

        return scores

    # -----------------------------------------------------

    def _sentence_level_analysis(self, text: str):

        sentences = tokenize_sentences(text)

        results = []

        for sentence in sentences:

            tokens = tokenize_words(sentence)

            counts, _ = self._detect_token_emotions(tokens)

            score = sum(counts.values()) / max(len(tokens), 1)

            results.append({
                "sentence": sentence,
                "emotion_intensity": round(score, 4)
            })

        return results

    # -----------------------------------------------------

    def analyze(self, text: str) -> EmotionResult:
        """
        Perform full emotion analysis.
        """

        tokens = tokenize_words(text)

        emotion_counts, emotion_tokens = self._detect_token_emotions(tokens)

        scores = self._compute_scores(emotion_counts, len(tokens))

        dominant_emotion = (
            max(scores, key=scores.get)
            if any(scores.values())
            else "neutral"
        )

        sentence_analysis = self._sentence_level_analysis(text)

        return EmotionResult(
            emotion_scores=scores,
            dominant_emotion=dominant_emotion,
            emotion_distribution=dict(emotion_counts),
            sentence_emotions=sentence_analysis,
            emotion_tokens=emotion_tokens
        )


# ---------------------------------------------------------
# Example Usage
# ---------------------------------------------------------

if __name__ == "__main__":

    analyzer = EmotionLexiconAnalyzer()

    text = """
    The shocking crisis caused fear and panic among citizens.
    Many people were angry and furious about the situation.
    """

    result = analyzer.analyze(text)

    print(result)