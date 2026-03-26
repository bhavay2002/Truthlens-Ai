"""
File Name: bias_lexicon.py
Module: Feature Engineering - Bias Lexicon Management

Description:
    Supports loading, normalization, validation, and querying of
    bias-related lexicons across multiple formats.

    Features:
        • multi-format lexicon loading
        • phrase lexicon support
        • weighted lexicons
        • fast token lookup
        • batch token matching
        • lexicon statistics and diagnostics

Dependencies:
    logging
    pathlib
    typing
    json
    yaml
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Iterable, Tuple, Optional

import yaml


logger = logging.getLogger(__name__)


class BiasLexiconManager:
    """
    Advanced lexicon manager for bias detection.

    Supports:
        • token lexicons
        • phrase lexicons
        • weighted lexicons
    """

    SUPPORTED_EXTENSIONS = {".txt", ".json", ".yaml", ".yml"}

    def __init__(self, lexicon_path: str) -> None:

        if not isinstance(lexicon_path, str) or not lexicon_path.strip():
            raise ValueError("lexicon_path must be a non-empty string")

        self.lexicon_path: Path = Path(lexicon_path)

        if not self.lexicon_path.exists():
            raise FileNotFoundError(f"Lexicon file not found: {lexicon_path}")

        if self.lexicon_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported format: {self.lexicon_path.suffix}")

        # token lexicon
        self.lexicon: Set[str] = set()

        # phrase lexicon (multi-word)
        self.phrase_lexicon: Set[str] = set()

        # optional weights
        self.weights: Dict[str, float] = {}

        self._load_lexicon()

        logger.info(
            "Bias lexicon loaded: %d tokens | %d phrases",
            len(self.lexicon),
            len(self.phrase_lexicon),
        )

    # -----------------------------------------------------

    def _load_lexicon(self) -> None:

        suffix = self.lexicon_path.suffix.lower()

        try:

            if suffix == ".txt":
                terms = self._load_txt()

            elif suffix == ".json":
                terms = self._load_json()

            elif suffix in {".yaml", ".yml"}:
                terms = self._load_yaml()

            else:
                raise ValueError("Unsupported lexicon format")

            self._parse_terms(terms)

        except Exception as exc:
            logger.exception("Lexicon loading failed")
            raise RuntimeError("Lexicon initialization failed") from exc

    # -----------------------------------------------------

    def _parse_terms(self, terms: Iterable) -> None:
        """
        Parse terms and detect:
            • tokens
            • phrases
            • weighted entries
        """

        for item in terms:

            if isinstance(item, str):

                term = item.strip().lower()

                if " " in term:
                    self.phrase_lexicon.add(term)
                else:
                    self.lexicon.add(term)

            elif isinstance(item, dict):

                term = item.get("term")
                weight = item.get("weight", 1.0)

                if not isinstance(term, str):
                    continue

                term = term.strip().lower()

                if " " in term:
                    self.phrase_lexicon.add(term)
                else:
                    self.lexicon.add(term)

                self.weights[term] = float(weight)

    # -----------------------------------------------------

    def _load_txt(self) -> List[str]:

        with self.lexicon_path.open("r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    # -----------------------------------------------------

    def _load_json(self) -> List:

        with self.lexicon_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return data

        if isinstance(data, dict):

            terms: List = []

            for value in data.values():
                if isinstance(value, list):
                    terms.extend(value)

            return terms

        raise ValueError("Invalid JSON lexicon structure")

    # -----------------------------------------------------

    def _load_yaml(self) -> List:

        with self.lexicon_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if isinstance(data, list):
            return data

        if isinstance(data, dict):

            terms: List = []

            for value in data.values():
                if isinstance(value, list):
                    terms.extend(value)

            return terms

        raise ValueError("Invalid YAML lexicon structure")

    # -----------------------------------------------------

    def contains(self, token: str) -> bool:

        if not isinstance(token, str):
            raise ValueError("token must be a string")

        return token.lower().strip() in self.lexicon

    # -----------------------------------------------------

    def phrase_matches(self, text: str) -> List[str]:
        """
        Detect phrase lexicon matches.
        """

        text = text.lower()

        matches = []

        for phrase in self.phrase_lexicon:
            if phrase in text:
                matches.append(phrase)

        return matches

    # -----------------------------------------------------

    def count_matches(self, tokens: Iterable[str]) -> int:

        matches = 0

        for token in tokens:

            if isinstance(token, str) and token.lower().strip() in self.lexicon:
                matches += 1

        return matches

    # -----------------------------------------------------

    def weighted_score(self, tokens: Iterable[str]) -> float:
        """
        Compute weighted bias score using lexicon weights.
        """

        score = 0.0

        for token in tokens:

            token = token.lower().strip()

            if token in self.weights:
                score += self.weights[token]

            elif token in self.lexicon:
                score += 1.0

        return score

    # -----------------------------------------------------

    def batch_match(self, documents: Iterable[List[str]]) -> List[int]:
        """
        Efficient lexicon match counts for multiple documents.
        """

        results: List[int] = []

        for tokens in documents:
            results.append(self.count_matches(tokens))

        return results

    # -----------------------------------------------------

    def coverage(self, tokens: Iterable[str]) -> float:
        """
        Measure lexicon coverage over token list.
        """

        tokens = list(tokens)

        if not tokens:
            return 0.0

        matches = self.count_matches(tokens)

        return matches / len(tokens)

    # -----------------------------------------------------

    def get_statistics(self) -> Dict[str, int]:

        return {
            "token_terms": len(self.lexicon),
            "phrase_terms": len(self.phrase_lexicon),
            "weighted_terms": len(self.weights),
        }

    # -----------------------------------------------------

    def get_all_terms(self) -> Set[str]:

        return set(self.lexicon)

    # -----------------------------------------------------

    def export(self) -> Dict:
        """
        Export lexicon for reproducibility.
        """

        return {
            "tokens": list(self.lexicon),
            "phrases": list(self.phrase_lexicon),
            "weights": self.weights,
        }