"""
File Name: token_alignment.py
Module: Explainability - Token Alignment
Description:
    Provides utilities for aligning tokenizer subword tokens (WordPiece,
    BPE, SentencePiece) with original words. Transformer tokenizers often
    split words into multiple subword tokens. For explainability methods
    (SHAP, Integrated Gradients, Attention), it is often necessary to
    aggregate these subwords back into full-word importance scores.

    This module reconstructs words from token pieces and merges their
    corresponding importance scores.

Author: TruthLens Engineering Team
Date: 2026-04-02

Dependencies:
    logging
    typing
    dataclasses
    numpy

Inputs:
    tokens: List[str]          # tokenizer output tokens
    scores: List[float]        # token-level importance scores

Outputs:
    aligned_tokens: List[str]
    aligned_scores: List[float]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AlignmentResult:
    """
    Result container for token alignment.
    """

    tokens: List[str]
    scores: List[float]


class TokenAlignment:
    """
    Aligns subword tokens back into full words.
    """

    def __init__(self) -> None:
        logger.info("TokenAlignment initialized")

    @staticmethod
    def _is_subword(token: str) -> bool:
        """
        Detect if token is a continuation subword.

        Supports:
        - WordPiece (##token)
        - SentencePiece (_token / ▁token)
        - BPE (common merges)
        """
        if token.startswith("##"):
            return True

        if token.startswith("▁"):
            return False

        return False

    @staticmethod
    def _clean_token(token: str) -> str:
        """
        Normalize tokenizer artifacts.
        """
        token = token.replace("##", "")
        token = token.replace("▁", "")
        return token

    def align(
        self,
        tokens: List[str],
        scores: List[float],
    ) -> Tuple[List[str], List[float]]:
        """
        Merge subword tokens into full words.

        Parameters
        ----------
        tokens : List[str]
            Tokenizer output tokens.
        scores : List[float]
            Token-level importance scores.

        Returns
        -------
        Tuple[List[str], List[float]]
            Aligned tokens and aggregated scores.
        """

        if not tokens or not scores:
            raise ValueError("tokens and scores must not be empty")

        if len(tokens) != len(scores):
            raise ValueError("tokens and scores must have the same length")

        merged_tokens: List[str] = []
        merged_scores: List[float] = []

        current_token = ""
        current_scores: List[float] = []

        for token, score in zip(tokens, scores):
            if token.startswith("##"):
                current_token += self._clean_token(token)
                current_scores.append(score)

            else:
                if current_token:
                    merged_tokens.append(current_token)
                    merged_scores.append(float(np.mean(current_scores)))

                current_token = self._clean_token(token)
                current_scores = [score]

        if current_token:
            merged_tokens.append(current_token)
            merged_scores.append(float(np.mean(current_scores)))

        return merged_tokens, merged_scores

    def align_to_words(
        self,
        tokens: List[str],
        scores: List[float],
    ) -> AlignmentResult:
        """
        Align tokens and return structured result.
        """

        aligned_tokens, aligned_scores = self.align(tokens, scores)

        return AlignmentResult(
            tokens=aligned_tokens,
            scores=aligned_scores,
        )