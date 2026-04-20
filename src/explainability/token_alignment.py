from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


def align_tokens(
    tokens: Sequence[str],
    scores: Sequence[float],
    tokenizer_type: str = "wordpiece",
) -> Tuple[List[str], List[float]]:
    if len(tokens) != len(scores):
        raise ValueError("tokens and scores must match")

    if tokenizer_type not in {"wordpiece", "sentencepiece"}:
        raise ValueError("tokenizer_type must be 'wordpiece' or 'sentencepiece'")

    merged_tokens: List[str] = []
    merged_scores: List[float] = []

    current_token = ""
    current_scores: List[float] = []

    for token, score in zip(tokens, scores):
        token = str(token)
        score = float(score)

        if tokenizer_type == "wordpiece":
            if token.startswith("##"):
                piece = token[2:]
                current_token = (current_token + piece) if current_token else piece
                current_scores.append(score)
            else:
                if current_token:
                    merged_tokens.append(current_token)
                    merged_scores.append(float(np.mean(current_scores)))
                current_token = token
                current_scores = [score]

        else:  # sentencepiece
            if token.startswith("▁"):
                if current_token:
                    merged_tokens.append(current_token)
                    merged_scores.append(float(np.mean(current_scores)))
                current_token = token[1:]
                current_scores = [score]
            else:
                current_token += token
                current_scores.append(score)

    if current_token:
        merged_tokens.append(current_token)
        merged_scores.append(float(np.mean(current_scores)))

    return merged_tokens, merged_scores
