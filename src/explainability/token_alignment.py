from __future__ import annotations

from typing import List, Sequence, Tuple, Literal
import numpy as np


AggregationMethod = Literal["mean", "sum", "max"]
SPECIAL_TOKENS = {"[CLS]", "[SEP]", "<s>", "</s>", "[PAD]", "<pad>"}


def align_tokens(
    tokens: Sequence[str],
    scores: Sequence[float],
    tokenizer_type: str = "wordpiece",
    aggregation: AggregationMethod = "mean",
) -> Tuple[List[str], List[float]]:
    """
    Align subword tokens into full words and aggregate their scores.

    Supports:
        - WordPiece (##)
        - SentencePiece (▁)

    Adds:
        - NaN/inf safety
        - configurable aggregation
        - stable merging
    """

    # --------------------------------------------------
    # VALIDATION
    # --------------------------------------------------

    if not isinstance(tokens, Sequence) or not isinstance(scores, Sequence):
        raise TypeError("tokens and scores must be sequences")

    if len(tokens) != len(scores):
        raise ValueError("tokens and scores must match in length")

    if tokenizer_type not in {"wordpiece", "sentencepiece"}:
        raise ValueError("tokenizer_type must be 'wordpiece' or 'sentencepiece'")

    if aggregation not in {"mean", "sum", "max"}:
        raise ValueError("aggregation must be 'mean', 'sum', or 'max'")

    if len(tokens) == 0:
        return [], []

    # --------------------------------------------------
    # AGGREGATION FUNCTION
    # --------------------------------------------------

    def agg(values: List[float]) -> float:
        arr = np.array(values, dtype=np.float32)
        # Neutralize only non-finite values. Do NOT clip magnitude here:
        # SHAP / IG / LIME scores may legitimately be signed and exceed [0, 1].
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        if aggregation == "mean":
            return float(np.mean(arr))
        elif aggregation == "sum":
            return float(np.sum(arr))
        elif aggregation == "max":
            # Absolute-max preserves the largest-magnitude signal (and its sign).
            idx = int(np.argmax(np.abs(arr)))
            return float(arr[idx])

        return float(np.mean(arr))  # fallback

    # --------------------------------------------------
    # MAIN LOGIC
    # --------------------------------------------------

    merged_tokens: List[str] = []
    merged_scores: List[float] = []

    current_token_parts: List[str] = []
    current_scores: List[float] = []

    for token, score in zip(tokens, scores):

        if token is None:
            continue

        token = str(token).strip()

        if token in SPECIAL_TOKENS:
            continue

        if not token:
            continue

        try:
            score = float(score)
        except Exception:
            score = 0.0

        if not np.isfinite(score):
            score = 0.0

        # -------------------------------
        # WORDPIECE
        # -------------------------------
        if tokenizer_type == "wordpiece":

            if token.startswith("##"):
                piece = token[2:]
                if piece:
                    current_token_parts.append(piece)
                    current_scores.append(score)
                continue

            # new token
            if current_token_parts:
                merged_tokens.append("".join(current_token_parts))
                merged_scores.append(agg(current_scores))

            current_token_parts = [token]
            current_scores = [score]

        # -------------------------------
        # SENTENCEPIECE
        # -------------------------------
        else:

            if token.startswith("▁"):
                # flush previous
                if current_token_parts:
                    merged_tokens.append("".join(current_token_parts))
                    merged_scores.append(agg(current_scores))

                piece = token[1:]
                current_token_parts = [piece] if piece else []
                current_scores = [score] if piece else []

            else:
                if not current_token_parts:
                    current_token_parts = [token]
                    current_scores = [score]
                else:
                    current_token_parts.append(token)
                    current_scores.append(score)

    # --------------------------------------------------
    # FINAL FLUSH
    # --------------------------------------------------

    if current_token_parts:
        merged_tokens.append("".join(current_token_parts))
        merged_scores.append(agg(current_scores))

    # Final NaN/Inf safety only — preserve sign and magnitude.
    merged_scores = [
        float(s) if np.isfinite(s) else 0.0 for s in merged_scores
    ]

    return merged_tokens, merged_scores