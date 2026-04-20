import numpy as np


def align_tokens(tokens, scores, tokenizer_type="wordpiece"):
    if len(tokens) != len(scores):
        raise ValueError("tokens and scores must match")

    merged_tokens = []
    merged_scores = []

    current_token = ""
    current_scores = []

    for token, score in zip(tokens, scores):
        if tokenizer_type == "wordpiece":
            if token.startswith("##"):
                piece = token[2:]
                if not current_token:
                    current_token = piece
                else:
                    current_token += piece
                current_scores.append(score)
            else:
                if current_token:
                    merged_tokens.append(current_token)
                    merged_scores.append(float(np.mean(current_scores)))
                current_token = token
                current_scores = [score]

        elif tokenizer_type == "sentencepiece":
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
