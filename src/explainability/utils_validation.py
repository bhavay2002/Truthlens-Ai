def validate_tokens_scores(tokens, scores):
    if not tokens or not scores:
        raise ValueError("Empty tokens or scores")
    if len(tokens) != len(scores):
        raise ValueError("tokens and scores must match length")
