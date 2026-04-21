import torch

def validate_checkpoint(state_dict):
    for k, v in state_dict.items():
        if not torch.isfinite(v).all():
            raise ValueError(f"Non-finite values in {k}")

    required_prefixes = [
        "encoder",
        "bias_head",
        "ideology_head",
        "propaganda_head",
        "narrative_head",
        "emotion_head"
    ]

    missing = [
        p for p in required_prefixes
        if not any(k.startswith(p) for k in state_dict)
    ]

    if missing:
        raise ValueError(f"Missing required components: {missing}")