from __future__ import annotations

from typing import Any

import torch


def extract_logits(output: Any) -> torch.Tensor:
    """
    Normalize model outputs to logits tensor.

    Supports:
    - torch.Tensor
    - dict with "logits"
    - tuple (first element assumed logits)
    """
    if isinstance(output, torch.Tensor):
        return output

    if isinstance(output, dict):
        logits = output.get("logits")
        if isinstance(logits, torch.Tensor):
            return logits
        raise TypeError("Dictionary output must contain tensor 'logits'.")

    if isinstance(output, tuple):
        return extract_logits(output[0])

    raise TypeError("Unsupported model output type.")
