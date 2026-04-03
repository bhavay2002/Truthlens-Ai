"""Utility loader for EmotionClassifier checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from ..tasks.emotion.emotion_classifier import EmotionClassifier, EmotionClassifierConfig


class EmotionModelLoader:
    """Load an emotion classifier from config/state dict artifacts."""

    @staticmethod
    def load(
        model_path: str | Path,
        config: Optional[EmotionClassifierConfig] = None,
        device: Optional[str] = None,
    ) -> EmotionClassifier:
        model = EmotionClassifier(config or EmotionClassifierConfig())

        path_obj = Path(model_path)
        if path_obj.exists():
            state_dict = torch.load(path_obj, map_location=device or "cpu")
            model.load_state_dict(state_dict)

        if device is not None:
            model = model.to(device)

        model.eval()
        return model


__all__ = ["EmotionModelLoader"]
