"""
Utility loader for EmotionClassifier checkpoints.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import torch

from ..tasks.emotion.emotion_classifier import (
    EmotionClassifier,
    EmotionClassifierConfig,
)

logger = logging.getLogger(__name__)


class EmotionModelLoader:
    """
    Production-grade loader for EmotionClassifier checkpoints.

    Supports:
    - raw state_dict checkpoints
    - training checkpoints with metadata
    - automatic device placement
    """

    # -----------------------------------------------------

    @staticmethod
    def _resolve_device(device: Optional[str]) -> torch.device:
        """
        Resolve compute device automatically.
        """

        if device is not None:
            return torch.device(device)

        if torch.cuda.is_available():
            return torch.device("cuda")

        return torch.device("cpu")

    # -----------------------------------------------------

    @staticmethod
    def _extract_state_dict(checkpoint: dict) -> dict:
        """
        Extract state_dict from different checkpoint formats.
        """

        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]

        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]

        return checkpoint

    # -----------------------------------------------------

    @staticmethod
    def load(
        model_path: Union[str, Path],
        config: Optional[EmotionClassifierConfig] = None,
        device: Optional[str] = None,
    ) -> EmotionClassifier:
        """
        Load EmotionClassifier from checkpoint.

        Parameters
        ----------
        model_path : str | Path
            Path to checkpoint file.

        config : EmotionClassifierConfig
            Optional configuration override.

        device : str
            Device placement ("cpu", "cuda").

        Returns
        -------
        EmotionClassifier
        """

        device_obj = EmotionModelLoader._resolve_device(device)

        model = EmotionClassifier(config or EmotionClassifierConfig())

        path_obj = Path(model_path)

        if not path_obj.exists():
            raise FileNotFoundError(f"Emotion model checkpoint not found: {model_path}")

        logger.info("Loading emotion model checkpoint: %s", path_obj)

        checkpoint = torch.load(path_obj, map_location=device_obj, weights_only=False)

        state_dict = EmotionModelLoader._extract_state_dict(checkpoint)

        _lr = model.load_state_dict(state_dict, strict=False)
        if _lr.missing_keys:
            raise RuntimeError(
                f"[CHECKPOINT ERROR] Missing keys in emotion model: {_lr.missing_keys}"
            )
        if _lr.unexpected_keys:
            raise RuntimeError(
                f"[CHECKPOINT ERROR] Unexpected keys in emotion model: {_lr.unexpected_keys}"
            )

        model = model.to(device_obj)

        model.eval()

        logger.info(
            "Emotion model loaded successfully with full parameter match on device: %s",
            device_obj,
        )

        return model


__all__ = ["EmotionModelLoader"]