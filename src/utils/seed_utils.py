"""
File: seed_utils.py

Purpose
-------
Utilities for setting random seeds across libraries to ensure
reproducibility of machine learning experiments in TruthLens AI.
"""

from __future__ import annotations

import logging
import random
import numpy as np
import torch


# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Set Seed
# ---------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility across Python, NumPy, and PyTorch.

    Parameters
    ----------
    seed : int
        Random seed value.
    """

    try:
        random.seed(seed)

        np.random.seed(seed)

        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # Ensure deterministic behavior for CuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        logger.info("Random seed set to %d", seed)

    except Exception:
        logger.exception("Failed to set random seed")
        raise