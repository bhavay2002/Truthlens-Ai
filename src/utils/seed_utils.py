"""
File Name: seed_utils.py
Module: src.utils
Description:
    Utilities for controlling randomness and ensuring reproducibility
    across machine learning experiments in TruthLens AI.

    This module sets deterministic seeds for Python, NumPy, and PyTorch
    and configures backend behavior to minimize nondeterminism during
    model training and inference.

Author: TruthLens Engineering
Date: 2026-04-03
Dependencies:
    - Python 3.10+
    - numpy
    - torch

Inputs:
    - Seed integer

Outputs:
    - Deterministic random state across supported libraries
"""

from __future__ import annotations

import logging
import os
import random
from typing import Optional

import numpy as np
import torch


# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Set Seed
# ---------------------------------------------------------


def set_seed(seed: int = 42, *, deterministic: bool = True) -> None:
    """
    Set random seed for reproducibility across Python, NumPy, and PyTorch.

    Parameters
    ----------
    seed : int
        Random seed value.

    deterministic : bool
        Whether to enforce deterministic computation for CUDA/CuDNN.
    """

    try:
        if not isinstance(seed, int):
            raise TypeError("seed must be an integer")

        # Python random
        random.seed(seed)

        # Python hash seed
        os.environ["PYTHONHASHSEED"] = str(seed)

        # NumPy
        np.random.seed(seed)

        # PyTorch CPU
        torch.manual_seed(seed)

        # PyTorch CUDA
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        if deterministic:
            _configure_deterministic_backend()

        logger.info("Random seed set to %d", seed)

    except Exception as exc:
        logger.exception("Failed to set random seed")
        raise RuntimeError("Random seed initialization failed") from exc


# ---------------------------------------------------------
# Deterministic Backend Configuration
# ---------------------------------------------------------


def _configure_deterministic_backend() -> None:
    """
    Configure PyTorch backends for deterministic behavior.
    """

    try:
        # CuDNN settings
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Torch deterministic algorithms (PyTorch >=1.8)
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            # Older PyTorch versions may not support this
            pass

        logger.debug("PyTorch deterministic backend enabled")

    except Exception as exc:
        logger.exception("Failed to configure deterministic backend")
        raise RuntimeError("Deterministic backend configuration failed") from exc


# ---------------------------------------------------------
# Seed Worker (DataLoader Support)
# ---------------------------------------------------------


def seed_worker(worker_id: int) -> None:
    """
    Initialize worker-specific seed for PyTorch DataLoader.

    Ensures reproducibility in multi-worker dataloaders.

    Parameters
    ----------
    worker_id : int
        Worker process ID.
    """

    try:
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

        logger.debug("Initialized DataLoader worker seed: %d", worker_seed)

    except Exception as exc:
        logger.exception("Failed to initialize DataLoader worker seed")
        raise RuntimeError("Worker seed initialization failed") from exc


# ---------------------------------------------------------
# Get Current Seed State
# ---------------------------------------------------------


def get_seed_state() -> dict[str, Optional[int]]:
    """
    Retrieve current seed-related state for debugging or logging.

    Returns
    -------
    dict[str, Optional[int]]
        Dictionary containing seed information.
    """

    try:
        state = {
            "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
            "torch_seed": torch.initial_seed(),
            "cuda_available": torch.cuda.is_available(),
        }

        if torch.cuda.is_available():
            state["cuda_seed"] = torch.cuda.initial_seed()

        return state

    except Exception as exc:
        logger.exception("Failed to retrieve seed state")
        raise RuntimeError("Seed state retrieval failed") from exc