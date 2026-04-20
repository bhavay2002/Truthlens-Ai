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

import numpy as np
import torch

logger = logging.getLogger(__name__)


# =========================================================
# Main Seed Function
# =========================================================

def set_seed(
    seed: int = 42,
    *,
    deterministic: bool = False,
    enable_tf32: bool = True,
    matmul_precision: str = "high",
) -> None:
    """
    Set global seed + configure backend for performance or determinism.

    Parameters
    ----------
    seed : int
        Random seed

    deterministic : bool
        True  -> reproducible but slower
        False -> faster (recommended for training)

    enable_tf32 : bool
        Enable TensorFloat-32 (Ampere+ GPUs)

    matmul_precision : str
        "high" | "medium" | "highest"
    """

    if not isinstance(seed, int):
        raise TypeError("seed must be int")

    if matmul_precision not in {"high", "medium", "highest"}:
        raise ValueError("matmul_precision must be one of: 'high', 'medium', 'highest'")

    # -----------------------------
    # Core Seeding
    # -----------------------------

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # -----------------------------
    # Backend Config
    # -----------------------------

    if deterministic:
        _set_deterministic()
    else:
        _set_fast_mode(enable_tf32, matmul_precision)

    logger.info(
        "Seed set to %d | deterministic=%s",
        seed,
        deterministic
    )


# =========================================================
# Deterministic Mode
# =========================================================

def _set_deterministic():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Recommended for stronger CUDA determinism
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

    logger.debug("Deterministic mode enabled")


# =========================================================
# Fast Mode (IMPORTANT)
# =========================================================

def _set_fast_mode(enable_tf32: bool, matmul_precision: str):

    # cuDNN auto-tuner
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    # TF32 (huge speedup on Ampere+)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = enable_tf32
        torch.backends.cudnn.allow_tf32 = enable_tf32

    # Matmul precision (PyTorch 2+)
    try:
        torch.set_float32_matmul_precision(matmul_precision)
    except Exception:
        pass

    logger.debug(
        "Fast mode enabled | TF32=%s | matmul_precision=%s",
        enable_tf32,
        matmul_precision,
    )


# =========================================================
# DataLoader Worker Seed
# =========================================================

def seed_worker(worker_id: int):

    worker_seed = torch.initial_seed() % 2**32

    np.random.seed(worker_seed)
    random.seed(worker_seed)

    # Optional: torch seed for worker
    torch.manual_seed(worker_seed)

    logger.debug("Worker seed initialized: %d", worker_seed)


# =========================================================
# Generator (IMPORTANT for DataLoader)
# =========================================================

def create_generator(seed: int) -> torch.Generator:
    """
    Create torch Generator for reproducible DataLoader.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# =========================================================
# Seed State Debugging
# =========================================================

def get_seed_state() -> dict[str, Optional[int]]:

    state = {
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "torch_seed": torch.initial_seed(),
        "cuda_available": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        state["cuda_seed"] = torch.cuda.initial_seed()

    return state