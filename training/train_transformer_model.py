"""
training/train_transformer_model.py

Root-level compatibility shim.

Re-exports everything from src.training.train_transformer_model
so that tests and scripts can import from either location.
"""

from src.training.train_transformer_model import *  # noqa: F401, F403
from src.training.train_transformer_model import (
    _compute_checkpoint_save_steps,
    _split_train_val_test,
    _validate_split_df,
    tokenize_function,
)
