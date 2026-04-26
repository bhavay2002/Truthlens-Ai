"""
DataLoader factory.

Optimisations vs the original:
- ``pin_memory`` is gated on CUDA availability.
- ``persistent_workers`` and ``prefetch_factor`` are exposed (fewer worker
  respawns, better pipelining).
- ``num_workers`` defaults to ``min(4, cpu_count // 2)``.
- Collate function is built with the tokenizer's ``pad_token_id`` so
  RoBERTa-family models pad correctly.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, Any, Optional

import torch
from torch.utils.data import DataLoader

from src.data_processing.data_contracts import get_contract  # noqa: F401  (kept for back-compat callers)
from src.data_processing.collate import build_collate_fn, collate_fn as _legacy_collate
from src.data_processing.samplers import build_sampler

logger = logging.getLogger(__name__)


# =========================================================
# CONFIG
# =========================================================

def _default_num_workers() -> int:
    cpu = os.cpu_count() or 1
    return max(0, min(4, cpu // 2))


@dataclass
class DataLoaderConfig:
    batch_size: int = 16
    num_workers: int = -1               # -1 → auto
    pin_memory: bool = True             # gated on CUDA at build time
    use_sampler: bool = True
    drop_last: bool = False
    persistent_workers: bool = True
    prefetch_factor: int = 4
    safety_check_collate: bool = True

    def resolved_num_workers(self) -> int:
        return _default_num_workers() if self.num_workers < 0 else self.num_workers

    def resolved_pin_memory(self) -> bool:
        return bool(self.pin_memory and torch.cuda.is_available())


# =========================================================
# SINGLE LOADER
# =========================================================

def build_dataloader(
    *,
    task: str,
    dataset,
    df,
    split: str,
    config: DataLoaderConfig,
    tokenizer: Any = None,
) -> DataLoader:
    """Build a DataLoader for one (task, split)."""
    sampler = None
    shuffle = False

    if split == "train" and config.use_sampler:
        sampler = build_sampler(task=task, df=df)
    elif split == "train":
        shuffle = True

    # collate with correct pad_token_id
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer is not None and tokenizer.pad_token_id is not None
        else getattr(dataset, "pad_token_id", 0)
    )
    collate = build_collate_fn(
        pad_token_id=pad_id,
        safety_check=config.safety_check_collate,
    )

    num_workers = config.resolved_num_workers()
    pin_memory = config.resolved_pin_memory()

    loader_kwargs: Dict[str, Any] = dict(
        batch_size=config.batch_size,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate,
        drop_last=config.drop_last if split == "train" else False,
    )

    if num_workers > 0:
        loader_kwargs["persistent_workers"] = config.persistent_workers
        loader_kwargs["prefetch_factor"] = config.prefetch_factor

    loader = DataLoader(dataset, **loader_kwargs)

    logger.info(
        "DataLoader | task=%s | split=%s | size=%d | workers=%d | pin=%s | pad_id=%d",
        task, split, len(dataset), num_workers, pin_memory, pad_id,
    )
    return loader


# =========================================================
# MULTI-TASK
# =========================================================

def build_all_dataloaders(
    *,
    datasets: Dict[str, Dict[str, Any]],
    raw_dfs: Dict[str, Dict[str, Any]],
    config: Optional[DataLoaderConfig] = None,
    tokenizer: Any = None,
) -> Dict[str, Dict[str, DataLoader]]:
    """Build dataloaders for every (task, split)."""
    config = config or DataLoaderConfig()
    loaders: Dict[str, Dict[str, DataLoader]] = {}

    for task, splits in datasets.items():
        loaders[task] = {}
        for split, ds in splits.items():
            loaders[task][split] = build_dataloader(
                task=task,
                dataset=ds,
                df=raw_dfs[task][split],
                split=split,
                config=config,
                tokenizer=tokenizer,
            )

    return loaders


__all__ = [
    "DataLoaderConfig",
    "build_dataloader",
    "build_all_dataloaders",
]
