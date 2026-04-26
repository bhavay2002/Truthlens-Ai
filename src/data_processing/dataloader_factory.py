from __future__ import annotations

import logging
from typing import Dict, Any, Optional

from torch.utils.data import DataLoader

from src.data.data_contracts import get_contract
from src.data.collate import collate_fn
from src.data.samplers import build_sampler

logger = logging.getLogger(__name__)


# =========================================================
# CONFIG
# =========================================================

class DataLoaderConfig:
    def __init__(
        self,
        batch_size: int = 16,
        num_workers: int = 4,
        pin_memory: bool = True,
        use_sampler: bool = True,
        drop_last: bool = False,
    ):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.use_sampler = use_sampler
        self.drop_last = drop_last


# =========================================================
# SINGLE LOADER BUILDER
# =========================================================

def build_dataloader(
    *,
    task: str,
    dataset,
    df,  # needed for sampler
    split: str,
    config: DataLoaderConfig,
) -> DataLoader:
    """
    Build a DataLoader for a specific task + split.
    """

    contract = get_contract(task)

    # -----------------------------------------------------
    # SAMPLER (TRAIN ONLY)
    # -----------------------------------------------------

    sampler = None
    shuffle = False

    if split == "train" and config.use_sampler:
        sampler = build_sampler(task=task, df=df)
        shuffle = False
    else:
        sampler = None
        shuffle = split == "train"

    # -----------------------------------------------------
    # LOADER
    # -----------------------------------------------------

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=collate_fn,
        drop_last=config.drop_last if split == "train" else False,
    )

    logger.info(
        "DataLoader built | task=%s | split=%s | size=%d",
        task,
        split,
        len(dataset),
    )

    return loader


# =========================================================
# MULTI-TASK BUILDER
# =========================================================

def build_all_dataloaders(
    *,
    datasets: Dict[str, Dict[str, Any]],
    raw_dfs: Dict[str, Dict[str, Any]],
    config: Optional[DataLoaderConfig] = None,
) -> Dict[str, Dict[str, DataLoader]]:
    """
    Build dataloaders for all tasks and splits.

    Args:
        datasets:
            {
                "bias": {"train": Dataset, "val": Dataset, "test": Dataset},
                ...
            }

        raw_dfs:
            Needed for samplers
            {
                "bias": {"train": df, "val": df, "test": df},
                ...
            }

    Returns:
        Same structure but DataLoaders
    """

    config = config or DataLoaderConfig()

    loaders = {}

    for task, splits in datasets.items():

        loaders[task] = {}

        for split, dataset in splits.items():

            df = raw_dfs[task][split]

            loaders[task][split] = build_dataloader(
                task=task,
                dataset=dataset,
                df=df,
                split=split,
                config=config,
            )

    return loaders