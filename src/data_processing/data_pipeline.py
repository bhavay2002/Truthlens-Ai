from __future__ import annotations

import logging
from typing import Dict, Any, Optional

import pandas as pd

# =========================
# CORE DATA SYSTEM
# =========================

from src.data.data_contracts import get_contract
from src.data.data_resolver import resolve_data_config
from src.data.data_loader import load_dataframe
from src.data.data_validator import validate_dataframe
from src.data.data_cleaning import clean_for_task
from src.data.data_augmentation import augment_dataset, AugmentationConfig
from src.data.data_profiler import profile_dataframe
from src.data.leakage_checker import check_leakage_all_tasks

from src.data.dataset_factory import build_all_datasets
from src.data.dataloader_factory import build_all_dataloaders, DataLoaderConfig

# =========================
# 🆕 ANALYSIS LAYER
# =========================

from src.analysis.label_analysis import analyze_labels, assert_label_health
from src.analysis.multitask_validator import (
    validate_multitask_dataframe,
    assert_multitask_health,
)

# =========================
# 🆕 CACHE SYSTEM
# =========================

from src.data.data_cache import (
    get_cache_key,
    load_cached_datasets,
    save_cached_datasets,
)

logger = logging.getLogger(__name__)


# =========================================================
# CONFIG
# =========================================================

class DataPipelineConfig:

    def __init__(
        self,
        enable_cleaning: bool = True,
        enable_validation: bool = True,
        enable_augmentation: bool = False,
        enable_profiling: bool = True,
        enable_leakage_check: bool = True,

        enable_multitask_validation: bool = True,
        enable_label_analysis: bool = True,

        # 🆕 CACHE CONTROL
        enable_cache: bool = True,
        force_rebuild: bool = False,

        augmentation_config: Optional[AugmentationConfig] = None,
        dataloader_config: Optional[DataLoaderConfig] = None,
    ):
        self.enable_cleaning = enable_cleaning
        self.enable_validation = enable_validation
        self.enable_augmentation = enable_augmentation
        self.enable_profiling = enable_profiling
        self.enable_leakage_check = enable_leakage_check

        self.enable_multitask_validation = enable_multitask_validation
        self.enable_label_analysis = enable_label_analysis

        self.enable_cache = enable_cache
        self.force_rebuild = force_rebuild

        self.augmentation_config = augmentation_config or AugmentationConfig()
        self.dataloader_config = dataloader_config or DataLoaderConfig()


# =========================================================
# CORE PIPELINE
# =========================================================

def run_data_pipeline(
    *,
    data_config: Dict[str, Dict[str, str]],
    tokenizer=None,
    build_dataloaders: bool = False,
    config: Optional[DataPipelineConfig] = None,
):

    config = config or DataPipelineConfig()

    # =====================================================
    # 🆕 CACHE CHECK
    # =====================================================

    cache_key = get_cache_key(data_config)

    if config.enable_cache and not config.force_rebuild:

        cached = load_cached_datasets(cache_key)

        if cached is not None:
            logger.info("✅ Using cached dataset")

            raw_datasets = cached

            if not build_dataloaders:
                return raw_datasets

            if tokenizer is None:
                raise ValueError("Tokenizer required for dataloaders")

            datasets = build_all_datasets(
                datasets=raw_datasets,
                tokenizer=tokenizer,
            )

            return build_all_dataloaders(
                datasets=datasets,
                raw_dfs=raw_datasets,
                config=config.dataloader_config,
            )

    logger.info("⚡ Building dataset from scratch")

    # =====================================================
    # 1. RESOLVE PATHS
    # =====================================================

    resolved_paths = resolve_data_config(data_config)

    # =====================================================
    # 2. LOAD + PROCESS
    # =====================================================

    raw_datasets: Dict[str, Dict[str, pd.DataFrame]] = {}

    for task in resolved_paths:

        logger.info("Processing task: %s", task)

        contract = get_contract(task)

        raw_datasets[task] = {}

        task_columns = {
            task: contract.label_columns[0]
            if contract.task_type == "classification"
            else contract.label_columns
        }

        for split, path in resolved_paths[task].items():

            df = load_dataframe(path)

            # -------------------------
            # VALIDATION
            # -------------------------
            if config.enable_validation:
                validate_dataframe(df, task=task)

            # -------------------------
            # CLEANING
            # -------------------------
            if config.enable_cleaning:
                df = clean_for_task(df, task)

            # -------------------------
            # MULTITASK VALIDATION
            # -------------------------
            if config.enable_multitask_validation:
                df, mt_result = validate_multitask_dataframe(
                    df,
                    task_columns=task_columns,
                )
                assert_multitask_health(mt_result)

            # -------------------------
            # LABEL ANALYSIS
            # -------------------------
            if config.enable_label_analysis:
                label_result = analyze_labels(
                    df,
                    task_columns=task_columns,
                )
                assert_label_health(
                    label_result,
                    fail_on_imbalance=False,
                    fail_on_rare=False,
                )

            # -------------------------
            # AUGMENTATION
            # -------------------------
            if config.enable_augmentation and split == "train":
                df = augment_dataset(
                    df,
                    task=task,
                    config=config.augmentation_config,
                )

            raw_datasets[task][split] = df

    # =====================================================
    # 🆕 SAVE CACHE
    # =====================================================

    if config.enable_cache:
        save_cached_datasets(raw_datasets, cache_key)
        logger.info("💾 Dataset cached")

    # =====================================================
    # LEAKAGE CHECK
    # =====================================================

    if config.enable_leakage_check:
        check_leakage_all_tasks(raw_datasets)

    # =====================================================
    # PROFILING
    # =====================================================

    if config.enable_profiling:
        for task in raw_datasets:
            profile_dataframe(
                raw_datasets[task]["train"],
                task=task,
            )

    # =====================================================
    # RETURN RAW
    # =====================================================

    if not build_dataloaders:
        return raw_datasets

    if tokenizer is None:
        raise ValueError("Tokenizer required")

    datasets = build_all_datasets(
        datasets=raw_datasets,
        tokenizer=tokenizer,
    )

    return build_all_dataloaders(
        datasets=datasets,
        raw_dfs=raw_datasets,
        config=config.dataloader_config,
    )