"""
File Name: batch_feature_pipeline.py
Module: Feature Engineering - Batch Feature Pipeline
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import torch
from torch.utils.data import Dataset, DataLoader

from src.features.base.base_feature import FeatureContext
from src.features.pipelines.feature_pipeline import (
    FeaturePipeline,
    partition_feature_sections,
)

logger = logging.getLogger(__name__)


class FeatureDataset(Dataset):
    def __init__(self, contexts: List[FeatureContext]):
        self.contexts = contexts

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, idx):
        return self.contexts[idx]


def collate_fn(batch: List[FeatureContext]) -> List[FeatureContext]:
    batch.sort(key=lambda x: len(x.text) if isinstance(x.text, str) else 0)
    return batch


@dataclass
class BatchFeaturePipeline:
    pipeline: FeaturePipeline
    batch_size: int = 32
    num_workers: int = 2
    pin_memory: bool = True
    use_amp: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    _initialized: bool = field(default=False, init=False)

    def initialize(self) -> None:
        if self._initialized:
            return

        self.pipeline.initialize()

        if hasattr(self.pipeline, "model"):
            try:
                self.pipeline.model = torch.compile(self.pipeline.model)
                logger.info("Model compiled with torch.compile")
            except Exception:  # noqa: BLE001
                logger.warning("torch.compile failed, skipping")

        self._initialized = True
        logger.info(
            "BatchFeaturePipeline initialized | batch_size=%d workers=%d device=%s",
            self.batch_size,
            self.num_workers,
            self.device,
        )

    def _run_batch_extract(self, batch: List[FeatureContext]) -> List[Dict[str, float]]:
        if hasattr(self.pipeline, "batch_extract"):
            return self.pipeline.batch_extract(batch)
        if hasattr(self.pipeline, "extract_batch"):
            return self.pipeline.extract_batch(batch)
        raise AttributeError("FeaturePipeline must expose batch_extract() or extract_batch().")

    def _process_batch(self, batch: List[FeatureContext]) -> List[Dict[str, float]]:
        shared_cache: Dict[str, Any] = {}

        for ctx in batch:
            if not isinstance(ctx.cache, dict):
                ctx.cache = {}
            ctx.cache.setdefault("_shared_cache", shared_cache)

        with torch.no_grad():
            if self.device == "cuda" and self.use_amp:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    features = self._run_batch_extract(batch)
            else:
                features = self._run_batch_extract(batch)

        return features

    def _dataloader_extract(self, contexts: List[FeatureContext]) -> List[Dict[str, float]]:
        dataset = FeatureDataset(contexts)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
            shuffle=False,
        )

        results: List[Dict[str, float]] = []
        logger.info("Starting optimized batch extraction | samples=%d", len(contexts))

        for batch in loader:
            results.extend(self._process_batch(batch))

        logger.info("Batch extraction completed | samples=%d", len(results))
        return results

    def extract(self, contexts: List[FeatureContext]) -> List[Dict[str, float]]:
        if not contexts:
            raise ValueError("Input contexts cannot be empty")
        if not all(isinstance(c, FeatureContext) for c in contexts):
            raise TypeError("contexts must be a list of FeatureContext")

        if not self._initialized:
            self.initialize()

        return self._dataloader_extract(contexts)

    def extract_by_section(self, contexts: List[FeatureContext]) -> List[Dict[str, Dict[str, float]]]:
        flat_results = self.extract(contexts)
        return [partition_feature_sections(f) for f in flat_results]

    def extract_with_labels(
        self,
        contexts: List[FeatureContext],
        labels: Optional[List[int]] = None,
        fit: bool = False,
    ) -> List[Dict[str, float]]:
        if fit:
            return self.pipeline.process(contexts, labels=labels, fit=True)
        return self.pipeline.process(contexts, labels=labels, fit=False)