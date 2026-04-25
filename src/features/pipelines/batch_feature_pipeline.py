# src/features/pipelines/feature_batch_pipeline.py

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


# =========================================================
# DATASET
# =========================================================

class FeatureDataset(Dataset):
    def __init__(self, contexts: List[FeatureContext]):
        self.contexts = contexts

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, idx):
        return idx, self.contexts[idx]


def collate_fn(batch):
    indices, contexts = zip(*batch)
    return list(indices), list(contexts)


# =========================================================
# PIPELINE
# =========================================================

@dataclass
class BatchFeaturePipeline:

    pipeline: FeaturePipeline

    batch_size: int = 32
    num_workers: int = 2
    pin_memory: bool = True
    use_amp: bool = True

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    _initialized: bool = field(default=False, init=False)

    # 🔥 GLOBAL SHARED CACHE
    _shared_cache: Dict[str, Any] = field(default_factory=dict, init=False)

    # 🔥 GRAPH CACHE (NEW)
    _graph_cache: Dict[str, Any] = field(default_factory=dict, init=False)

    # -----------------------------------------------------

    def initialize(self) -> None:

        if self._initialized:
            return

        self.pipeline.initialize()

        # 🔥 Move model
        if hasattr(self.pipeline, "model"):
            try:
                self.pipeline.model.to(self.device)
            except Exception:
                logger.warning("Model device move failed")

            try:
                self.pipeline.model = torch.compile(self.pipeline.model)
            except Exception:
                pass

        self._initialized = True

        logger.info(
            "BatchFeaturePipeline initialized | batch_size=%d device=%s",
            self.batch_size,
            self.device,
        )

    # =====================================================
    # 🔥 EMBEDDING OPTIMIZATION (NEW)
    # =====================================================

    def _compute_embeddings(self, batch: List[FeatureContext]):

        if not hasattr(self.pipeline, "encoder") or not hasattr(self.pipeline, "tokenizer"):
            return

        texts = [ctx.text for ctx in batch]

        try:
            device = torch.device(self.device)

            with torch.no_grad():
                with torch.autocast(self.device, enabled=self.device == "cuda"):

                    inputs = self.pipeline.tokenizer(
                        texts,
                        padding=True,
                        truncation=True,
                        return_tensors="pt",
                    ).to(device)

                    outputs = self.pipeline.encoder(**inputs)

            embeddings = outputs.last_hidden_state

            # 🔥 store per context
            for i, ctx in enumerate(batch):
                ctx.cache["_shared_cache"]["embedding"] = embeddings[i]

        except Exception as e:
            logger.warning("Embedding computation failed: %s", e)

    # =====================================================
    # 🔥 GRAPH CACHE (NEW)
    # =====================================================

    def _attach_graph_cache(self, batch: List[FeatureContext]):

        if not self.pipeline.graph_pipeline:
            return

        for ctx in batch:

            text = ctx.text

            if text not in self._graph_cache:
                try:
                    self._graph_cache[text] = self.pipeline.graph_pipeline.run(text)
                except Exception as e:
                    logger.warning("Graph failed: %s", e)
                    self._graph_cache[text] = {}

            ctx.cache["graph_pipeline_output"] = self._graph_cache[text]

    # =====================================================
    # CORE EXECUTION
    # =====================================================

    def _run_batch_extract(self, batch: List[FeatureContext]):

        if hasattr(self.pipeline, "batch_extract"):
            return self.pipeline.batch_extract(batch)

        if hasattr(self.pipeline, "extract_batch"):
            return self.pipeline.extract_batch(batch)

        raise AttributeError("Pipeline missing batch API")

    # -----------------------------------------------------

    def _process_batch(self, batch: List[FeatureContext]):

        # 🔥 attach shared cache
        for ctx in batch:
            if not isinstance(ctx.cache, dict):
                ctx.cache = {}
            ctx.cache["_shared_cache"] = self._shared_cache

        # 🔥 NEW: embedding optimization
        self._compute_embeddings(batch)

        # 🔥 NEW: graph cache
        self._attach_graph_cache(batch)

        try:
            with torch.no_grad():

                if self.device == "cuda" and self.use_amp:

                    with torch.autocast(
                        device_type="cuda",
                        dtype=(
                            torch.bfloat16
                            if torch.cuda.is_bf16_supported()
                            else torch.float16
                        ),
                    ):
                        return self._run_batch_extract(batch)

                return self._run_batch_extract(batch)

        except Exception as e:
            logger.exception("Batch failed: %s", e)
            return [{} for _ in batch]

    # =====================================================
    # DATALOADER
    # =====================================================

    def _dataloader_extract(self, contexts: List[FeatureContext]):

        dataset = FeatureDataset(contexts)

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers if self.device == "cuda" else 0,
            pin_memory=self.pin_memory if self.device == "cuda" else False,
            collate_fn=collate_fn,
            shuffle=False,
        )

        results: List[Optional[Dict[str, float]]] = [None] * len(contexts)

        logger.info("Starting batch extraction | samples=%d", len(contexts))

        for indices, batch in loader:

            batch_features = self._process_batch(batch)

            for idx, feat in zip(indices, batch_features):
                results[idx] = feat

        if any(r is None for r in results):
            raise RuntimeError("Incomplete extraction")

        logger.info("Batch extraction completed")

        return results  # type: ignore

    # =====================================================
    # PUBLIC API
    # =====================================================

    def extract(self, contexts: List[FeatureContext]):

        if not contexts:
            raise ValueError("Empty input")

        if not self._initialized:
            self.initialize()

        return self._dataloader_extract(contexts)

    # -----------------------------------------------------

    def extract_by_section(self, contexts: List[FeatureContext]):

        flat = self.extract(contexts)

        return [partition_feature_sections(f) for f in flat]

    # -----------------------------------------------------

    def extract_with_labels(
        self,
        contexts: List[FeatureContext],
        labels: Optional[List[int]] = None,
        fit: bool = False,
    ):

        return self.pipeline.process(contexts, labels=labels, fit=fit)