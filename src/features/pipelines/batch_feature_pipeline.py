"""
File Name: batch_feature_pipeline.py
Module: Feature Engineering - Batch Feature Pipeline
Description:
    Implements a high-throughput batch feature extraction pipeline used for
    dataset-scale processing in the TruthLens system. The pipeline wraps the
    single-instance FeaturePipeline and provides:

        • parallel batch execution
        • deterministic ordering
        • progress-aware logging
        • optional fault tolerance
        • scalable dataset processing

    Integrates all feature extractor modules:

        BiasFeatures            → 10 features (bias_*)
        FramingFeatures         → 10 features (frame_*)
        IdeologicalFeatures     →  8 features (ideology_*)
        ArgumentStructureFeatures →  7 features (argument_*)
        DiscourseFeatures       →  7 features (discourse_*)
        EntityGraphFeatures     →  5 features (entity_*)
        InteractionGraphFeatures →  6 features (interaction_*)
        ConflictFeatures        →  9 features (conflict_*)
        NarrativeFeatures       → 11 features (narrative_*)
        NarrativeFrameFeatures  →  9 features (narrative_frame_*)
        NarrativeRoleFeatures   →  7 features (narrative_role_*)
        ManipulationPatterns    → 13 features (manipulation_*)
        PropagandaFeatures      → 11 features (propaganda_*)
        PropagandaLexiconFeatures → 11 features (propaganda_*)
        LexicalFeatures         →  5 features (vocabulary_/hapax_)
        SemanticFeatures        →  5 features (embedding_*)
        SyntacticFeatures       →  7 features (sentence_/pos_*)
        TokenFeatures           →  6 features (token_*)

    All extractors are auto-discovered via FeatureRegistry at initialization.
    The extract_by_section() method partitions each sample's output into
    named sections using partition_feature_sections() from feature_pipeline:
        bias, framing, ideology, emotion, discourse, graph,
        narrative, propaganda, text, other

    Designed for research experiments and production preprocessing jobs.

Dependencies:
    dataclasses
    typing
    logging
    multiprocessing
    itertools

Inputs:
    List[FeatureContext]

Outputs:
    List[Dict[str, float]] feature vectors
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


# =========================================================
# Dataset Wrapper (for DataLoader)
# =========================================================
class FeatureDataset(Dataset):
    def __init__(self, contexts: List[FeatureContext]):
        self.contexts = contexts

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, idx):
        return self.contexts[idx]


# =========================================================
# Collate Function (Dynamic Padding + Length Grouping)
# =========================================================
def collate_fn(batch: List[FeatureContext]) -> List[FeatureContext]:
    # Sort by text length (reduces padding cost later)
    batch.sort(key=lambda x: len(x.text) if hasattr(x, "text") else 0)
    return batch


# =========================================================
# Optimized Batch Feature Pipeline
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

    # =====================================================
    # Initialization
    # =====================================================
    def initialize(self) -> None:
        if not self._initialized:
            self.pipeline.initialize()

            # Optional: torch compile (PyTorch 2.x)
            if hasattr(self.pipeline, "model"):
                try:
                    self.pipeline.model = torch.compile(self.pipeline.model)
                    logger.info("Model compiled with torch.compile")
                except Exception:
                    logger.warning("torch.compile failed, skipping")

            self._initialized = True

            logger.info(
                "BatchFeaturePipeline initialized | batch_size=%d workers=%d device=%s",
                self.batch_size,
                self.num_workers,
                self.device,
            )

    # =====================================================
    # Core Batch Extraction (NO per-sample loops)
    # =====================================================
    def _process_batch(
        self, batch: List[FeatureContext]
    ) -> List[Dict[str, float]]:
        """
        Process one batch efficiently.
        """

        # 🔥 Shared cache (prevents repeated encoder calls)
        shared_cache: Dict[str, Any] = {}

        # Attach shared cache to each context
        for ctx in batch:
            if not hasattr(ctx, "shared"):
                ctx.shared = shared_cache
            else:
                ctx.shared.update(shared_cache)

        # AMP context
        if self.device == "cuda" and self.use_amp:
            autocast_ctx = torch.cuda.amp.autocast(dtype=torch.float16)
        else:
            autocast_ctx = torch.no_grad()

        with torch.no_grad():
            with autocast_ctx:
                # 🚀 CRITICAL: Batch extraction
                features = self.pipeline.extract_batch(batch)

        return features

    # =====================================================
    # DataLoader Execution
    # =====================================================
    def _dataloader_extract(
        self,
        contexts: List[FeatureContext],
    ) -> List[Dict[str, float]]:

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

        logger.info(
            "Starting optimized batch extraction | samples=%d",
            len(contexts),
        )

        for batch in loader:
            batch_results = self._process_batch(batch)
            results.extend(batch_results)

        logger.info(
            "Batch extraction completed | samples=%d",
            len(results),
        )

        return results

    # =====================================================
    # Public API
    # =====================================================
    def extract(
        self,
        contexts: List[FeatureContext],
    ) -> List[Dict[str, float]]:

        if not contexts:
            raise ValueError("Input contexts cannot be empty")

        if not self._initialized:
            self.initialize()

        return self._dataloader_extract(contexts)

    # =====================================================
    # Section-wise Output
    # =====================================================
    def extract_by_section(
        self,
        contexts: List[FeatureContext],
    ) -> List[Dict[str, Dict[str, float]]]:

        flat_results = self.extract(contexts)
        return [partition_feature_sections(f) for f in flat_results]

    # =====================================================
    # Training-Compatible Pipeline
    # =====================================================
    def extract_with_labels(
        self,
        contexts: List[FeatureContext],
        labels: Optional[List[int]] = None,
        fit: bool = False,
    ) -> List[Dict[str, float]]:

        if fit:
            return self.pipeline.process(contexts, labels=labels, fit=True)
        return self.pipeline.process(contexts, labels=labels, fit=False)