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
import multiprocessing as mp
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.features.base.base_feature import FeatureContext
from src.features.pipelines.feature_pipeline import FeaturePipeline

logger = logging.getLogger(__name__)


def _worker_extract(args: tuple[FeaturePipeline, FeatureContext]) -> Dict[str, float]:
    """
    Worker function used for multiprocessing feature extraction.
    """
    pipeline, context = args
    return pipeline.extract(context)


@dataclass
class BatchFeaturePipeline:
    """
    High-throughput batch feature extraction system.
    """

    pipeline: FeaturePipeline
    num_workers: int = 1
    chunk_size: int = 32
    fail_fast: bool = True

    _initialized: bool = field(default=False, init=False)

    def initialize(self) -> None:
        """
        Initialize underlying feature pipeline.
        """

        if not self._initialized:
            self.pipeline.initialize()
            self._initialized = True

            logger.info(
                "BatchFeaturePipeline initialized | workers=%d",
                self.num_workers,
            )

    def _sequential_extract(
        self,
        contexts: List[FeatureContext],
    ) -> List[Dict[str, float]]:
        """
        Sequential feature extraction.
        """

        results: List[Dict[str, float]] = []

        for ctx in contexts:
            try:
                features = self.pipeline.extract(ctx)
                results.append(features)

            except Exception:  # noqa: BLE001
                logger.exception("Feature extraction failed")

                if self.fail_fast:
                    raise

                results.append({})

        return results

    def _parallel_extract(
        self,
        contexts: List[FeatureContext],
    ) -> List[Dict[str, float]]:
        """
        Parallel feature extraction using multiprocessing.
        """

        logger.info(
            "Starting parallel feature extraction | samples=%d workers=%d",
            len(contexts),
            self.num_workers,
        )

        tasks = [(self.pipeline, ctx) for ctx in contexts]

        with mp.Pool(self.num_workers) as pool:
            results = pool.map(
                _worker_extract,
                tasks,
                chunksize=self.chunk_size,
            )

        return results

    def extract(
        self,
        contexts: List[FeatureContext],
    ) -> List[Dict[str, float]]:
        """
        Extract features for a dataset.

        Parameters
        ----------
        contexts : List[FeatureContext]

        Returns
        -------
        List[Dict[str, float]]
        """

        if not contexts:
            raise ValueError("Input contexts cannot be empty")

        if not self._initialized:
            self.initialize()

        logger.info(
            "Batch feature extraction started | samples=%d",
            len(contexts),
        )

        if self.num_workers <= 1:
            results = self._sequential_extract(contexts)
        else:
            results = self._parallel_extract(contexts)

        logger.info(
            "Batch feature extraction completed | samples=%d",
            len(results),
        )

        return results

    def extract_with_labels(
        self,
        contexts: List[FeatureContext],
        labels: Optional[List[int]] = None,
        fit: bool = False,
    ) -> List[Dict[str, float]]:
        """
        Execute full pipeline including scaling and selection.
        """

        if fit:
            features = self.pipeline.process(contexts, labels=labels, fit=True)
        else:
            features = self.pipeline.process(contexts, labels=labels, fit=False)

        return features
