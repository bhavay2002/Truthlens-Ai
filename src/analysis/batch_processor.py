# src/analysis/batch_processor.py

from __future__ import annotations

import logging
from typing import Iterable, List, Dict, Any, Generator, Optional

from src.analysis.pipeline import AnalysisPipeline

logger = logging.getLogger(__name__)


# =========================================================
# Batch Processor
# =========================================================

class BatchProcessor:
    """
    High-performance batch processor for text analysis.

    Key Features:
    - Uses spaCy nlp.pipe() under the hood (via pipeline)
    - Dynamic batching
    - Memory-safe iteration
    - Optional streaming output
    """

    def __init__(
        self,
        pipeline: AnalysisPipeline,
        batch_size: int = 32,
        max_length: int = 100_000,
        drop_empty: bool = True,
    ):
        self.pipeline = pipeline
        self.batch_size = batch_size
        self.max_length = max_length
        self.drop_empty = drop_empty

        logger.info(
            "BatchProcessor initialized | batch_size=%d max_length=%d",
            batch_size,
            max_length,
        )

    # -----------------------------------------------------

    def process(
        self,
        texts: Iterable[str],
        *,
        return_generator: bool = False,
    ) -> List[Dict[str, Any]] | Generator[Dict[str, Any], None, None]:
        """
        Process texts in batches.

        Args:
            texts: iterable of input strings
            return_generator: if True, yields results lazily

        Returns:
            list or generator of results
        """

        if return_generator:
            return self._process_generator(texts)

        return list(self._process_generator(texts))

    # -----------------------------------------------------

    def _process_generator(
        self,
        texts: Iterable[str],
    ) -> Generator[Dict[str, Any], None, None]:

        batch: List[str] = []

        for text in texts:

            # -----------------------------
            # Input validation
            # -----------------------------

            if not isinstance(text, str):
                logger.warning("Skipping non-string input")
                continue

            text = text.strip()

            if self.drop_empty and not text:
                continue

            if len(text) > self.max_length:
                logger.warning("Text too long, truncating")
                text = text[: self.max_length]

            batch.append(text)

            # -----------------------------
            # Batch full → process
            # -----------------------------

            if len(batch) >= self.batch_size:
                yield from self._run_batch(batch)
                batch.clear()

        # -----------------------------
        # Final batch
        # -----------------------------

        if batch:
            yield from self._run_batch(batch)

    # -----------------------------------------------------

    def _run_batch(
        self,
        batch: List[str],
    ) -> Generator[Dict[str, Any], None, None]:

        try:
            results = self.pipeline.run_batch(batch)

            for result in results:
                yield result

        except Exception as e:
            logger.exception("Batch processing failed")

            # fallback: process individually
            for text in batch:
                try:
                    yield self.pipeline.run(text)
                except Exception:
                    logger.exception("Failed single text processing")
                    yield self._empty_result()

    # -----------------------------------------------------

    def _empty_result(self) -> Dict[str, Any]:
        return {
            "features": {},
            "profile": {},
            "propaganda": {},
            "meta": {"error": True},
        }