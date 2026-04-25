from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except Exception:
    PANDAS_AVAILABLE = False

from src.features.base.base_feature import FeatureContext
from src.features.pipelines.batch_feature_pipeline import BatchFeaturePipeline
from src.features.pipelines.feature_pipeline import (
    partition_feature_sections,
    BIAS_FEATURE_NAMES,
    FRAMING_FEATURE_NAMES,
    IDEOLOGICAL_FEATURE_NAMES,
)
from src.features.cache.cache_manager import CacheManager

logger = logging.getLogger(__name__)

FeatureVector = Dict[str, float]


@dataclass
class DatasetFeatureGenerator:

    pipeline: BatchFeaturePipeline
    cache_manager: Optional[CacheManager] = None
    cache_namespace: str = "dataset_features"

    _feature_order: List[str] = field(default_factory=list, init=False)

    # =====================================================
    # CONTEXT BUILD
    # =====================================================

    def _build_contexts(self, texts: List[str]) -> List[FeatureContext]:

        contexts = []

        for text in texts:
            if not text or not isinstance(text, str):
                raise ValueError("Input texts must be non-empty strings")

            contexts.append(FeatureContext(text=text))

        return contexts

    # =====================================================
    # CACHE-AWARE EXTRACTION
    # =====================================================

    def _cached_extract(self, contexts: List[FeatureContext]) -> List[FeatureVector]:

        if self.cache_manager is None:
            return self.pipeline.extract(contexts)

        cache = self.cache_manager.get_cache(self.cache_namespace)

        results: List[Optional[FeatureVector]] = [None] * len(contexts)

        uncached_contexts = []
        uncached_indices = []

        for i, ctx in enumerate(contexts):

            key = self.cache_manager._context_key(ctx)
            cached = cache.load(key)

            if cached is not None:
                results[i] = cached
            else:
                uncached_contexts.append(ctx)
                uncached_indices.append(i)

        if uncached_contexts:

            new_features = self.pipeline.extract(uncached_contexts)

            for idx, feat, ctx in zip(uncached_indices, new_features, uncached_contexts):
                key = self.cache_manager._context_key(ctx)
                cache.save(key, feat)
                results[idx] = feat

        if any(r is None for r in results):
            raise RuntimeError("Incomplete feature extraction (cache mismatch)")

        return [r for r in results if r is not None]

    # =====================================================
    # MATRIX GENERATION
    # =====================================================

    def generate(
        self,
        texts: List[str],
        labels: Optional[List[int]] = None,
        fit: bool = False,
    ) -> Tuple[np.ndarray, List[str]]:

        if not texts:
            raise ValueError("Input text list cannot be empty")

        contexts = self._build_contexts(texts)

        logger.info("Generating dataset features | samples=%d", len(contexts))

        # ---------------------------
        # Extraction
        # ---------------------------

        if self.cache_manager:
            features = self._cached_extract(contexts)
            pipeline = self.pipeline.pipeline

            if pipeline.scaler is not None:
                if fit:
                    pipeline.fit_scaler(features)
                features = pipeline.transform_scaler(features)

            if pipeline.selector is not None:
                if fit:
                    pipeline.fit_selector(features, labels)
                features = pipeline.transform_selector(features)

        else:
            features = self.pipeline.extract_with_labels(
                contexts,
                labels=labels,
                fit=fit,
            )

        if not features:
            raise RuntimeError("Empty feature output")

        # ---------------------------
        #  FIX: UNION OF ALL KEYS
        # ---------------------------

        all_keys = set()
        for f in features:
            all_keys.update(f.keys())

        feature_names = sorted(all_keys)

        name_to_idx = {name: j for j, name in enumerate(feature_names)}

        # ---------------------------
        # Matrix build (optimized)
        # ---------------------------

        n_samples = len(features)
        n_features = len(feature_names)

        matrix = np.zeros((n_samples, n_features), dtype=np.float32)

        for i, feat in enumerate(features):
            row = matrix[i]
            for key, value in feat.items():
                j = name_to_idx.get(key)
                if j is not None:
                    row[j] = float(value)

        self._feature_order = feature_names

        logger.info(
            "Feature matrix ready | shape=%s",
            matrix.shape,
        )

        return matrix, feature_names

    # =====================================================
    # SECTION SPLIT
    # =====================================================

    def generate_by_section(
        self,
        texts: List[str],
        sections: Optional[List[str]] = None,
    ) -> Dict[str, Tuple[np.ndarray, List[str]]]:

        if not texts:
            raise ValueError("Input text list cannot be empty")

        contexts = self._build_contexts(texts)

        if not self.pipeline._initialized:
            self.pipeline.initialize()

        raw_features = self.pipeline.extract(contexts)

        partitioned: Dict[str, List[FeatureVector]] = {}

        for sample in raw_features:
            sec_map = partition_feature_sections(sample)

            for sec_name, sec_features in sec_map.items():
                partitioned.setdefault(sec_name, []).append(sec_features)

        result: Dict[str, Tuple[np.ndarray, List[str]]] = {}

        for sec_name, samples in partitioned.items():

            if sections and sec_name not in sections:
                continue

            if not samples or not samples[0]:
                continue

            # union keys (safe)
            keys = set()
            for s in samples:
                keys.update(s.keys())

            feature_names = sorted(keys)
            name_to_idx = {n: i for i, n in enumerate(feature_names)}

            matrix = np.zeros((len(samples), len(feature_names)), dtype=np.float32)

            for i, sample in enumerate(samples):
                row = matrix[i]
                for k, v in sample.items():
                    j = name_to_idx.get(k)
                    if j is not None:
                        row[j] = float(v)

            result[sec_name] = (matrix, feature_names)

            logger.info("Section %s → shape=%s", sec_name, matrix.shape)

        return result

    # =====================================================
    # DATAFRAME
    # =====================================================

    def generate_dataframe(
        self,
        texts: List[str],
        labels: Optional[List[int]] = None,
        fit: bool = False,
    ):

        if not PANDAS_AVAILABLE:
            raise RuntimeError("pandas not installed")

        matrix, names = self.generate(texts, labels, fit)

        df = pd.DataFrame(matrix, columns=names)

        if labels is not None:
            df["label"] = labels

        return df

    # =====================================================
    # UTILITIES
    # =====================================================

    def get_feature_order(self) -> List[str]:

        if not self._feature_order:
            raise RuntimeError("Call generate() first")

        return self._feature_order

    def get_bias_module_feature_names(self) -> Dict[str, List[str]]:

        return {
            "bias": BIAS_FEATURE_NAMES,
            "framing": FRAMING_FEATURE_NAMES,
            "ideology": IDEOLOGICAL_FEATURE_NAMES,
        }