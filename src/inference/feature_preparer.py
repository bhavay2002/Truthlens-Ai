from __future__ import annotations

import logging
import multiprocessing as mp
from multiprocessing.pool import Pool
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import atexit

import numpy as np
import torch

from src.graph.graph_pipeline import GraphPipeline
from src.features.base.base_feature import FeatureContext
from src.features.bias.bias_features import BiasFeatures
from src.features.bias.framing_features import FramingFeatures
from src.features.bias.ideological_features import IdeologicalFeatures
from src.features.feature_schema_validator import FeatureSchemaValidator
from src.features.feature_statistics import FeatureStatistics
from src.features.pipelines.feature_pipeline import ALL_BIAS_MODULE_FEATURE_NAMES

logger = logging.getLogger(__name__)


# =========================================================
# FAST WORKER
# =========================================================

def _prepare_flat_features_worker(features: Dict[str, Any]) -> Dict[str, float]:
    flat = {}

    for key, value in features.items():
        if key == "text":
            continue

        if isinstance(value, (int, float)):
            flat[key] = float(value)

        elif isinstance(value, (list, tuple, set)):
            flat[f"{key}_count"] = float(len(value))

        elif isinstance(value, dict):
            for k, v in value.items():
                if isinstance(v, (int, float)):
                    flat[f"{key}_{k}"] = float(v)

    return flat


# =========================================================
# CONFIG
# =========================================================

@dataclass
class FeaturePreparationConfig:
    feature_schema: List[str]
    apply_scaling: bool = True
    apply_feature_selection: bool = True
    return_tensor: bool = True
    dtype: str = "float32"
    derive_graph_features: bool = False  # 🔥 disabled by default for speed


# =========================================================
# MAIN CLASS
# =========================================================

class FeaturePreparer:

    def __init__(
        self,
        config: FeaturePreparationConfig,
        scaler: Optional[Any] = None,
        selector: Optional[Any] = None,
    ):

        self.config = config
        self.scaler = scaler
        self.selector = selector

        self.feature_dim = len(config.feature_schema)
        self.feature_index = {f: i for i, f in enumerate(config.feature_schema)}

        self.schema_validator = FeatureSchemaValidator(
            expected_features=config.feature_schema,
            strict=False,
            allow_missing=True,
            allow_extra=True,
        )

        self.graph_pipeline = GraphPipeline() if config.derive_graph_features else None
        self._pool: Optional[Pool] = None

        atexit.register(self.close_pool)

        logger.info(f"FeaturePreparer initialized | dim={self.feature_dim}")

    # =====================================================
    # MULTIPROCESS POOL
    # =====================================================

    def _get_pool(self):
        if self._pool is None:
            ctx = mp.get_context("spawn")
            self._pool = ctx.Pool(max(1, mp.cpu_count() - 1))
        return self._pool

    def close_pool(self):
        if self._pool:
            self._pool.close()
            self._pool.join()
            self._pool = None

    # =====================================================
    # CORE FLATTEN
    # =====================================================

    def _flatten(self, features: Dict[str, Any]) -> Dict[str, float]:

        flat = {}

        for key, value in features.items():

            if key == "text":
                continue

            if isinstance(value, (int, float)):
                flat[key] = float(value)

            elif isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(v, (int, float)):
                        flat[f"{key}_{k}"] = float(v)

        return flat

    # =====================================================
    # VECTORIZE
    # =====================================================

    def _to_vector(self, flat: Dict[str, float]):

        vec = np.zeros(self.feature_dim, dtype=np.float32)

        for k, v in flat.items():
            idx = self.feature_index.get(k)
            if idx is not None:
                vec[idx] = v

        return vec

    # =====================================================
    # TRANSFORMS
    # =====================================================

    def _transform(self, X):

        if self.scaler:
            X = self.scaler.transform(X)

        if self.selector:
            X = self.selector.transform(X)

        return X

    # =====================================================
    # SINGLE
    # =====================================================

    def prepare_single(self, features: Dict[str, Any]):

        flat = self._flatten(features)
        vec = self._to_vector(flat)[None, :]

        vec = self._transform(vec)

        if self.config.return_tensor:
            return torch.tensor(vec, dtype=torch.float32)

        return vec

    # =====================================================
    # BATCH
    # =====================================================

    def prepare_batch(self, feature_dicts: List[Dict[str, Any]]):

        if len(feature_dicts) < 32:
            flats = [self._flatten(f) for f in feature_dicts]
        else:
            pool = self._get_pool()
            flats = pool.map(_prepare_flat_features_worker, feature_dicts)

        X = np.zeros((len(flats), self.feature_dim), dtype=np.float32)

        for i, flat in enumerate(flats):
            for k, v in flat.items():
                idx = self.feature_index.get(k)
                if idx is not None:
                    X[i, idx] = v

        X = self._transform(X)

        if self.config.return_tensor:
            return torch.tensor(X, dtype=torch.float32)

        return X

    # =====================================================
    # DIRECT TEXT → FEATURES
    # =====================================================

    def prepare_from_text(self, text: str):

        ctx = FeatureContext(text=text)

        features = {"text": text}

        features.update(BiasFeatures().extract(ctx))
        features.update(FramingFeatures().extract(ctx))
        features.update(IdeologicalFeatures().extract(ctx))

        return self.prepare_single(features)

    # =====================================================
    # STATS
    # =====================================================

    def compute_feature_statistics(self, feature_dicts):

        flats = [self._flatten(f) for f in feature_dicts]

        stats = FeatureStatistics()

        return {
            "summary": stats.dataset_summary(flats),
            "constant_features": stats.detect_constant_features(flats),
        }

    # =====================================================
    # UTIL
    # =====================================================

    def get_feature_schema(self):
        return self.config.feature_schema

    def feature_dimension(self):
        return self.feature_dim