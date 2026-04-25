from __future__ import annotations

import logging
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
import torch

logger = logging.getLogger(__name__)


# =========================================================
# CONFIG
# =========================================================

@dataclass
class InferenceConfig:
    """
    Runtime configuration for inference.
    """

    # -------------------------
    # CORE
    # -------------------------
    device: str = "cpu"              # cpu | cuda | auto
    batch_size: int = 32
    max_sequence_length: int = 512

    # -------------------------
    # PIPELINE CONTROL
    # -------------------------
    use_graph_analysis: bool = True
    enable_entity_graph: bool = True
    enable_narrative_analysis: bool = True
    enable_explainability: bool = True

    # -------------------------
    # PREDICTION OUTPUT (🔥 IMPORTANT)
    # -------------------------
    return_logits: bool = True
    return_probabilities: bool = True

    # -------------------------
    # CACHE
    # -------------------------
    cache_predictions: bool = False
    cache_dir: str = "cache"
    cache_ttl_seconds: Optional[int] = None

    # -------------------------
    # SAFETY
    # -------------------------
    prediction_timeout: Optional[int] = None

    # -------------------------
    # REPRODUCIBILITY
    # -------------------------
    config_version: str = "v2"


# =========================================================
# LOADER
# =========================================================

class InferenceConfigLoader:

    REQUIRED_FIELDS = {
        "device": str,
        "batch_size": int,
    }

    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path)

        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")

        logger.info("InferenceConfigLoader initialized")

    # =====================================================
    # LOAD
    # =====================================================

    def load(self) -> InferenceConfig:

        with open(self.config_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f) or {}

        if not isinstance(config_dict, dict):
            raise TypeError("Config must be a dictionary")

        self._validate_config(config_dict)

        allowed = {f.name for f in fields(InferenceConfig)}

        filtered = {k: v for k, v in config_dict.items() if k in allowed}

        unknown = sorted(set(config_dict.keys()) - allowed)
        if unknown:
            logger.warning("Unknown config keys ignored: %s", unknown)

        config = InferenceConfig(**filtered)

        # 🔥 RESOLVE DEVICE
        config.device = self._resolve_device(config.device)

        logger.info("Inference config loaded (device=%s)", config.device)

        return config

    # =====================================================
    # VALIDATION
    # =====================================================

    def _validate_config(self, config: Dict[str, Any]):

        for field, expected_type in self.REQUIRED_FIELDS.items():

            if field not in config:
                continue

            if not isinstance(config[field], expected_type):
                raise TypeError(
                    f"{field} must be {expected_type.__name__}"
                )

        # SAFE ACCESS
        batch_size = config.get("batch_size")
        if batch_size is not None and batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        device = config.get("device")
        if device is not None and device not in {"cpu", "cuda", "auto"}:
            raise ValueError("device must be cpu | cuda | auto")

    # =====================================================
    # DEVICE RESOLUTION
    # =====================================================

    def _resolve_device(self, device: str) -> str:

        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"

        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available → falling back to CPU")
            return "cpu"

        return device

    # =====================================================
    # FROM DICT
    # =====================================================

    @staticmethod
    def from_dict(config: Dict[str, Any]) -> InferenceConfig:

        if not isinstance(config, dict):
            raise TypeError("config must be dict")

        return InferenceConfig(**config)


# =========================================================
# HELPER
# =========================================================

def load_inference_config(path: str | Path) -> InferenceConfig:
    return InferenceConfigLoader(path).load()