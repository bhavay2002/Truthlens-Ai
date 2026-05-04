from __future__ import annotations

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import logging
from pathlib import Path
from typing import Any, List, Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict

from src.inference.predict_api import predict, predict_batch
from src.analysis.argument_mining import ArgumentMiningAnalyzer
from src.analysis.bias_profile_builder import BiasProfileBuilder
from src.graph.graph_pipeline import GraphPipeline, get_default_pipeline
from src.graph.graph_embeddings import GraphEmbeddingGenerator
from src.graph.temporal_graph import TemporalGraphAnalyzer
from src.analysis.context_omission_detector import ContextOmissionDetector
from src.analysis.discourse_coherence_analyzer import DiscourseCoherenceAnalyzer
from src.analysis.emotion_target_analysis import EmotionTargetAnalyzer
from src.analysis.framing_analysis import FramingAnalyzer
from src.analysis.ideological_language_detector import IdeologicalLanguageDetector
from src.analysis.information_density_analyzer import InformationDensityAnalyzer
from src.analysis.information_omission_detector import InformationOmissionDetector
from src.analysis.narrative_conflict import NarrativeConflictAnalyzer
from src.analysis.narrative_propagation import NarrativePropagationAnalyzer
from src.analysis.narrative_role_extractor import NarrativeRoleExtractor
from src.analysis.narrative_temporal_analyzer import NarrativeTemporalAnalyzer
from src.analysis.propaganda_pattern_detector import PropagandaPatternDetector
from src.analysis.rhetorical_device_detector import RhetoricalDeviceDetector
from src.analysis.source_attribution_analyzer import SourceAttributionAnalyzer
from src.features.bias.bias_lexicon import compute_bias_features
from src.features.emotion.emotion_lexicon import EmotionLexiconAnalyzer
from src.explainability.emotion_explainer import explain_emotion
from src.explainability.lime_explainer import explain_prediction
from src.explainability.explainability_pipeline import (
    run_explainability_pipeline,
    ExplainabilityConfig,
)
from src.aggregation.aggregation_pipeline import AggregationPipeline
from src.utils import ensure_non_empty_text, ensure_non_empty_text_list, get_device
from src.utils.logging_utils import configure_logging
from src.utils.settings import load_settings

from huggingface_hub import hf_hub_download
from src.inference.postprocessing import Postprocessor
from src.inference.inference_cache import InferenceCache, InferenceCacheConfig
from src.inference.inference_logger import InferenceLogger
from src.inference.result_formatter import ResultFormatter
from src.inference.report_generator import ReportGenerator, ReportConfig
from src.inference.inference_engine import (
    InferenceEngine,
    InferenceConfig as EngineConfig,
)
from src.models.calibration.calibration_metrics import (
    CalibrationMetrics,
    CalibrationMetricConfig,
)
from src.models.calibration.temperature_scaling import (
    TemperatureScaler,
    TemperatureScalingConfig,
)
from src.models.calibration.isotonic_calibration import (
    IsotonicCalibrator,
    IsotonicCalibrationConfig,
)
from src.models.ensemble.ensemble_model import EnsembleConfig
from src.models.ensemble.weighted_ensemble import WeightedEnsembleConfig
from src.models.export.onnx_export import ONNXExporter, ONNXExportConfig
from src.models.export.torchscript_export import (
    TorchScriptExporter,
    TorchScriptExportConfig,
)
from src.models.export.quantization import QuantizationEngine, QuantizationConfig

configure_logging()
logger = logging.getLogger(__name__)
SETTINGS = load_settings()
MODEL_PATH = SETTINGS.model.path
VECTORIZER_PATH = SETTINGS.paths.tfidf_vectorizer_path
TRAINING_TEXT_COLUMN = SETTINGS.training.text_column
APP_TITLE = SETTINGS.api.title
APP_DESCRIPTION = SETTINGS.api.description
APP_VERSION = SETTINGS.api.version
TEXT_PREVIEW_CHARS = max(int(SETTINGS.api.text_preview_chars), 1)
INFERENCE_ALLOW_RAW_TEXT_FALLBACK = bool(SETTINGS.inference.allow_raw_text_fallback)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

HF_REPO_V1 = "bhavaygupta2002/truthlens_v1"
HF_CHECKPOINT_FILE = "checkpoint.pt"
HF_REPO_V2 = "bhavaygupta2002/truthlens2"
MAX_LENGTH = 512

MODEL_SUBPACKAGES = (
    "emotion", "encoder", "ideology", "multitask", "narrative", "propaganda",
)
LIME_NUM_SAMPLES = 16

EMOTION_ANALYZER = EmotionLexiconAnalyzer()
ARGUMENT_ANALYZER = ArgumentMiningAnalyzer()
BIAS_PROFILE_BUILDER = BiasProfileBuilder()
CONTEXT_OMISSION_DETECTOR = ContextOmissionDetector()
DISCOURSE_ANALYZER = DiscourseCoherenceAnalyzer()
EMOTION_TARGET_ANALYZER = EmotionTargetAnalyzer()
FRAMING_ANALYZER = FramingAnalyzer()
IDEOLOGICAL_DETECTOR = IdeologicalLanguageDetector()
INFO_DENSITY_ANALYZER = InformationDensityAnalyzer()
INFO_OMISSION_DETECTOR = InformationOmissionDetector()
NARRATIVE_CONFLICT_ANALYZER = NarrativeConflictAnalyzer()
NARRATIVE_PROPAGATION_ANALYZER = NarrativePropagationAnalyzer()
NARRATIVE_ROLE_EXTRACTOR = NarrativeRoleExtractor()
NARRATIVE_TEMPORAL_ANALYZER = NarrativeTemporalAnalyzer()
PROPAGANDA_PATTERN_DETECTOR = PropagandaPatternDetector()
RHETORICAL_DETECTOR = RhetoricalDeviceDetector()
SOURCE_ATTRIBUTION_ANALYZER = SourceAttributionAnalyzer()
GRAPH_PIPELINE = get_default_pipeline()
GRAPH_EMBEDDING_GENERATOR = GraphEmbeddingGenerator()
TEMPORAL_GRAPH_ANALYZER = TemporalGraphAnalyzer()

INFERENCE_CACHE = InferenceCache(
    InferenceCacheConfig(
        cache_dir="cache/inference",
        enable_disk_cache=False,
        enable_memory_cache=True,
        ttl_seconds=3600,
    )
)
INFERENCE_LOGGER = InferenceLogger(service_name="truthlens-api", enable_json_logs=True)
RESULT_FORMATTER = ResultFormatter()
REPORT_GENERATOR = ReportGenerator(
    ReportConfig(include_timestamp=True, pretty_json=False, validate_fields=True)
)
AGGREGATION_PIPELINE = AggregationPipeline()

_INFERENCE_ENGINE: Optional[InferenceEngine] = None


# ---------------------------------------------------------------------------
# Multitask model adapter (truthlens_v1/checkpoint.pt)
# ---------------------------------------------------------------------------

class _ForwardResult:
    __slots__ = ("logits",)

    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits


class _MultiTaskModelWrapper(torch.nn.Module):
    """Makes MultiTaskTruthLensModel compatible with InferenceEngine._forward().

    InferenceEngine._forward() calls model(**encoded).logits, but the multitask
    model returns a dict.  This wrapper picks the best 2-class task head
    (propaganda → bias → first available) and returns a _ForwardResult with a
    .logits attribute so the rest of the engine pipeline works unchanged.
    """

    _PREFERRED_TASKS = ("propaganda", "bias")

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self._inner = model
        task_keys = list(getattr(model, "task_heads", {}).keys())
        self._primary = next(
            (t for t in self._PREFERRED_TASKS if t in task_keys),
            task_keys[0] if task_keys else None,
        )
        logger.info("_MultiTaskModelWrapper: primary task head = %s", self._primary)

    def forward(self, **inputs: Any) -> _ForwardResult:
        outputs = self._inner(**inputs)
        if self._primary and self._primary in outputs:
            logits = outputs[self._primary]["logits"]
        else:
            task_logits = outputs.get("task_logits", {})
            logits = next(iter(task_logits.values()))
        return _ForwardResult(logits=logits)


def _patch_checkpoint_model(model: torch.nn.Module, device: torch.device) -> None:
    """Back-fill instance attributes missing after unpickling from an older
    transformers version, and reset any cached CUDA device references."""
    try:
        from transformers.models.roberta.modeling_roberta import (
            RobertaAttention,
            RobertaSelfAttention,
        )
        _hf_config = None
        for _m in model.modules():
            if hasattr(_m, "config") and hasattr(
                getattr(_m, "config", None), "_attn_implementation"
            ):
                _hf_config = _m.config
                break

        for _m in model.modules():
            if hasattr(_m, "_device"):
                _m._device = device
            if hasattr(_m, "_cached_device"):
                _m._cached_device = device

            if isinstance(_m, RobertaAttention):
                if not hasattr(_m, "is_cross_attention"):
                    _m.is_cross_attention = False
                if not hasattr(_m, "pruned_heads"):
                    _m.pruned_heads = set()

            if isinstance(_m, RobertaSelfAttention):
                if not hasattr(_m, "is_cross_attention"):
                    _m.is_cross_attention = False
                if not hasattr(_m, "pruned_heads"):
                    _m.pruned_heads = set()
                if not hasattr(_m, "position_embedding_type"):
                    _m.position_embedding_type = "absolute"
                if not hasattr(_m, "config") and _hf_config is not None:
                    _m.config = _hf_config
                if not hasattr(_m, "scaling"):
                    head_size = getattr(_m, "attention_head_size", 64)
                    _m.scaling = head_size ** -0.5
                if not hasattr(_m, "is_decoder"):
                    _m.is_decoder = False
                if not hasattr(_m, "is_causal"):
                    _m.is_causal = False
                if not hasattr(_m, "layer_idx"):
                    _m.layer_idx = None

    except ImportError:
        for _m in model.modules():
            if hasattr(_m, "_device"):
                _m._device = device
            if hasattr(_m, "_cached_device"):
                _m._cached_device = device


def _build_engine_from_hf_checkpoint() -> Optional[InferenceEngine]:
    """Download checkpoint.pt from bhavaygupta2002/truthlens_v1 and wire up
    an InferenceEngine using the multitask model's propaganda/bias head."""
    try:
        logger.info("Downloading checkpoint from HuggingFace: %s/%s", HF_REPO_V1, HF_CHECKPOINT_FILE)
        ckpt_path = hf_hub_download(HF_REPO_V1, HF_CHECKPOINT_FILE)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        model = ckpt["model"]
        encoder = ckpt.get("encoder", SETTINGS.model.encoder)

        device = get_device(prefer_gpu=True)
        model.to(device)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        _patch_checkpoint_model(model, device)

        tokenizer = AutoTokenizer.from_pretrained(encoder, use_fast=True)

        amp_dtype = torch.float32
        if device.type == "cuda":
            requested = (os.environ.get("TRUTHLENS_AMP_DTYPE") or "float16").lower()
            if requested in ("bf16", "bfloat16"):
                amp_dtype = torch.bfloat16
            elif requested in ("fp16", "float16", "half"):
                amp_dtype = torch.float16

        cfg = EngineConfig(
            model_path=ckpt_path,
            max_length=SETTINGS.model.max_length,
            device=str(device),
        )

        engine = InferenceEngine.__new__(InferenceEngine)
        engine.config = cfg
        engine.device = device
        engine.temperature_scaler = None
        engine.isotonic_calibrator = None
        engine.postprocessor = Postprocessor()
        engine.use_amp = device.type == "cuda" and cfg.use_amp
        engine.amp_dtype = amp_dtype
        engine.model = _MultiTaskModelWrapper(model)
        engine.tokenizer = tokenizer
        engine.label_map = {0: "real", 1: "fake"}
        engine.prediction_service = None

        logger.warning(
            "InferenceEngine: no calibrator attached — "
            "'calibrated_probabilities' will equal raw softmax probabilities."
        )

        if cfg.enable_full_pipeline:
            from src.inference.prediction_service import PredictionService
            engine.prediction_service = PredictionService(engine=engine)

        logger.info(
            "InferenceEngine initialised from HuggingFace checkpoint "
            "(encoder=%s, device=%s)", encoder, device,
        )
        return engine
    except Exception as exc:
        logger.warning("Could not build InferenceEngine from HuggingFace: %s", exc)
        return None


def _get_inference_engine() -> Optional[InferenceEngine]:
    """Return the shared InferenceEngine.

    Priority:
      1. Already-initialised singleton.
      2. Local saved_models directory (trained artefact).
      3. HuggingFace checkpoint — bhavaygupta2002/truthlens_v1/checkpoint.pt.
    """
    global _INFERENCE_ENGINE
    if _INFERENCE_ENGINE is not None:
        return _INFERENCE_ENGINE

    if MODEL_PATH.exists():
        try:
            _INFERENCE_ENGINE = InferenceEngine(
                EngineConfig(
                    model_path=str(MODEL_PATH),
                    tokenizer_path=str(MODEL_PATH),
                    max_length=SETTINGS.model.max_length,
                    device=SETTINGS.inference.device,
                )
            )
            logger.info("InferenceEngine initialised from %s", MODEL_PATH)
        except Exception as exc:
            logger.warning("InferenceEngine could not load local model: %s", exc)

    if _INFERENCE_ENGINE is None:
        _INFERENCE_ENGINE = _build_engine_from_hf_checkpoint()

    return _INFERENCE_ENGINE


# ---------------------------------------------------------------------------
# TruthLens2 model (bhavaygupta2002/truthlens2) — simple REAL/FAKE classifier
# ---------------------------------------------------------------------------

_v2_model: Optional[AutoModelForSequenceClassification] = None
_v2_tokenizer: Optional[AutoTokenizer] = None
_v2_idx_to_label: Optional[dict[int, str]] = None
_v2_device: Optional[torch.device] = None


def _build_idx_to_label(model) -> dict[int, str]:
    idx_to_label: dict[int, str] = {}
    id2label = getattr(model.config, "id2label", None) or {}
    for idx, label in id2label.items():
        idx_to_label[int(idx)] = str(label).strip().upper()
    if idx_to_label:
        return idx_to_label
    label2id = getattr(model.config, "label2id", None) or {}
    for label, idx in label2id.items():
        idx_to_label[int(idx)] = str(label).strip().upper()
    if not idx_to_label:
        idx_to_label = {0: "REAL", 1: "FAKE"}
    return idx_to_label


def _get_label_index(idx_to_label: dict[int, str], target: str) -> Optional[int]:
    target = target.strip().upper()
    for idx, label in idx_to_label.items():
        if label == target:
            return idx
    return None


def _load_v2_model():
    global _v2_model, _v2_tokenizer, _v2_idx_to_label, _v2_device
    if _v2_model is not None:
        return _v2_model, _v2_tokenizer, _v2_idx_to_label, _v2_device

    logger.info("Loading TruthLens2 model from HuggingFace: %s", HF_REPO_V2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(HF_REPO_V2)
    model = AutoModelForSequenceClassification.from_pretrained(HF_REPO_V2)
    model.to(device)
    model.eval()
    idx_to_label = _build_idx_to_label(model)
    logger.info("TruthLens2 loaded on %s | labels: %s", device, idx_to_label)
    _v2_model = model
    _v2_tokenizer = tokenizer
    _v2_idx_to_label = idx_to_label
    _v2_device = device
    return _v2_model, _v2_tokenizer, _v2_idx_to_label, _v2_device


def _v2_predict_single(text: str) -> dict[str, Any]:
    model, tokenizer, idx_to_label, device = _load_v2_model()
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LENGTH
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=1)[0]
    pred_idx = int(torch.argmax(probs).item())
    pred_label = idx_to_label.get(pred_idx, f"CLASS_{pred_idx}")
    real_idx = _get_label_index(idx_to_label, "REAL")
    fake_idx = _get_label_index(idx_to_label, "FAKE")
    fake_prob = float(probs[fake_idx].item()) if fake_idx is not None else 0.0
    real_prob = float(probs[real_idx].item()) if real_idx is not None else 0.0
    confidence = float(probs[pred_idx].item())
    class_probabilities = {
        idx_to_label[i]: round(float(probs[i].item()), 6)
        for i in sorted(idx_to_label.keys())
    }
    return {
        "prediction": pred_label,
        "fake_probability": round(fake_prob, 6),
        "real_probability": round(real_prob, 6),
        "confidence": round(confidence, 6),
        "class_probabilities": class_probabilities,
    }


def _v2_predict_batch(texts: list[str]) -> list[dict[str, Any]]:
    model, tokenizer, idx_to_label, device = _load_v2_model()
    inputs = tokenizer(
        texts, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LENGTH
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    all_probs = F.softmax(outputs.logits, dim=1)
    real_idx = _get_label_index(idx_to_label, "REAL")
    fake_idx = _get_label_index(idx_to_label, "FAKE")
    results = []
    for probs in all_probs:
        pred_idx = int(torch.argmax(probs).item())
        pred_label = idx_to_label.get(pred_idx, f"CLASS_{pred_idx}")
        fake_prob = float(probs[fake_idx].item()) if fake_idx is not None else 0.0
        real_prob = float(probs[real_idx].item()) if real_idx is not None else 0.0
        confidence = float(probs[pred_idx].item())
        class_probabilities = {
            idx_to_label[j]: round(float(probs[j].item()), 6)
            for j in sorted(idx_to_label.keys())
        }
        results.append({
            "prediction": pred_label,
            "fake_probability": round(fake_prob, 6),
            "real_probability": round(real_prob, 6),
            "confidence": round(confidence, 6),
            "class_probabilities": class_probabilities,
        })
    return results


# ---------------------------------------------------------------------------
# Calibration singletons
# ---------------------------------------------------------------------------

CALIBRATION_METRICS = CalibrationMetrics()
ONNX_EXPORTER = ONNXExporter(ONNXExportConfig(verify_export=False))
TORCHSCRIPT_EXPORTER = TorchScriptExporter(TorchScriptExportConfig(verify_export=False))

_QUANTIZATION_ENGINE: Optional[QuantizationEngine] = None


def _get_quantization_engine(method: str = "dynamic") -> Optional[QuantizationEngine]:
    global _QUANTIZATION_ENGINE
    if _QUANTIZATION_ENGINE is None:
        try:
            _QUANTIZATION_ENGINE = QuantizationEngine(
                QuantizationConfig(method=method, backend="fbgemm")
            )
        except (ValueError, RuntimeError) as exc:
            logger.warning("QuantizationEngine not available: %s", exc)
    return _QUANTIZATION_ENGINE


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="TruthLens AI — Unified API",
    description=(
        "Unified misinformation detection and news credibility platform. "
        "Combines the deep-analysis pipeline (truthlens_v1) with the fast "
        "REAL/FAKE classifier (truthlens2). All models load automatically "
        "from HuggingFace — no local training required."
    ),
    version=APP_VERSION,
)

FEATURE_CACHE_MAX_AGE_DAYS = float(
    getattr(SETTINGS, "feature_cache_max_age_days", 0) or 14.0
)
FEATURE_CACHE_MAX_BYTES = int(
    getattr(SETTINGS, "feature_cache_max_bytes", 0) or (512 * 1024 * 1024)
)


@app.on_event("startup")
def _prune_feature_cache_on_startup() -> None:
    try:
        from src.features.cache.cache_manager import CacheManager
        cache_root = getattr(getattr(SETTINGS, "paths", None), "cache_dir", None)
        manager = CacheManager(base_cache_dir=Path(cache_root) if cache_root else None)
        results = manager.prune_all(
            max_bytes_per_namespace=FEATURE_CACHE_MAX_BYTES,
            max_age_days=FEATURE_CACHE_MAX_AGE_DAYS,
        )
        if results:
            total_removed = sum(
                int(r.get("removed_age", 0)) + int(r.get("removed_size", 0))
                for r in results.values()
            )
            logger.info(
                "Feature cache prune complete | namespaces=%d removed=%d",
                len(results), total_removed,
            )
    except Exception as exc:
        logger.warning("Feature cache prune skipped: %s", exc)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class NewsRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {"text": "Breaking news: Scientists discover new species in Amazon rainforest."}
        }
    )
    text: str = Field(..., min_length=10, max_length=10_000, description="News article text to analyze")


class BatchNewsRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "texts": ["First news article text here.", "Second news article text here."]
            }
        }
    )
    texts: list[str] = Field(
        ..., min_length=1, max_length=50,
        description="List of news article texts to analyze (max 50 items)",
    )


class NewsResponse(BaseModel):
    text: str
    fake_probability: float = Field(..., ge=0, le=1)
    prediction: str
    confidence: float


class BatchNewsResponse(BaseModel):
    results: list[NewsResponse]
    total: int
    cache_hits: int


class AnalysisResponse(BaseModel):
    text: str
    prediction: str
    fake_probability: float = Field(..., ge=0, le=1)
    confidence: float = Field(..., ge=0, le=1)
    bias: dict[str, Any]
    emotion: dict[str, Any]
    narrative: dict[str, Any]
    framing: dict[str, Any]
    rhetoric: dict[str, Any]
    discourse: dict[str, Any]
    propaganda_analysis: dict[str, Any]
    credibility_profile: dict[str, Any]
    graph_analysis: dict[str, Any]
    explainability: dict[str, Any]


class ReportResponse(BaseModel):
    article_summary: dict[str, Any]
    bias_analysis: dict[str, Any]
    emotion_analysis: dict[str, Any]
    narrative_structure: dict[str, Any]
    entity_graph: dict[str, Any]
    credibility_score: Optional[float]


class ModelInfoResponse(BaseModel):
    available: bool
    model_path: str
    device: Optional[str]
    num_parameters: Optional[int]
    num_trainable_parameters: Optional[int]
    label_map: Optional[dict[str, Any]]


class CalibrationMetricsRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "probabilities": [[0.8, 0.2], [0.3, 0.7], [0.6, 0.4]],
                "labels": [0, 1, 0],
                "n_bins": 15,
            }
        }
    )
    probabilities: List[List[float]] = Field(
        ..., description="Per-sample probability distributions [[p_real, p_fake], ...]",
    )
    labels: List[int] = Field(
        ..., description="Ground-truth class indices (0=real, 1=fake)",
    )
    n_bins: int = Field(default=15, ge=2, le=100)


class CalibrationMetricsResponse(BaseModel):
    ece: float
    mce: float
    brier_score: float
    nll: float
    n_samples: int
    n_bins: int


class EnsemblePredictRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model_probabilities": [[0.7, 0.3], [0.4, 0.6], [0.6, 0.4]],
                "weights": [0.5, 0.3, 0.2],
                "strategy": "weighted_average",
            }
        }
    )
    model_probabilities: List[List[float]] = Field(
        ..., description="Probability vectors [[p_real, p_fake], ...] from each model",
    )
    weights: Optional[List[float]] = Field(
        default=None, description="Per-model weights for weighted_average strategy",
    )
    strategy: str = Field(
        default="average",
        description="Combination strategy: average | weighted_average | majority_vote",
    )


class EnsemblePredictResponse(BaseModel):
    strategy: str
    ensemble_probabilities: List[float]
    prediction: str
    fake_probability: float
    confidence: float
    num_models: int


class ExportRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={"example": {"output_path": "exports/model.onnx"}}
    )
    output_path: str = Field(..., description="Destination file path for the exported model")


class ExportResponse(BaseModel):
    format: str
    output_path: str
    success: bool
    message: str


# TruthLens2 (v2) schemas

class V2PredictRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "Mixed messages from Trump leave more questions than answers over war's end"
            }
        }
    )
    text: str = Field(..., min_length=10, max_length=10_000, description="News article text to classify")


class V2PredictResponse(BaseModel):
    text_preview: str
    prediction: str = Field(..., description='"REAL" or "FAKE"')
    fake_probability: float = Field(..., ge=0, le=1)
    real_probability: float = Field(..., ge=0, le=1)
    confidence: float = Field(..., ge=0, le=1)
    class_probabilities: dict[str, float]


class V2BatchPredictRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "texts": [
                    "Scientists confirm climate change is accelerating based on new data.",
                    "Government hiding truth about vaccines, insider reveals shocking secret.",
                ]
            }
        }
    )
    texts: List[str] = Field(..., min_length=1, max_length=50, description="List of news texts to classify (max 50)")


class V2BatchPredictResponse(BaseModel):
    results: List[V2PredictResponse]
    total: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _preview_text(text: str) -> str:
    if len(text) <= TEXT_PREVIEW_CHARS:
        return text
    return text[:TEXT_PREVIEW_CHARS] + "..."


def _safe_run(fn, *args, **kwargs) -> dict:
    try:
        result = fn(*args, **kwargs)
        return result if isinstance(result, dict) else {}
    except Exception as exc:
        name = getattr(fn, "__qualname__", type(fn).__name__)
        logger.warning("Analysis step '%s' failed: %s", name, exc)
        return {}


def _serialize_graph_result(result: dict) -> dict:
    out = {}
    for k, v in result.items():
        if hasattr(v, "tolist"):
            out[k] = v.tolist()
        else:
            out[k] = v
    return out


def _build_project_view() -> dict[str, Any]:
    src_dir = PROJECT_ROOT / "src"
    model_dir = src_dir / "models"
    model_subpackages = {}
    for subpackage in MODEL_SUBPACKAGES:
        package_dir = model_dir / subpackage
        model_subpackages[subpackage] = {
            "directory_exists": package_dir.exists(),
            "package_init_exists": (package_dir / "__init__.py").exists(),
        }
    return {
        "project_root": str(PROJECT_ROOT),
        "api": {"title": APP_TITLE, "version": APP_VERSION, "description": APP_DESCRIPTION},
        "config": {
            "model_name": SETTINGS.model.name,
            "model_path": str(MODEL_PATH),
            "training_text_column": TRAINING_TEXT_COLUMN,
            "vectorizer_path": str(VECTORIZER_PATH),
        },
        "structure": {
            "src_exists": src_dir.exists(),
            "api_exists": (PROJECT_ROOT / "api").exists(),
            "config_exists": (PROJECT_ROOT / "config").exists(),
            "tests_exists": (PROJECT_ROOT / "tests").exists(),
            "models_package_init_exists": (model_dir / "__init__.py").exists(),
            "model_subpackages": model_subpackages,
        },
    }


def _decode_prediction_result(prediction_result) -> tuple[float, str, float]:
    if isinstance(prediction_result, dict):
        prob = float(prediction_result.get("fake_probability", 0.0))
        prediction = str(prediction_result.get("label", "Fake")).upper()
        confidence = float(prediction_result.get("confidence", max(prob, 1 - prob)))
    else:
        prob = float(prediction_result)
        prediction = "FAKE" if prob > 0.5 else "REAL"
        confidence = prob if prob > 0.5 else (1 - prob)
    return prob, prediction, confidence


def _heuristic_predict_fn(text: str) -> dict:
    try:
        bias = compute_bias_features(text)
        bias_score = float(getattr(bias, "bias_score", 0.0))
    except Exception:
        bias_score = 0.0
    try:
        emo = EMOTION_ANALYZER.analyze(text)
        emo_scores: dict = getattr(emo, "emotion_scores", {}) or {}
        emo_intensity = sum(emo_scores.values()) / len(emo_scores) if emo_scores else 0.0
    except Exception:
        emo_intensity = 0.0
    fake_prob = min(max(0.5 * bias_score + 0.3 * emo_intensity + 0.1, 0.05), 0.95)
    return {
        "fake_probability": round(fake_prob, 4),
        "label": "FAKE" if fake_prob > 0.5 else "REAL",
        "prediction": "FAKE" if fake_prob > 0.5 else "REAL",
        "confidence": round(max(fake_prob, 1.0 - fake_prob), 4),
        "source": "heuristic_fallback",
    }


# ---------------------------------------------------------------------------
# Routes — root + health
# ---------------------------------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "TruthLens AI — Unified API",
        "status": "online",
        "models": {
            "truthlens_v1": f"https://huggingface.co/{HF_REPO_V1}",
            "truthlens2": f"https://huggingface.co/{HF_REPO_V2}",
        },
        "endpoints": {
            "predict": "/predict",
            "batch_predict": "/batch-predict",
            "analyze": "/analyze",
            "explain": "/explain",
            "report": "/report",
            "v2_predict": "/v2/predict",
            "v2_batch_predict": "/v2/batch-predict",
            "v2_health": "/v2/health",
            "health": "/health",
            "inference_model_info": "/inference/model-info",
            "cache_clear": "/cache/clear",
            "calibration_info": "/calibration/info",
            "calibration_metrics": "/calibration/metrics",
            "ensemble_info": "/ensemble/info",
            "ensemble_predict": "/ensemble/predict",
            "export_info": "/export/info",
            "export_onnx": "/export/onnx",
            "export_torchscript": "/export/torchscript",
            "project_view": "/project-view",
            "docs": "/docs",
        },
    }


@app.get("/health")
def health_check():
    try:
        model_exists = MODEL_PATH.exists()
        vectorizer_required = TRAINING_TEXT_COLUMN == "engineered_text"
        vectorizer_exists = (not vectorizer_required) or VECTORIZER_PATH.exists()
        vectorizer_fallback_enabled = INFERENCE_ALLOW_RAW_TEXT_FALLBACK
        vectorizer_effective_ready = (
            vectorizer_exists if not vectorizer_required
            else (vectorizer_exists or vectorizer_fallback_enabled)
        )
        required_files = ["config.json", "tokenizer.json"]
        weight_files = ["model.safetensors", "pytorch_model.bin"]
        has_weight_file = any((MODEL_PATH / f).exists() for f in weight_files) if model_exists else False
        model_files_exist = (
            all((MODEL_PATH / f).exists() for f in required_files) and has_weight_file
            if model_exists else False
        )
        engine = _get_inference_engine()
        hf_engine_ready = engine is not None
        v2_ready = _v2_model is not None
        cache_size = len(INFERENCE_CACHE.memory_cache)
        return {
            "status": "healthy" if (hf_engine_ready or model_files_exist) else "degraded",
            "model_path": str(MODEL_PATH),
            "model_exists": model_exists,
            "model_files_complete": model_files_exist,
            "hf_engine_ready": hf_engine_ready,
            "hf_repo_v1": HF_REPO_V1,
            "hf_repo_v2": HF_REPO_V2,
            "v2_model_loaded": v2_ready,
            "training_text_column": TRAINING_TEXT_COLUMN,
            "vectorizer_required": vectorizer_required,
            "vectorizer_exists": vectorizer_exists,
            "vectorizer_fallback_enabled": vectorizer_fallback_enabled,
            "vectorizer_effective_ready": vectorizer_effective_ready,
            "vectorizer_path": str(VECTORIZER_PATH),
            "inference_cache_entries": cache_size,
        }
    except Exception as exc:
        logger.error("Health check failed: %s", exc)
        return {"status": "unhealthy", "error": str(exc)}


@app.get("/project-view")
def project_view():
    return _build_project_view()


# ---------------------------------------------------------------------------
# Routes — predict (truthlens_v1 multitask engine)
# ---------------------------------------------------------------------------

@app.post("/predict", response_model=NewsResponse)
def predict_news(request: NewsRequest):
    """Predict whether a news article is fake or real using the truthlens_v1
    HuggingFace checkpoint.  Results are cached for one hour."""
    try:
        text = ensure_non_empty_text(request.text, name="request.text")
        logger.info("Received /predict request (text length: %d)", len(text))
        timer_start = INFERENCE_LOGGER.start_timer()

        cached = INFERENCE_CACHE.get(text)
        if cached is not None:
            return NewsResponse(**cached)

        engine = _get_inference_engine()
        if engine is not None:
            engine_result = engine.predict_single(text)
            fake_prob = engine_result.get("fake_probability")
            confidence = float(engine_result.get("confidence") or 0.5)
            if fake_prob is not None:
                prob = round(float(fake_prob), 4)
            else:
                label = int(engine_result.get("label", 0))
                prob = round(confidence if label == 1 else 1.0 - confidence, 4)
            confidence = round(confidence, 4)
            prediction = "FAKE" if prob > 0.5 else "REAL"
        else:
            fallback = _heuristic_predict_fn(text)
            prob = fallback["fake_probability"]
            prediction = fallback["prediction"]
            confidence = fallback["confidence"]

        response_data = {
            "text": _preview_text(text),
            "fake_probability": round(prob, 4),
            "prediction": prediction,
            "confidence": round(confidence, 4),
        }
        INFERENCE_CACHE.set(text, response_data)
        INFERENCE_LOGGER.log_prediction(
            article_id=None,
            start_time=timer_start,
            model_versions={"roberta": SETTINGS.model.name},
            feature_count=0,
            prediction_confidence=round(confidence, 4),
        )
        logger.info("Prediction: %s (confidence: %.4f)", prediction, confidence)
        return NewsResponse(**response_data)

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("Prediction error: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error during prediction")


@app.post("/batch-predict", response_model=BatchNewsResponse)
def batch_predict_news(request: BatchNewsRequest):
    """Batch predict fake/real for up to 50 news articles using truthlens_v1."""
    try:
        normalized_texts = ensure_non_empty_text_list(request.texts, name="request.texts")
        logger.info("Received /batch-predict request (%d texts)", len(normalized_texts))

        results: list[NewsResponse] = []
        cache_hits = 0
        uncached_texts: list[str] = []
        uncached_indices: list[int] = []

        for i, text in enumerate(normalized_texts):
            cached = INFERENCE_CACHE.get(text)
            if cached is not None:
                results.append(NewsResponse(**cached))
                cache_hits += 1
            else:
                results.append(None)
                uncached_texts.append(text)
                uncached_indices.append(i)

        if uncached_texts:
            timer_start = INFERENCE_LOGGER.start_timer()
            engine = _get_inference_engine()

            if engine is not None:
                engine_results = engine.predict(uncached_texts)
                for idx, engine_result in zip(uncached_indices, engine_results):
                    text = normalized_texts[idx]
                    fake_prob = engine_result.get("fake_probability")
                    confidence = float(engine_result.get("confidence") or 0.5)
                    if fake_prob is not None:
                        prob = round(float(fake_prob), 4)
                    else:
                        label = int(engine_result.get("label", 0))
                        prob = round(confidence if label == 1 else 1.0 - confidence, 4)
                    confidence = round(confidence, 4)
                    prediction = "FAKE" if prob > 0.5 else "REAL"
                    response_data = {
                        "text": _preview_text(text),
                        "fake_probability": prob,
                        "prediction": prediction,
                        "confidence": confidence,
                    }
                    INFERENCE_CACHE.set(text, response_data)
                    results[idx] = NewsResponse(**response_data)
            else:
                batch_probs = predict_batch(uncached_texts)
                for idx, probs, text in zip(uncached_indices, batch_probs, uncached_texts):
                    prob_real, prob_fake = float(probs[0]), float(probs[1])
                    prob = round(prob_fake, 4)
                    confidence = round(max(prob_real, prob_fake), 4)
                    prediction = "FAKE" if prob > 0.5 else "REAL"
                    response_data = {
                        "text": _preview_text(text),
                        "fake_probability": prob,
                        "prediction": prediction,
                        "confidence": confidence,
                    }
                    INFERENCE_CACHE.set(text, response_data)
                    results[idx] = NewsResponse(**response_data)

            INFERENCE_LOGGER.log_prediction(
                article_id=None,
                start_time=timer_start,
                model_versions={"roberta": SETTINGS.model.name},
                feature_count=0,
                prediction_confidence=None,
            )

        return BatchNewsResponse(results=results, total=len(normalized_texts), cache_hits=cache_hits)

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("Batch prediction error: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error during batch prediction")


# ---------------------------------------------------------------------------
# Routes — v2 predict (truthlens2 — fast REAL/FAKE classifier)
# ---------------------------------------------------------------------------

@app.get("/v2/health")
def v2_health():
    """Health check for the TruthLens2 (bhavaygupta2002/truthlens2) model."""
    loaded = _v2_model is not None
    return {
        "status": "healthy" if loaded else "model_not_loaded",
        "model_repo": HF_REPO_V2,
        "model_loaded": loaded,
        "device": str(_v2_device) if _v2_device is not None else None,
        "labels": _v2_idx_to_label,
    }


@app.post("/v2/predict", response_model=V2PredictResponse, tags=["TruthLens2"])
def v2_predict_news(request: V2PredictRequest):
    """Classify a single news article as REAL or FAKE using
    bhavaygupta2002/truthlens2 (RoBERTa fine-tuned on multi-source news datasets)."""
    try:
        result = _v2_predict_single(request.text)
        return V2PredictResponse(text_preview=request.text[:200], **result)
    except Exception as exc:
        logger.error("V2 prediction failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}")


@app.post("/v2/batch-predict", response_model=V2BatchPredictResponse, tags=["TruthLens2"])
def v2_batch_predict_news(request: V2BatchPredictRequest):
    """Classify up to 50 news articles in a single batched forward pass using
    bhavaygupta2002/truthlens2."""
    try:
        if not request.texts:
            raise ValueError("texts list is empty")
        raw_results = _v2_predict_batch(request.texts)
        responses = [
            V2PredictResponse(text_preview=text[:200], **result)
            for text, result in zip(request.texts, raw_results)
        ]
        return V2BatchPredictResponse(results=responses, total=len(responses))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("V2 batch prediction failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {exc}")


# ---------------------------------------------------------------------------
# Routes — deep analysis
# ---------------------------------------------------------------------------

@app.post("/analyze", response_model=AnalysisResponse)
def analyze_news(request: NewsRequest):
    """Unified deep-analysis endpoint: prediction + linguistic, narrative,
    framing, rhetoric, discourse, propaganda, and credibility analysis."""
    try:
        text = ensure_non_empty_text(request.text, name="request.text")
        logger.info("Received /analyze request (text length: %d)", len(text))
        timer_start = INFERENCE_LOGGER.start_timer()

        cache_key = f"analyze:{text}"
        cached = INFERENCE_CACHE.get(cache_key)
        if cached is not None:
            return AnalysisResponse(**cached)

        _model_unavailable = False
        try:
            prediction_result = predict(text)
            fake_probability, prediction, confidence = _decode_prediction_result(prediction_result)
            _analyze_predict_fn = predict_batch
        except FileNotFoundError:
            engine = _get_inference_engine()
            if engine is not None:
                er = engine.predict_single(text)
                fake_probability = float(er.get("fake_probability") or 0.5)
                confidence = float(er.get("confidence") or 0.5)
                prediction = "FAKE" if fake_probability > 0.5 else "REAL"
                _analyze_predict_fn = predict_batch
            else:
                _model_unavailable = True
                _fallback = _heuristic_predict_fn(text)
                fake_probability = _fallback["fake_probability"]
                prediction = _fallback["prediction"]
                confidence = _fallback["confidence"]
                _analyze_predict_fn = _heuristic_predict_fn

        bias_result = compute_bias_features(text)
        emotion_result = EMOTION_ANALYZER.analyze(text)
        emotion_scores: dict[str, float] = getattr(emotion_result, "emotion_scores", {})

        narrative_roles: dict = _safe_run(NARRATIVE_ROLE_EXTRACTOR.analyze, request.text)
        hero_entities: list = narrative_roles.get("hero_entities", [])
        villain_entities: list = narrative_roles.get("villain_entities", [])
        victim_entities: list = narrative_roles.get("victim_entities", [])

        narrative_conflict: dict = _safe_run(
            NARRATIVE_CONFLICT_ANALYZER.analyze, request.text,
            hero_entities=hero_entities, villain_entities=villain_entities, victim_entities=victim_entities,
        )
        narrative_propagation: dict = _safe_run(
            NARRATIVE_PROPAGATION_ANALYZER.analyze, request.text,
            hero_entities=hero_entities, villain_entities=villain_entities, victim_entities=victim_entities,
        )
        narrative_temporal: dict = _safe_run(NARRATIVE_TEMPORAL_ANALYZER.analyze, request.text)
        framing: dict = _safe_run(FRAMING_ANALYZER.analyze, request.text)
        rhetorical: dict = _safe_run(RHETORICAL_DETECTOR.analyze, request.text)
        argument: dict = _safe_run(ARGUMENT_ANALYZER.analyze, request.text)
        info_density: dict = _safe_run(INFO_DENSITY_ANALYZER.analyze, request.text)
        info_omission: dict = _safe_run(INFO_OMISSION_DETECTOR.analyze, request.text)
        context_omission: dict = _safe_run(CONTEXT_OMISSION_DETECTOR.analyze, request.text)
        discourse_coherence: dict = _safe_run(DISCOURSE_ANALYZER.analyze, request.text)
        ideological: dict = _safe_run(IDEOLOGICAL_DETECTOR.analyze, request.text)
        emotion_target: dict = _safe_run(EMOTION_TARGET_ANALYZER.analyze, request.text)
        source_attribution: dict = _safe_run(SOURCE_ATTRIBUTION_ANALYZER.analyze, request.text)

        combined_narrative: dict = {**narrative_conflict, **narrative_propagation, **narrative_temporal}
        combined_info: dict = {**info_density, **info_omission}
        propaganda_patterns: dict = _safe_run(
            PROPAGANDA_PATTERN_DETECTOR.analyze,
            emotion_features=emotion_scores,
            narrative_features=combined_narrative,
            rhetorical_features=rhetorical,
            argument_features=argument,
            information_features=combined_info,
        )

        combined_discourse: dict = {
            **discourse_coherence, **context_omission,
            **info_density, **info_omission, **source_attribution,
        }
        credibility_profile: dict = _safe_run(
            BIAS_PROFILE_BUILDER.build_profile,
            bias={"bias_score": float(bias_result.bias_score)},
            emotion=emotion_scores,
            narrative=combined_narrative,
            discourse=combined_discourse,
            ideology=ideological,
        )

        raw_graph_result: dict = _safe_run(GRAPH_PIPELINE.run, request.text)
        graph_result: dict = _serialize_graph_result(raw_graph_result)
        entity_graph: dict = graph_result.get("entity_graph", {})
        entity_embeddings: list = []
        if entity_graph:
            try:
                embedding_arr = GRAPH_EMBEDDING_GENERATOR.generate_embedding(entity_graph)
                entity_embeddings = embedding_arr.tolist()
            except Exception as emb_err:
                logger.warning("Entity graph embedding failed: %s", emb_err)

        raw_temporal = _safe_run(lambda t: TEMPORAL_GRAPH_ANALYZER.analyze(t).to_dict(), request.text)
        graph_analysis: dict = {
            "entity_graph": entity_graph,
            "entity_graph_metrics": graph_result.get("entity_graph_metrics", {}),
            "entity_embeddings": entity_embeddings,
            "narrative_graph": graph_result.get("narrative_graph", {}),
            "narrative_graph_metrics": graph_result.get("narrative_graph_metrics", {}),
            "graph_features": graph_result.get("graph_features", {}),
            "temporal_graph": raw_temporal,
        }

        emotion_explanation = _safe_run(explain_emotion, request.text)
        try:
            lime_result = explain_prediction(
                _analyze_predict_fn, request.text, num_features=8, num_samples=LIME_NUM_SAMPLES,
            )
        except Exception as lime_error:
            logger.warning("LIME explanation unavailable: %s", lime_error)
            lime_result = {"text": request.text, "important_features": [], "error": "lime_unavailable"}

        response_data: dict[str, Any] = {
            "text": _preview_text(request.text),
            "prediction": prediction,
            "fake_probability": round(fake_probability, 4),
            "confidence": round(confidence, 4),
            "bias": {
                "bias_score": round(float(bias_result.bias_score), 4),
                "media_bias": bias_result.media_bias,
                "biased_tokens": bias_result.biased_tokens,
                "sentence_heatmap": bias_result.sentence_heatmap,
            },
            "emotion": {
                "dominant_emotion": emotion_result.dominant_emotion,
                "emotion_scores": emotion_scores,
                "emotion_distribution": emotion_scores,
            },
            "narrative": {
                "roles": narrative_roles,
                "conflict": narrative_conflict,
                "propagation": narrative_propagation,
                "temporal": narrative_temporal,
            },
            "framing": framing,
            "rhetoric": {"rhetorical_devices": rhetorical, "argument_structure": argument},
            "discourse": {
                "coherence": discourse_coherence,
                "context_omission": context_omission,
                "information_density": info_density,
                "information_omission": info_omission,
                "source_attribution": source_attribution,
                "ideological_language": ideological,
                "emotion_targets": emotion_target,
            },
            "propaganda_analysis": propaganda_patterns,
            "credibility_profile": credibility_profile,
            "graph_analysis": graph_analysis,
            "explainability": {"emotion_explanation": emotion_explanation, "lime": lime_result},
        }

        INFERENCE_CACHE.set(cache_key, response_data)
        credibility_score = credibility_profile.get("credibility_score")
        INFERENCE_LOGGER.log_prediction(
            article_id=None,
            start_time=timer_start,
            model_versions={"roberta": SETTINGS.model.name},
            feature_count=len(combined_info) + len(combined_narrative),
            prediction_confidence=round(confidence, 4),
        )
        return AnalysisResponse(**response_data)

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("Analysis error: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error during analysis")


@app.post("/explain")
def explain_article(request: NewsRequest):
    """Full explainability pipeline with credibility aggregation."""
    try:
        text = ensure_non_empty_text(request.text, name="request.text")
        logger.info("Received /explain request (text length: %d)", len(text))

        engine = _get_inference_engine()
        if engine is not None:
            def _model_predict(t: str) -> dict:
                try:
                    result = predict(t)
                    return result if isinstance(result, dict) else {"fake_probability": float(result)}
                except Exception:
                    return _heuristic_predict_fn(t)
            predict_fn = _model_predict
            predict_source = "model"
        else:
            predict_fn = _heuristic_predict_fn
            predict_source = "heuristic_fallback"

        logger.info("Explainability predict_fn source: %s", predict_source)

        expl_config = ExplainabilityConfig(
            enabled=True,
            use_lime=True,
            use_shap=False,
            use_bias_emotion=False,
            use_attention_rollout=False,
            use_graph_explainer=True,
            use_aggregation=True,
            use_consistency=True,
            use_explanation_metrics=True,
            cache_enabled=False,
        )
        expl_result = run_explainability_pipeline(text=text, predict_fn=predict_fn, config=expl_config)

        bias_result = compute_bias_features(text)
        emotion_result = EMOTION_ANALYZER.analyze(text)
        emotion_scores: dict = getattr(emotion_result, "emotion_scores", {}) or {}
        narrative_features: dict = _safe_run(NARRATIVE_CONFLICT_ANALYZER.analyze, text)
        discourse_features: dict = _safe_run(DISCOURSE_ANALYZER.analyze, text)

        credibility_profile: dict = _safe_run(
            BIAS_PROFILE_BUILDER.build_profile,
            bias={"bias_score": float(bias_result.bias_score)},
            emotion=emotion_scores,
            narrative=narrative_features,
            discourse=discourse_features,
            ideology={},
        )

        aggregation_result: dict = {}
        try:
            aggregation_result = AGGREGATION_PIPELINE.run(profile=credibility_profile or {}, text=text)
        except Exception as agg_err:
            logger.warning("Credibility aggregation failed in /explain: %s", agg_err)

        def _ser(obj: Any) -> Any:
            if obj is None:
                return None
            if hasattr(obj, "model_dump"):
                return obj.model_dump()
            if hasattr(obj, "__dict__"):
                return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
            return obj

        base_pred = predict_fn(text)
        return {
            "text": _preview_text(text),
            "prediction": base_pred,
            "predict_source": predict_source,
            "explainability": {
                "lime": _ser(expl_result.lime_explanation),
                "aggregated": _ser(expl_result.aggregated_explanation),
                "consistency_metrics": expl_result.consistency_metrics,
                "explanation_metrics": expl_result.explanation_metrics,
                "explanation_quality_score": expl_result.explanation_quality_score,
                "emotion_explanation": _ser(expl_result.emotion_explanation),
                "module_failures": expl_result.module_failures,
                "metadata": expl_result.metadata,
            },
            "aggregation": aggregation_result,
        }

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("Explain pipeline error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Explainability pipeline error: {exc}")


@app.post("/report", response_model=ReportResponse)
def generate_report(request: NewsRequest):
    """Generate a structured analysis report (lighter than /analyze)."""
    try:
        text = ensure_non_empty_text(request.text, name="request.text")
        logger.info("Received /report request (text length: %d)", len(text))

        cache_key = f"report:{text}"
        cached = INFERENCE_CACHE.get(cache_key)
        if cached is not None:
            return ReportResponse(**cached)

        bias_result = compute_bias_features(text)
        emotion_result = EMOTION_ANALYZER.analyze(text)
        emotion_scores: dict[str, float] = getattr(emotion_result, "emotion_scores", {})
        narrative_roles: dict = _safe_run(NARRATIVE_ROLE_EXTRACTOR.analyze, text)
        combined_narrative: dict = {**_safe_run(NARRATIVE_CONFLICT_ANALYZER.analyze, text)}
        combined_discourse: dict = {**_safe_run(DISCOURSE_ANALYZER.analyze, text)}

        credibility_profile: dict = _safe_run(
            BIAS_PROFILE_BUILDER.build_profile,
            bias={"bias_score": float(bias_result.bias_score)},
            emotion=emotion_scores,
            narrative=combined_narrative,
            discourse=combined_discourse,
            ideology={},
        )
        credibility_score: Optional[float] = credibility_profile.get("credibility_score")

        report = REPORT_GENERATOR.generate_report(
            article_text=text,
            title=None,
            source=None,
            analysis={
                "bias": {
                    "bias_score": round(float(bias_result.bias_score), 4),
                    "media_bias": bias_result.media_bias,
                    "biased_tokens": bias_result.biased_tokens,
                },
                "emotion": {
                    "dominant_emotion": emotion_result.dominant_emotion,
                    "emotion_scores": emotion_scores,
                },
                "narrative": narrative_roles,
                "credibility_score": credibility_score,
            },
        )

        response_data = {
            "article_summary": report.get("article_summary", {}),
            "bias_analysis": report.get("bias_analysis", {}),
            "emotion_analysis": report.get("emotion_analysis", {}),
            "narrative_structure": report.get("narrative_structure", {}),
            "entity_graph": report.get("entity_graph", {}),
            "credibility_score": report.get("credibility_score"),
        }
        INFERENCE_CACHE.set(cache_key, response_data)
        return ReportResponse(**response_data)

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("Report generation error: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error during report generation")


# ---------------------------------------------------------------------------
# Routes — inference model info + cache
# ---------------------------------------------------------------------------

@app.get("/inference/model-info", response_model=ModelInfoResponse)
def inference_model_info():
    engine = _get_inference_engine()
    if engine is None:
        return ModelInfoResponse(
            available=False, model_path=str(MODEL_PATH), device=None,
            num_parameters=None, num_trainable_parameters=None, label_map=None,
        )
    try:
        info = engine.get_model_info()
        label_map = (
            {str(k): v for k, v in engine.label_map.items()} if engine.label_map else None
        )
        return ModelInfoResponse(
            available=True,
            model_path=info["model_path"],
            device=info["device"],
            num_parameters=info["num_parameters"],
            num_trainable_parameters=info["num_trainable_parameters"],
            label_map=label_map,
        )
    except Exception as exc:
        logger.error("Failed to retrieve model info: %s", exc)
        raise HTTPException(status_code=500, detail="Could not retrieve model information")


@app.post("/cache/clear")
def clear_inference_cache():
    try:
        INFERENCE_CACHE.clear()
        logger.info("Inference cache cleared via /cache/clear")
        return {"message": "Inference cache cleared successfully"}
    except Exception as exc:
        logger.error("Cache clear failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to clear inference cache")


# ---------------------------------------------------------------------------
# Routes — calibration
# ---------------------------------------------------------------------------

@app.get("/calibration/info")
def calibration_info():
    return {
        "methods": {
            "temperature_scaling": {
                "class": "TemperatureScaler",
                "description": "Learns a scalar temperature T that divides logits before softmax.",
                "reference": "Guo et al. (2017) — 'On Calibration of Modern Neural Networks'",
                "parameters": {
                    "lr": TemperatureScalingConfig().lr,
                    "max_iter": TemperatureScalingConfig().max_iter,
                    "tolerance": TemperatureScalingConfig().tolerance,
                },
            },
            "isotonic_regression": {
                "class": "IsotonicCalibrator",
                "description": "Non-parametric monotone calibration using scikit-learn IsotonicRegression.",
                "reference": "Zadrozny & Elkan (2002)",
                "parameters": {
                    "out_of_bounds": IsotonicCalibrationConfig().out_of_bounds,
                    "increasing": IsotonicCalibrationConfig().increasing,
                },
            },
        },
        "metrics_endpoint": "POST /calibration/metrics",
    }


@app.post("/calibration/metrics", response_model=CalibrationMetricsResponse)
def compute_calibration_metrics(request: CalibrationMetricsRequest):
    try:
        n = len(request.probabilities)
        if n == 0:
            raise HTTPException(status_code=400, detail="probabilities list must not be empty")
        if len(request.labels) != n:
            raise HTTPException(
                status_code=400,
                detail=f"probabilities has {n} rows but labels has {len(request.labels)} entries",
            )
        metrics_obj = CalibrationMetrics(CalibrationMetricConfig(n_bins=request.n_bins))
        probs_tensor = torch.tensor(request.probabilities, dtype=torch.float32)
        labels_tensor = torch.tensor(request.labels, dtype=torch.long)
        metrics = metrics_obj.compute_all_metrics(probs_tensor, labels_tensor)
        return CalibrationMetricsResponse(
            ece=round(metrics["ece"], 6),
            mce=round(metrics["mce"], 6),
            brier_score=round(metrics["brier_score"], 6),
            nll=round(metrics["nll"], 6),
            n_samples=n,
            n_bins=request.n_bins,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Calibration metrics computation failed: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error during calibration metric computation")


# ---------------------------------------------------------------------------
# Routes — ensemble
# ---------------------------------------------------------------------------

@app.get("/ensemble/info")
def ensemble_info():
    return {
        "strategies": {
            "average": {"description": "Averages logits from all member models before applying softmax."},
            "weighted_average": {"description": "Each model's logits multiplied by its weight before summation."},
            "majority_vote": {"description": "Each model votes for a class; the class with most votes wins."},
        },
        "predict_endpoint": "POST /ensemble/predict",
    }


@app.post("/ensemble/predict", response_model=EnsemblePredictResponse)
def ensemble_predict(request: EnsemblePredictRequest):
    try:
        valid_strategies = {"average", "weighted_average", "majority_vote"}
        if request.strategy not in valid_strategies:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown strategy '{request.strategy}'. Must be one of {sorted(valid_strategies)}.",
            )
        n_models = len(request.model_probabilities)
        if n_models == 0:
            raise HTTPException(status_code=400, detail="model_probabilities must not be empty")
        for i, probs in enumerate(request.model_probabilities):
            if len(probs) != 2:
                raise HTTPException(
                    status_code=400,
                    detail=f"model_probabilities[{i}] must have exactly 2 values [p_real, p_fake]",
                )
        probs_tensor = torch.tensor(request.model_probabilities, dtype=torch.float32)

        if request.strategy == "average":
            combined = torch.mean(probs_tensor, dim=0)
        elif request.strategy == "weighted_average":
            if request.weights is None:
                raise HTTPException(status_code=400, detail="weights required for weighted_average")
            if len(request.weights) != n_models:
                raise HTTPException(status_code=400, detail="weights length must match number of models")
            weights_tensor = torch.tensor(request.weights, dtype=torch.float32)
            weight_sum = weights_tensor.sum()
            if weight_sum <= 0:
                raise HTTPException(status_code=400, detail="Sum of weights must be positive")
            weights_tensor = weights_tensor / weight_sum
            combined = torch.sum(probs_tensor * weights_tensor.unsqueeze(1), dim=0)
        else:
            votes = torch.argmax(probs_tensor, dim=1)
            n_classes = probs_tensor.shape[1]
            vote_counts = torch.zeros(n_classes)
            for v in votes:
                vote_counts[v] += 1
            combined = vote_counts / vote_counts.sum()

        combined_list = combined.tolist()
        fake_prob = round(combined_list[1], 4)
        confidence = round(float(combined.max().item()), 4)
        prediction = "FAKE" if fake_prob > 0.5 else "REAL"

        return EnsemblePredictResponse(
            strategy=request.strategy,
            ensemble_probabilities=[round(p, 4) for p in combined_list],
            prediction=prediction,
            fake_probability=fake_prob,
            confidence=confidence,
            num_models=n_models,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Ensemble predict failed: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error during ensemble prediction")


# ---------------------------------------------------------------------------
# Routes — export
# ---------------------------------------------------------------------------

@app.get("/export/info")
def export_info():
    engine = _get_inference_engine()
    return {
        "model_ready": engine is not None,
        "model_path": str(MODEL_PATH),
        "formats": {
            "onnx": {
                "endpoint": "POST /export/onnx",
                "config": {"opset_version": ONNXExportConfig().opset_version, "dynamic_batch": ONNXExportConfig().dynamic_batch},
            },
            "torchscript": {
                "endpoint": "POST /export/torchscript",
                "config": {"method": TorchScriptExportConfig().method},
            },
        },
    }


@app.post("/export/onnx", response_model=ExportResponse)
def export_onnx(request: ExportRequest):
    try:
        engine = _get_inference_engine()
        if engine is None:
            raise HTTPException(status_code=503, detail="Model not available.")
        model = engine.model
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded in InferenceEngine")
        max_length = SETTINGS.model.max_length
        example_input = torch.zeros(1, max_length, dtype=torch.long)
        output_path = ONNX_EXPORTER.export(model, example_input, request.output_path)
        return ExportResponse(
            format="onnx", output_path=str(output_path), success=True,
            message=f"Model exported to ONNX at '{output_path}'",
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("ONNX export failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"ONNX export failed: {exc}")


@app.post("/export/torchscript", response_model=ExportResponse)
def export_torchscript(request: ExportRequest):
    try:
        engine = _get_inference_engine()
        if engine is None:
            raise HTTPException(status_code=503, detail="Model not available.")
        model = engine.model
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded in InferenceEngine")
        max_length = SETTINGS.model.max_length
        example_input = torch.zeros(1, max_length, dtype=torch.long)
        output_path = TORCHSCRIPT_EXPORTER.export(model, example_input, request.output_path)
        return ExportResponse(
            format="torchscript", output_path=str(output_path), success=True,
            message=f"Model exported to TorchScript at '{output_path}'",
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("TorchScript export failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"TorchScript export failed: {exc}")
