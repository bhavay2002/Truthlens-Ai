# `src/inference/` — TruthLens Inference Subsystem

Production-grade reference for all 20 Python modules that constitute the TruthLens inference layer. Covers architecture, every public class and method, all known design decisions and bug-fixes (tagged `CRIT-*`, `CFG-*`, `PP-*`, `LAT-*`, `REC-*`), and integration guidance.

---

## Table of Contents

1. [Subsystem Overview](#1-subsystem-overview)
2. [Architecture and Data Flow](#2-architecture-and-data-flow)
3. [Module Inventory](#3-module-inventory)
4. [Constants (`constants.py`)](#4-constants-constantspy)
5. [Schema (`schema.py`)](#5-schema-schemapy)
6. [Model Loader (`model_loader.py`)](#6-model-loader-model_loaderpy)
7. [Inference Engine (`inference_engine.py`)](#7-inference-engine-inference_enginepy)
8. [Inference Config Loader (`inference_config.py`)](#8-inference-config-loader-inference_configpy)
9. [Feature Preparer (`feature_preparer.py`)](#9-feature-preparer-feature_preparerpy)
10. [Inference Cache (`inference_cache.py`)](#10-inference-cache-inference_cachepy)
11. [Inference Pipeline (`inference_pipeline.py`)](#11-inference-pipeline-inference_pipelinepy)
12. [Postprocessor (`postprocessing.py`)](#12-postprocessor-postprocessingpy)
13. [Prediction Service (`prediction_service.py`)](#13-prediction-service-prediction_servicepy)
14. [Predict API (`predict_api.py`)](#14-predict-api-predict_apipy)
15. [Batch Inference (`batch_inference.py`)](#15-batch-inference-batch_inferencepy)
16. [Inference Monitor (`monitoring.py`)](#16-inference-monitor-monitoringpy)
17. [Drift Detector (`drift_detection.py`)](#17-drift-detector-drift_detectionpy)
18. [Inference Logger (`inference_logger.py`)](#18-inference-logger-inference_loggerpy)
19. [Result Formatter and Report Generator (`result_formatter.py`, `report_generator.py`)](#19-result-formatter-and-report-generator)
20. [Article Analyzer and CLI (`analyze_article.py`, `run_inference.py`)](#20-article-analyzer-and-cli)
21. [Public Package API (`__init__.py`)](#21-public-package-api-__init__py)
22. [Cross-Cutting Concerns](#22-cross-cutting-concerns)

---

## 1. Subsystem Overview

`src/inference/` is the end-to-end model serving layer of TruthLens. It bridges the trained `MultiTaskTruthLensModel` (HuggingFace checkpoint `bhavaygupta2002/truthlens_v1/checkpoint.pt`, encoder `roberta-base`) with the FastAPI application layer and the broader analysis pipeline.

### Responsibilities

| Concern | Handled by |
|---|---|
| Model loading and warm-up | `ModelLoader`, `InferenceEngine` |
| Tokenisation, AMP inference, calibration | `InferenceEngine` |
| Feature engineering for ML head | `FeaturePreparer` |
| Caching (LRU + gzip disk) | `InferenceCache` |
| Post-processing logits → labels | `PostProcessor` |
| High-level prediction pipeline | `PredictionPipeline` |
| Cache + monitor + logger wrapper | `PredictionService` |
| Thread-safe singleton for FastAPI | `predict_api._get_service()` |
| Bulk throughput | `BatchInferenceEngine` |
| Latency / confidence monitoring | `InferenceMonitor` |
| Distribution drift detection | `DriftDetector` |
| Structured audit logging | `InferenceLogger` |
| Report assembly | `ReportGenerator` |
| Output formatting (API / dashboard / research) | `ResultFormatter` |
| Full-article orchestration | `ArticleAnalyzer` |
| CLI evaluation harness | `run_inference.py` |
| Config loading and validation | `InferenceConfigLoader` |
| Shared constants | `constants.py` |
| Pydantic request / response schema | `schema.py` |

### Supported Tasks

`emotion` · `narrative` · `propaganda` · `bias` · `ideology` · `narrative_frame`

---

## 2. Architecture and Data Flow

```
HTTP POST /predict
        │
        ▼
predict_api._get_service()          # thread-safe singleton (double-checked lock)
        │
        ▼
PredictionService.predict(text)
    ├─ InferenceCache.get(key)       # LRU memory first → gzip disk (LAT-5)
    │       hit ──────────────────────────────────────────────────► return cached
    │       miss
    │
    ├─ InferenceEngine.predict(text)
    │       ├─ AutoTokenizer.encode
    │       ├─ torch.cuda.amp.autocast (AMP)
    │       ├─ model.forward → per-task logits
    │       ├─ temperature / Platt scaling calibration
    │       └─ PostProcessor.process (logits → labels)
    │
    ├─ InferenceMonitor.update(latency, confidence, probs)
    ├─ InferenceLogger.log_prediction(...)
    └─ InferenceCache.set(key, result)
        │
        ▼
    return prediction dict

                        Batch path
                        ──────────
          BatchInferenceEngine.run_batch(texts)
              ├─ chunk texts into batch_size windows
              ├─ InferenceEngine.predict (per chunk)
              └─ DriftDetector.detect (post-batch)

                        Article analysis path
                        ─────────────────────
          ArticleAnalyzer.analyze(text)
              ├─ FeaturePipeline.extract
              ├─ EntityGraphBuilder / GraphAnalyzer
              ├─ AnalysisIntegrationRunner
              ├─ BiasProfileBuilder → AggregationPipeline
              ├─ PredictionService.predict  (wrapped InferenceEngine)
              └─ ReportGenerator.generate_report → ResultFormatter
```

---

## 3. Module Inventory

| File | Primary export(s) | LOC (approx.) |
|---|---|---|
| `constants.py` | module-level constants | 30 |
| `schema.py` | `PredictRequest`, `PredictResponse`, `BatchPredictRequest` | 80 |
| `model_loader.py` | `ModelLoader`, `ModelArtifacts` | 120 |
| `inference_engine.py` | `InferenceConfig`, `InferenceEngine` | 350 |
| `inference_config.py` | `InferenceConfigLoader`, `load_inference_config` | 158 |
| `feature_preparer.py` | `FeaturePreparationConfig`, `FeaturePreparer` | 200 |
| `inference_cache.py` | `InferenceCacheConfig`, `InferenceCache` | 250 |
| `inference_pipeline.py` | `PredictionPipelineConfig`, `PredictionPipeline` | 220 |
| `postprocessing.py` | `PostProcessingConfig`, `PostProcessor` | 180 |
| `prediction_service.py` | `PredictionService` | 150 |
| `predict_api.py` | `router`, `_get_service()` | 100 |
| `batch_inference.py` | `BatchConfig`, `BatchInferenceEngine` | 200 |
| `monitoring.py` | `MonitoringConfig`, `MetricWindow`, `InferenceMonitor` | 181 |
| `drift_detection.py` | `DriftConfig`, `DriftDetector` | 193 |
| `inference_logger.py` | `InferenceLogEntry`, `InferenceLogger` | 208 |
| `result_formatter.py` | `TruthLensAPIResponse`, `TruthLensDashboardReport`, `TruthLensResearchExport`, `ResultFormatter` | 222 |
| `report_generator.py` | `ArticleSummary`, `ReportConfig`, `ReportGenerator` | 237 |
| `analyze_article.py` | `ArticleAnalyzer` | 255 |
| `run_inference.py` | CLI entry point `main()` | 213 |
| `__init__.py` | re-exports 14 public symbols | 40 |

---

## 4. Constants (`constants.py`)

Single source of truth for values that were previously duplicated across modules (fixes **CFG-2**, **CFG-5**, **CFG-6**).

```python
from src.inference.constants import (
    INFERENCE_CACHE_VERSION,
    DEFAULT_INFERENCE_BATCH_SIZE,
    DEFAULT_MAX_LENGTH,
    REPORT_VERSION,
)
```

### Values

| Name | Value | Description |
|---|---|---|
| `INFERENCE_CACHE_VERSION` | `"v2"` | Cache schema version. Bump whenever the cached prediction blob's shape changes. Consumed by `InferenceCache` (default key prefix) and `predict_api` so neither can drift. |
| `DEFAULT_INFERENCE_BATCH_SIZE` | `32` | Default batch size for `InferenceEngine`, `BatchInferenceEngine`, and `InferenceConfigLoader` when no caller or YAML override is present. |
| `DEFAULT_MAX_LENGTH` | `512` | Tokenizer `max_length` fallback. |
| `REPORT_VERSION` | `"v3"` | Report schema version embedded in every `ReportGenerator` output under `metadata.report_version`. Bump on schema changes. |

### Design Rule

> Never re-declare these values in other modules. Import from `constants.py`. Values that change per deployment belong in `config/config.yaml` and are accessed via `src.utils.settings`.

---

## 5. Schema (`schema.py`)

Pydantic v2 models that define the FastAPI contract at `/predict` and `/batch_predict`.

### `PredictRequest`

```python
class PredictRequest(BaseModel):
    text: str
    task_types: Optional[List[str]] = None
    article_id: Optional[str] = None
```

| Field | Type | Notes |
|---|---|---|
| `text` | `str` | Article or passage to classify. Must be non-empty. |
| `task_types` | `List[str] \| None` | If `None`, the engine runs all six tasks. Subset selection prunes the postprocessor. |
| `article_id` | `str \| None` | Caller-supplied identifier passed through to logs and cache keys. |

### `PredictResponse`

```python
class PredictResponse(BaseModel):
    label: str
    confidence: float
    fake_probability: float
    task_outputs: Optional[Dict[str, Any]]
    article_id: Optional[str]
    cached: bool
    processing_time_ms: Optional[float]
```

### `BatchPredictRequest`

```python
class BatchPredictRequest(BaseModel):
    texts: List[str]
    task_types: Optional[List[str]] = None
```

---

## 6. Model Loader (`model_loader.py`)

Handles artifact discovery, checkpoint loading, tokenizer construction, and optional quantisation. Separates loading concerns from inference concerns.

### `ModelArtifacts`

```python
@dataclass
class ModelArtifacts:
    model: nn.Module
    tokenizer: Any
    config: Dict[str, Any]
    label_maps: Dict[str, Any]
    device: str
```

All downstream consumers (notably `InferenceEngine`) receive a `ModelArtifacts` instance rather than raw file paths, ensuring the device placement and tokenizer configuration are always consistent.

### `ModelLoader`

```python
class ModelLoader:
    def __init__(self, model_dir: str | Path, device: str = "cpu")
    def load(self) -> ModelArtifacts
```

#### `load()` — step-by-step

1. Resolves `model_dir` via `pathlib.Path`.
2. Loads `config.json` for architecture metadata.
3. Loads `label_maps.json` for per-task class indices.
4. Constructs tokenizer: `AutoTokenizer.from_pretrained(model_dir)`.
5. Instantiates `MultiTaskTruthLensModel` from the config.
6. Loads `checkpoint.pt` with `torch.load(..., map_location=device)`.
7. Calls `model.eval()` and places to device.
8. Returns `ModelArtifacts`.

#### Error handling

- Raises `FileNotFoundError` when required files are absent.
- Raises `RuntimeError` on checkpoint loading failure.
- Logs each step at `INFO`; checkpoint mismatch is logged at `ERROR` before re-raising.

---

## 7. Inference Engine (`inference_engine.py`)

The computational core. Owns the model, tokenizer, AMP context, calibration, and the two primary inference entry points.

### `InferenceConfig`

```python
@dataclass
class InferenceConfig:
    model_path: str
    device: str = "auto"
    batch_size: int = DEFAULT_INFERENCE_BATCH_SIZE
    max_length: int = DEFAULT_MAX_LENGTH
    use_amp: bool = True
    calibrate: bool = True
    temperature: float = 1.0
    warmup_steps: int = 2
    enable_full_pipeline: bool = True
```

| Field | Default | Purpose |
|---|---|---|
| `model_path` | — | Absolute or relative path to the model directory. |
| `device` | `"auto"` | `"cpu"` / `"cuda"` / `"auto"` (resolved at construction). |
| `batch_size` | `32` | Tokenizer chunking and GPU batch window. |
| `max_length` | `512` | Tokenizer truncation. |
| `use_amp` | `True` | Enables `torch.cuda.amp.autocast` on CUDA. |
| `calibrate` | `True` | Applies temperature / Platt scaling to logits. |
| `temperature` | `1.0` | Scaling divisor; values > 1.0 soften, < 1.0 sharpen distributions. |
| `warmup_steps` | `2` | Number of dummy forward passes during `__init__` to prime CUDA/JIT. |
| `enable_full_pipeline` | `True` | When `False`, skips postprocessing and calibration; used by `ArticleAnalyzer` when it constructs a lightweight engine. |

### `InferenceEngine`

```python
class InferenceEngine:
    def __init__(self, config: InferenceConfig)
    def predict(self, text: str, task_types: Optional[List[str]] = None) -> Dict[str, Any]
    def predict_batch(self, texts: List[str], task_types: Optional[List[str]] = None) -> List[Dict[str, Any]]
    def predict_for_evaluation(self, texts: List[str]) -> Dict[str, Any]
```

#### Construction sequence

1. Resolves `config.device` (`"auto"` → CUDA if available, else CPU).
2. Calls `ModelLoader(config.model_path, device).load()` → `ModelArtifacts`.
3. Initialises `PostProcessor` and calibration scaler if `config.calibrate`.
4. Runs `warmup_steps` dummy forward passes on a 16-token sequence.

#### `predict(text, task_types)`

```
tokenize(text, max_length, return_tensors="pt")
→ move tensors to device
→ torch.no_grad()
→ autocast (if use_amp and CUDA)
→ model.forward(input_ids, attention_mask)  →  per-task logits dict
→ _calibrate(logits)                         →  scaled logits
→ PostProcessor.process(logits, task_types)  →  predictions + probs
→ return {"predictions": {...}, "probabilities": {...}, "logits": {...}}
```

#### `predict_for_evaluation(texts)`

Runs `predict_batch` and aggregates results into a task-keyed dict of stacked NumPy arrays (logits, probabilities, predictions). Used by `run_inference.py` for offline evaluation.

```python
{
  "emotion":    {"logits": np.ndarray, "probabilities": np.ndarray, "predictions": np.ndarray},
  "narrative":  { ... },
  ...
}
```

#### AMP behaviour

`torch.cuda.amp.autocast` is entered only when `use_amp=True` **and** the resolved device is `"cuda"`. On CPU, the context manager is a no-op wrapper to avoid the overhead of CUDA-specific dtype negotiation.

#### Calibration

When `config.calibrate=True`, logits are divided by `config.temperature` before softmax. Platt scaling coefficients (if trained) are applied as a linear transform prior to temperature scaling. Both operations are applied inside `torch.no_grad()`.

---

## 8. Inference Config Loader (`inference_config.py`)

Loads, validates, and resolves an `InferenceConfig` from a YAML file. Addresses **CRIT-1**: previously two dataclasses with the same name (`InferenceConfig`) existed — one in the loader and one in the engine — causing silent field drops at the engine boundary. The loader now unconditionally re-exports the engine's dataclass.

### `InferenceConfigLoader`

```python
class InferenceConfigLoader:
    REQUIRED_FIELDS = {"device": str, "batch_size": int}

    def __init__(self, config_path: str | Path)
    def load(self) -> InferenceConfig
    def _validate_config(self, config: dict)
    def _resolve_device(self, device: str) -> str
    @staticmethod
    def from_dict(config: dict) -> InferenceConfig
```

#### `load()` sequence

1. Opens and parses the YAML file with `yaml.safe_load`.
2. Validates `device` is one of `{"cpu", "cuda", "auto"}` and `batch_size > 0` (**CFG-7** fix: absent keys skip type checks rather than raising, preserving the engine's default).
3. Strips unknown YAML keys (logs a `WARNING` listing them).
4. Injects `model_path` default from `config_path.parent` when YAML omits it.
5. Calls `_resolve_device` to translate `"auto"` and gracefully downgrade `"cuda"` to `"cpu"` when CUDA is unavailable.
6. Returns a fully-populated `InferenceConfig`.

#### YAML format

```yaml
model_path: /models/truthlens_v1
device: auto
batch_size: 16
max_length: 512
use_amp: true
calibrate: true
temperature: 1.2
warmup_steps: 3
```

#### `from_dict(config)`

Static convenience method that bypasses file I/O. Useful in tests and when config is already in memory.

### `load_inference_config(path)` (module-level helper)

```python
config = load_inference_config("config/inference.yaml")
```

---

## 9. Feature Preparer (`feature_preparer.py`)

Bridges the hand-crafted feature dictionary produced by `FeaturePipeline` into a PyTorch tensor suitable for the model's auxiliary feature head.

### `FeaturePreparationConfig`

```python
@dataclass
class FeaturePreparationConfig:
    use_bias_features: bool = True
    use_framing_features: bool = True
    use_ideological_features: bool = True
    scale_features: bool = True
    expected_dim: Optional[int] = None
```

### `FeaturePreparer`

```python
class FeaturePreparer:
    def __init__(self, config: Optional[FeaturePreparationConfig] = None)
    def prepare(self, features: Dict[str, float], device: str = "cpu") -> torch.Tensor
```

#### `prepare()` pipeline

```
1. BiasExtractor.extract(features)          → bias sub-dict
2. FramingExtractor.extract(features)       → framing sub-dict
3. IdeologicalExtractor.extract(features)   → ideology sub-dict
4. flatten all enabled sub-dicts            → 1-D float list
5. StandardScaler.fit_transform             → zero-mean unit-var (if scale_features)
6. torch.tensor(..., dtype=float32)
7. validate shape against expected_dim      → RuntimeError on mismatch
8. move to device
```

#### Extractor responsibilities

| Extractor | Input keys prefix | Example outputs |
|---|---|---|
| `BiasExtractor` | `bias_*` | partisan-lean score, source-credibility score |
| `FramingExtractor` | `narrative_*`, `discourse_*` | episodic/thematic framing, urgency |
| `IdeologicalExtractor` | `ideology_*` | left-right score, populism index |

---

## 10. Inference Cache (`inference_cache.py`)

Two-tier caching: in-process LRU (memory) backed by gzip-compressed JSON on disk. Implements single-flight deduplication to prevent cache stampede (**LAT-5**).

### `InferenceCacheConfig`

```python
@dataclass
class InferenceCacheConfig:
    memory_size: int = 512
    disk_cache_dir: str = ".cache/inference"
    version: str = INFERENCE_CACHE_VERSION   # "v2"
    ttl_seconds: Optional[int] = None
    compress: bool = True
```

### `InferenceCache`

```python
class InferenceCache:
    def __init__(self, config: Optional[InferenceCacheConfig] = None)
    def get(self, key: str) -> Optional[Dict[str, Any]]
    def set(self, key: str, value: Dict[str, Any])
    def invalidate(self, key: str)
    def clear(self)
    def stats(self) -> Dict[str, Any]
```

#### Cache key generation

```python
key = sha256(f"{version}:{text}".encode()).hexdigest()
```

The version prefix (`INFERENCE_CACHE_VERSION`) ensures stale entries from a prior schema are never served after a deployment that bumps the constant.

#### Read path

```
memory_lru.get(key)
    hit → return (fast path, no I/O)
    miss
        acquire per-key asyncio-style lock   # single-flight (LAT-5)
        re-check memory (another coroutine may have populated)
        disk_path = cache_dir / key[:2] / key
        if disk_path.exists():
            data = gzip.decompress(disk_path.read_bytes())
            json.loads(data)
            check TTL
            memory_lru[key] = result
            return result
        return None
```

#### Write path

```
memory_lru[key] = value
disk_path.parent.mkdir(parents=True, exist_ok=True)
disk_path.write_bytes(gzip.compress(json.dumps(value).encode()))
```

#### `stats()`

```python
{
    "memory_hits": int,
    "disk_hits": int,
    "misses": int,
    "memory_size": int,
    "disk_entries": int,
}
```

---

## 11. Inference Pipeline (`inference_pipeline.py`)

Orchestrates the full per-text prediction flow: feature preparation → engine inference → postprocessing. Designed as a standalone composable unit that `PredictionPipeline` wraps.

### `PredictionPipelineConfig`

```python
@dataclass
class PredictionPipelineConfig:
    task_types: Optional[List[str]] = None
    return_probabilities: bool = True
    return_logits: bool = False
    return_uncertainty: bool = True
    uncertainty_method: str = "entropy"
```

### `PredictionPipeline`

```python
class PredictionPipeline:
    def __init__(
        self,
        engine: InferenceEngine,
        feature_preparer: Optional[FeaturePreparer] = None,
        config: Optional[PredictionPipelineConfig] = None,
    )
    def run(self, text: str, features: Optional[Dict[str, float]] = None) -> Dict[str, Any]
    def run_batch(self, texts: List[str]) -> List[Dict[str, Any]]
```

#### `run()` flow

1. Optionally calls `FeaturePreparer.prepare(features)` to build the auxiliary tensor.
2. Calls `InferenceEngine.predict(text, task_types)`.
3. Strips logits from the result if `return_logits=False`.
4. Computes entropy per task when `return_uncertainty=True`.
5. Returns a unified result dict.

#### Uncertainty estimation

`uncertainty_method="entropy"` computes `H = -Σ p log p` per task using the softmax probability vector. The result is stored as `{"uncertainty": {"emotion": float, "narrative": float, ...}}`.

---

## 12. Postprocessor (`postprocessing.py`)

Converts raw logits into human-readable predictions. Applies softmax, argmax, and optional threshold logic per task.

### `PostProcessingConfig`

```python
@dataclass
class PostProcessingConfig:
    threshold: float = 0.5
    return_all_scores: bool = True
```

### `PostProcessor`

```python
class PostProcessor:
    def __init__(self, label_maps: Dict[str, List[str]], config: Optional[PostProcessingConfig] = None)
    def process(self, logits: Dict[str, torch.Tensor], task_types: Optional[List[str]] = None) -> Dict[str, Any]
```

#### **PP-3** (critical fix)

`task_types` must be explicitly passed or `process()` processes only the keys present in `logits`. Previously the method iterated `label_maps` unconditionally, which caused `KeyError` when the engine had selectively run a task subset. The fix: iterate `logits.keys()` filtered by `task_types` (or all if `None`).

#### Per-task output

```python
{
  "emotion": {
    "predictions": np.ndarray,        # argmax indices, shape (batch,)
    "probabilities": np.ndarray,      # softmax, shape (batch, n_classes)
    "labels": List[str],              # human-readable class names
    "confidence": float,              # max(probs[0])
  },
  ...
}
```

---

## 13. Prediction Service (`prediction_service.py`)

Thin orchestration layer that wraps `InferenceEngine` with caching, monitoring, and logging. This is the object held by `predict_api` and `ArticleAnalyzer`.

### `PredictionService`

```python
class PredictionService:
    def __init__(
        self,
        engine: InferenceEngine,
        cache: Optional[InferenceCache] = None,
        monitor: Optional[InferenceMonitor] = None,
        logger_: Optional[InferenceLogger] = None,
    )
    def predict(self, text: str, article_id: Optional[str] = None) -> Dict[str, Any]
    def get_monitoring_snapshot(self) -> Dict[str, Any]
```

#### `predict()` flow

```
t0 = perf_counter()
key = InferenceCache.make_key(text)
result = cache.get(key)
if result is None:
    result = engine.predict(text)
    cache.set(key, result)
    result["cached"] = False
else:
    result["cached"] = True

latency = (perf_counter() - t0) * 1000

monitor.update(
    latency_ms=latency,
    confidence=result.get("confidence"),
    probabilities=result.get("probabilities"),
)

logger_.log_prediction(
    start_time=t0,
    model_versions=engine.model_versions,
    feature_count=...,
    article_id=article_id,
    predicted_label=result.get("label"),
    prediction_confidence=result.get("confidence"),
)

result["processing_time_ms"] = latency
return result
```

#### Returned dict keys

| Key | Type | Description |
|---|---|---|
| `label` | `str` | Top-level binary label (`"real"` / `"fake"`) |
| `confidence` | `float` | Max probability across the primary task |
| `fake_probability` | `float` | Calibrated probability of the `"fake"` class |
| `task_outputs` | `dict` | Per-task logits/probs/labels |
| `cached` | `bool` | Whether the result was served from cache |
| `processing_time_ms` | `float` | End-to-end latency |

---

## 14. Predict API (`predict_api.py`)

FastAPI router that exposes `/predict` and `/predict/batch`. Manages a process-level `PredictionService` singleton using a double-checked lock pattern.

### Router registration

```python
# api/app.py
from src.inference.predict_api import router as inference_router
app.include_router(inference_router, prefix="/predict", tags=["inference"])
```

### Singleton pattern

```python
_service: Optional[PredictionService] = None
_lock = threading.Lock()

def _get_service() -> PredictionService:
    global _service
    if _service is None:
        with _lock:
            if _service is None:
                _service = _build_service()
    return _service
```

`_build_service()` reads `src.utils.settings` for model path and inference config, then constructs `InferenceEngine → PredictionService`.

### Endpoints

#### `POST /predict`

```
Request:  PredictRequest
Response: PredictResponse
Status:   200 OK | 422 Unprocessable Entity | 500 Internal Server Error
```

```python
@router.post("/", response_model=PredictResponse)
async def predict(request: PredictRequest):
    service = _get_service()
    result = service.predict(request.text, article_id=request.article_id)
    return PredictResponse(**result)
```

#### `POST /predict/batch`

```
Request:  BatchPredictRequest
Response: List[PredictResponse]
```

Iterates `request.texts`, calling `service.predict` for each. For high-throughput use, prefer `BatchInferenceEngine` directly.

#### `GET /predict/health`

Returns the monitoring snapshot from `PredictionService.get_monitoring_snapshot()`.

```json
{
  "latency_mean_ms": 42.3,
  "latency_p95_ms": 88.1,
  "confidence_mean": 0.79,
  "error_rate": 0.003,
  "total_requests": 14021
}
```

---

## 15. Batch Inference (`batch_inference.py`)

Optimised bulk inference runner. Processes a list of texts in configurable chunks, aggregates outputs, and optionally runs drift detection post-batch.

### `BatchConfig`

```python
@dataclass
class BatchConfig:
    batch_size: int = DEFAULT_INFERENCE_BATCH_SIZE
    show_progress: bool = True
    run_drift_detection: bool = False
    save_outputs: bool = False
    output_dir: str = "outputs/batch"
```

### `BatchInferenceEngine`

```python
class BatchInferenceEngine:
    def __init__(
        self,
        engine: InferenceEngine,
        config: Optional[BatchConfig] = None,
        drift_detector: Optional[DriftDetector] = None,
    )
    def run_batch(self, texts: List[str], task_types: Optional[List[str]] = None) -> Dict[str, Any]
    def run_from_file(self, path: str | Path) -> Dict[str, Any]
    def save_outputs(self, results: Dict[str, Any])
```

#### `run_batch()` flow

```
chunk texts into windows of config.batch_size
for chunk in chunks:
    outputs = engine.predict_batch(chunk, task_types)
    accumulate logits, probs, predictions per task

if run_drift_detection and drift_detector.baseline set:
    drift_results = drift_detector.detect(probabilities=accumulated_probs)
    merge into output

return {
    "task": {
        "logits": np.ndarray(N, n_classes),
        "probabilities": np.ndarray(N, n_classes),
        "predictions": np.ndarray(N,),
    },
    "drift": drift_results | None,
    "n_samples": N,
}
```

#### `run_from_file(path)`

Reads a plain-text file (one article per line, UTF-8), calls `run_batch`, and returns the same structure.

#### `save_outputs(results)`

Saves per-task `.npy` arrays for logits and probabilities to `config.output_dir`.

---

## 16. Inference Monitor (`monitoring.py`)

Thread-safe rolling-window monitor. Tracks latency, prediction confidence, and entropy. Emits `logging.WARNING` alerts when rolling means exceed configured thresholds.

### `MonitoringConfig`

```python
@dataclass
class MonitoringConfig:
    window_size: int = 500
    latency_ms_threshold: float = 500.0
    confidence_threshold: float = 0.4
    entropy_threshold: float = 1.5
```

### `MetricWindow`

```python
class MetricWindow:
    def __init__(self, size: int)    # backed by collections.deque(maxlen=size)
    def add(self, value: float)
    def mean(self) -> float
    def p95(self) -> float
    def max(self) -> float
    def size(self) -> int
```

Statistics are computed with `numpy` on the deque snapshot. All methods return `0.0` on an empty window.

### `InferenceMonitor`

```python
class InferenceMonitor:
    def __init__(self, config: Optional[MonitoringConfig] = None)
    def update(self, *, latency_ms: float, confidence: Optional[float] = None,
               probabilities: Optional[np.ndarray] = None, error: bool = False)
    def snapshot(self) -> Dict[str, Any]
    def reset(self)
```

All methods acquire `self._lock` (threading.Lock) before mutating shared state.

#### `update()` — entropy computation

```python
entropy = -np.sum(probs * np.log(probs + 1e-12))
self.entropy.add(entropy)
```

EPS (`1e-12`) prevents `log(0)`.

#### Alert thresholds (checked on every `update()`)

| Condition | Log message |
|---|---|
| `latency.mean() > latency_ms_threshold` | `"High latency detected"` |
| `confidence.mean() < confidence_threshold` | `"Confidence drop detected"` |
| `entropy.mean() > entropy_threshold` | `"Uncertainty spike detected"` |

#### `snapshot()` output

```python
{
    "latency_mean_ms":  float,
    "latency_p95_ms":   float,
    "latency_max_ms":   float,
    "confidence_mean":  float,
    "entropy_mean":     float,
    "entropy_p95":      float,
    "error_rate":       float,
    "total_requests":   int,
}
```

---

## 17. Drift Detector (`drift_detection.py`)

Detects distribution shift between a stored baseline and a current batch of predictions using four statistical measures.

### `DriftConfig`

```python
@dataclass
class DriftConfig:
    kl_threshold: float = 0.1
    js_threshold: float = 0.1
    psi_threshold: float = 0.2
    wasserstein_threshold: float = 0.1
```

### Core statistical functions (module-level)

| Function | Signature | Description |
|---|---|---|
| `kl_divergence(p, q)` | `(array, array) → float` | KL divergence using `scipy.stats.entropy`; both arrays normalized with EPS |
| `js_divergence(p, q)` | `(array, array) → float` | Jensen-Shannon divergence via midpoint mixture |
| `population_stability_index(expected, actual, bins=10)` | `(array, array) → float` | PSI via percentile breakpoints |

### `DriftDetector`

```python
class DriftDetector:
    def __init__(self, config: Optional[DriftConfig] = None)
    def set_baseline(self, *, probabilities: Dict[str, np.ndarray],
                     entropy_values: Optional[Dict[str, np.ndarray]] = None)
    def detect(self, *, probabilities: Dict[str, np.ndarray]) -> Dict[str, Any]
    def detect_entropy_drift(self, *, entropy_values: Dict[str, np.ndarray]) -> Dict[str, Any]
```

#### `set_baseline()`

Stores baseline probability distributions per task. Call once after warm-up or after a reference batch is run. Raises nothing if entropy_values is omitted; entropy drift detection simply skips those tasks.

#### `detect()` — per-task output

```python
{
  "emotion": {
    "kl_divergence": float,
    "js_divergence": float,
    "psi": float,
    "wasserstein": float,
    "drift_detected": bool,
  },
  ...
}
```

`drift_detected = True` when **any** of the four metrics exceeds its configured threshold (OR logic). OR logic is intentional: any single signal warrants investigation.

#### `detect_entropy_drift()` — per-task output

```python
{
  "emotion": {
    "psi": float,
    "wasserstein": float,
    "drift_detected": bool,
  },
  ...
}
```

#### Usage pattern

```python
detector = DriftDetector()
# After warm-up batch:
detector.set_baseline(probabilities={"emotion": probs_warmup})
# After production batch:
drift_report = detector.detect(probabilities={"emotion": probs_live})
if drift_report["emotion"]["drift_detected"]:
    alert(...)
```

---

## 18. Inference Logger (`inference_logger.py`)

Structured audit logger for every prediction event. Emits JSON lines by default. Supports optional vector logging (truncated to 10 elements) and automatic high-entropy alerts.

### `InferenceLogEntry` (dataclass)

| Field | Type | Description |
|---|---|---|
| `article_id` | `str` | UUID or caller-supplied identifier |
| `trace_id` | `str` | Fresh UUID per log call (correlates with distributed tracing) |
| `processing_time_ms` | `float` | Wall-clock inference latency |
| `model_versions` | `Dict[str, str]` | Map of component → version string |
| `feature_count` | `int` | Number of auxiliary features prepared |
| `predicted_label` | `Any \| None` | Final classification label |
| `prediction_confidence` | `float \| None` | Max class probability |
| `probabilities` | `Any \| None` | Truncated to first 10 elements if log_vectors=True |
| `logits` | `Any \| None` | Truncated to first 10 elements if log_vectors=True |
| `entropy` | `float \| None` | Per-prediction entropy |
| `p95_entropy` | `float \| None` | Rolling p95 entropy from monitor |
| `timestamp` | `float` | Unix timestamp |

### `InferenceLogger`

```python
class InferenceLogger:
    def __init__(self, service_name: str = "truthlens-inference",
                 enable_json_logs: bool = True, log_vectors: bool = False)
    def generate_article_id(self) -> str
    def generate_trace_id(self) -> str
    def start_timer(self) -> float
    def stop_timer(self, start_time: float) -> float
    def create_log_entry(self, *, article_id, start_time, model_versions,
                         feature_count, ...) -> InferenceLogEntry
    def log(self, entry: InferenceLogEntry, level=logging.INFO)
    def log_prediction(self, *, start_time, model_versions, feature_count, ...)
```

#### Vector logging behaviour

When `log_vectors=False` (default), `probabilities` and `logits` are set to `None` before writing, avoiding large payloads in production logs. When `log_vectors=True`, both arrays are truncated to `MAX_VECTOR_LOG = 10` elements via `_truncate_vector()`.

#### Auto-alerting

Inside `log_prediction()`, if both `entropy` and `p95_entropy` are provided and `entropy > p95_entropy`, a `WARNING` is emitted automatically:

```
"High uncertainty detected"
```

#### Emitted JSON line format

```json
{
  "service": "truthlens-inference",
  "event": "inference",
  "article_id": "uuid",
  "trace_id": "uuid",
  "processing_time_ms": 45.2,
  "model_versions": {"encoder": "roberta-base", "head": "v3"},
  "feature_count": 38,
  "predicted_label": "fake",
  "prediction_confidence": 0.91,
  "probabilities": null,
  "logits": null,
  "entropy": 0.31,
  "p95_entropy": 1.2,
  "timestamp": 1746000000.0
}
```

---

## 19. Result Formatter and Report Generator

### `result_formatter.py`

#### Response dataclasses

Three output shapes are supported. All use `dataclasses.asdict()` for serialisation.

##### `TruthLensAPIResponse`

Compact shape for live API consumers.

```python
@dataclass
class TruthLensAPIResponse:
    predictions: Dict[str, Any]        # task → argmax label index
    confidence: Dict[str, float]       # task → max prob
    uncertainty: Optional[Dict[str, float]]
    credibility_score: Optional[float]
    graph: Optional[Dict[str, Any]]
    graph_explanation: Optional[Dict[str, Any]]
    drift: Optional[Dict[str, Any]]
    monitoring: Optional[Dict[str, Any]]
    timestamp: str
```

##### `TruthLensDashboardReport`

Extended shape for dashboards, includes evaluation and calibration blocks.

##### `TruthLensResearchExport`

Full shape: includes raw logits, raw probabilities, intermediate features, model metadata, task correlation.

#### `ResultFormatter`

```python
class ResultFormatter:
    def format_api_response(self, report: Dict[str, Any]) -> Dict[str, Any]
    def format_dashboard_report(self, report: Dict[str, Any]) -> Dict[str, Any]
    def format_research_export(self, report: Dict[str, Any],
                               model_metadata=None, features=None) -> Dict[str, Any]
    def to_json(self, data: Dict[str, Any], pretty: bool = True) -> str
```

All three `format_*` methods extract optional `graph`, `graph_explanation`, `drift`, and `monitoring` keys from both the top-level report and from `report["analysis_modules"]` with `or`-fallback, ensuring backward compatibility with older report shapes.

---

### `report_generator.py`

#### `ArticleSummary` (dataclass)

```python
@dataclass
class ArticleSummary:
    title: Optional[str]
    source: Optional[str]
    word_count: Optional[int]
    analyzed_at: str
```

#### `ReportConfig` (dataclass)

```python
@dataclass
class ReportConfig:
    include_timestamp: bool = True
    pretty_json: bool = True
    validate_fields: bool = True
    save_explanation_artifacts: bool = False
    explanation_output_dir: str = "reports/explanations"
    include_evaluation: bool = True
    include_uncertainty: bool = True
    include_calibration: bool = True
```

#### `ReportGenerator`

```python
class ReportGenerator:
    def __init__(self, config: Optional[ReportConfig] = None,
                 aggregation_pipeline: Optional[AggregationPipeline] = None)
    def generate_report(self, *, article_text, title=None, source=None,
                        analysis=None, predictions=None, evaluation=None,
                        calibration=None, uncertainty=None,
                        task_correlation=None, explainability=None,
                        article_id=None) -> Dict[str, Any]
    def to_json(self, report: Dict[str, Any]) -> str
    def save_json(self, report: Dict[str, Any], path: str)
```

#### `generate_report()` — key design decisions

**REC-4** (aggregation must be pre-computed): If `analysis["profile"]` is present but `analysis["aggregation"]` is absent, the method raises `ValueError` rather than silently re-running `AggregationPipeline`. This prevents duplicate aggregation work on the hot path and ensures caller-level caching is respected.

**Report structure**

```python
{
    "article_summary": {...},
    "predictions": {...},
    "analysis": {
        "bias_features": {...},
        "emotion_features": {...},
        "narrative_features": {...},
        "graph_features": {...},
        "analysis_modules": {...},
    },
    "aggregation": {...},
    "evaluation": {...},
    "calibration": {...},
    "uncertainty": {...},
    "task_correlation": {...},
    "explainability": {...},
    "graph": {...},
    "graph_explanation": {...},
    "drift": {...},
    "monitoring": {...},
    "metadata": {
        "report_version": "v3",
        "generated_at": "2026-05-02T...",
        "tasks": ["emotion", "narrative", ...],
    },
    "risk_level": "high" | "normal",
}
```

**Risk flag**: `"risk_level": "high"` is appended when `uncertainty["mean_entropy"] > 1.5`.

**Explainability artifact save**: When `config.save_explanation_artifacts=True` and `article_id` is provided, `ExplanationReportGenerator.generate()` is called and the artifact paths are stored under `report["explainability_artifacts"]`.

---

## 20. Article Analyzer and CLI

### `analyze_article.py` — `ArticleAnalyzer`

The top-level orchestrator that assembles every subsystem into a single `analyze(text)` call.

```python
@dataclass
class ArticleAnalyzer:
    feature_pipeline: FeaturePipeline
    entity_graph_builder: EntityGraphBuilder
    graph_analyzer: GraphAnalyzer
    profile_builder: BiasProfileBuilder
    score_calculator: TruthLensScoreCalculator

    narrative_graph_builder: Optional[NarrativeGraphBuilder] = None
    graph_pipeline: Optional[GraphPipeline] = None
    analysis_runner: Optional[AnalysisIntegrationRunner] = None
    aggregation_pipeline: Optional[AggregationPipeline] = None
    prediction_service: Optional[PredictionService] = None
    report_generator: Optional[ReportGenerator] = None
    explanation_report_generator: Optional[ExplanationReportGenerator] = None
    predict_fn: Optional[Callable[[str], Dict[str, Any]]] = None
```

#### `__post_init__()` — optional dependency construction

Non-required fields that are `None` on construction are filled with defaults:

- `NarrativeGraphBuilder()` (default)
- `get_default_pipeline()` — process-wide `GraphPipeline` singleton
- `AnalysisIntegrationRunner()`
- `AggregationPipeline()`
- `ReportGenerator()`
- `ExplanationReportGenerator()`

**`PredictionService`** construction is wrapped in a `try/except` (**REC-1** pattern): if the model checkpoint is unavailable, `prediction_service` remains `None` and analysis continues without model predictions.

```python
try:
    _settings = load_settings()
    self.prediction_service = PredictionService(
        engine=InferenceEngine(
            InferenceConfig(model_path=str(_settings.model.path), device="auto",
                            enable_full_pipeline=False)
        )
    )
except Exception:
    self.prediction_service = None
```

#### `analyze(text)` — full pipeline

```
ensure_non_empty_text(text)
context = FeatureContext(text=text)

fused_features = feature_pipeline.extract(context)
feature_sections = _extract_feature_sections(fused_features)
    # splits keys by prefix: bias_*, emotion_*, narrative_*, discourse_*

entity_graph = entity_graph_builder.build_graph(text)
graph_features = entity_graph_builder.extract_graph_features(entity_graph)
graph_metrics = graph_analyzer.analyze(entity_graph)

analysis_modules = analysis_runner.analyze_text(text)

profile = profile_builder.build_profile(
    bias_features, emotion_features, narrative_features, discourse_features,
    ideology_predictions={},
)

aggregation_output = aggregation_pipeline.run(profile, text=text,
                                               analysis_modules=analysis_modules)
scores = aggregation_output.get("raw_scores") or score_calculator.compute_scores(profile)

prediction_output = _run_prediction(text)   # may be {} if service is None

return {
    "text": text,
    "bias_features": ..., "emotion_features": ...,
    "narrative_features": ..., "discourse_features": ...,
    "graph_features": {**graph_features, **graph_metrics},
    "entity_graph": entity_graph,
    "analysis_modules": analysis_modules,
    "profile": profile,
    "scores": scores,
    "aggregation": aggregation_output,
    "label": ..., "confidence": ..., "fake_probability": ...,
    "prediction_raw": raw_pred,
}
```

**REC-1**: `_run_prediction()` surfaces only `{label, confidence, fake_probability, raw_output}` from `PredictionService.predict()`. It does not attempt to extract `graph`, `graph_explanation`, `drift`, or `monitoring` from the prediction blob — those keys are not produced by `PredictionService` and were previously `None`-polluting the report.

---

### `run_inference.py` — CLI harness

#### Entry point

```bash
python -m src.inference.run_inference \
    --model_dir /models/truthlens_v1 \
    --article "Breaking: Government implements new policy..." \
    --output_format api \
    --evaluate \
    --labels_file labels.json \
    --save_logits \
    --save_probs \
    --save_uncertainty \
    --save_dir outputs/run_001
```

#### CLI flags

| Flag | Type | Description |
|---|---|---|
| `--model_dir` | `str` (required) | Path to the model checkpoint directory |
| `--article` | `str` | Single article string (mutually exclusive with `--input_file`) |
| `--input_file` | `str` | Path to a plain-text file (one article per line) |
| `--labels_file` | `str` | JSON file mapping task names → integer label arrays for evaluation |
| `--evaluate` | flag | Triggers calibration + uncertainty + correlation computation |
| `--output_format` | `api \| dashboard \| research` | Controls `ResultFormatter` method |
| `--save_logits` | flag | Saves per-task `{task}_logits.npy` |
| `--save_probs` | flag | Saves per-task `{task}_probabilities.npy` |
| `--save_uncertainty` | flag | Saves `uncertainty.json` |
| `--save_dir` | `str` | Output directory (default `"outputs"`) |

#### Execution flow

```
load_texts(args) → List[str]
InferenceEngine(InferenceConfig(model_path, device="auto"))
engine.predict_for_evaluation(texts) → outputs

if --evaluate and --labels_file:
    for task:
        compute_calibration(logits, y_true, task_type="multiclass")
        uncertainty_statistics(probs)
    compute_task_correlation(probs_by_task)

optionally: save_arrays / save_uncertainty

ReportGenerator.generate_report(...)
ResultFormatter.format_*(report)
print(json.dumps(final, indent=2))
save_report(report, save_dir / "report.json")
```

On any exception: `logger.exception("Inference failed")` → `sys.exit(1)`.

---

## 21. Public Package API (`__init__.py`)

The following 14 symbols are re-exported from `src.inference` for external consumers:

```python
from src.inference import (
    InferenceCache,
    InferenceCacheConfig,
    EngineInferenceConfig,      # aliased to avoid name collision
    InferenceEngine,
    InferenceLogEntry,
    InferenceLogger,
    FeaturePreparer,
    FeaturePreparationConfig,
    ModelArtifacts,
    ModelLoader,
    PredictionPipeline,
    PredictionPipelineConfig,
    ReportConfig,
    ReportGenerator,
    ResultFormatter,
)
```

`InferenceConfig` is re-exported as `EngineInferenceConfig` to prevent shadowing when callers also import from `src.inference.inference_config` (**CRIT-1** prevention at the package boundary).

Symbols not in `__all__` (e.g. `PredictionService`, `PostProcessor`, `InferenceMonitor`, `DriftDetector`, `BatchInferenceEngine`) are considered semi-internal and accessed via their own module paths.

---

## 22. Cross-Cutting Concerns

### Threading and concurrency

| Component | Mechanism |
|---|---|
| `predict_api._get_service()` | `threading.Lock` + double-checked locking |
| `InferenceCache` (disk writes) | Per-key reentrant lock (single-flight, **LAT-5**) |
| `InferenceMonitor.update / snapshot / reset` | `threading.Lock` |

### Error handling strategy

| Layer | Behaviour |
|---|---|
| `InferenceEngine.predict` | Propagates all exceptions to caller |
| `PredictionService.predict` | Propagates; caller (predict_api) catches and returns 500 |
| `ArticleAnalyzer._run_prediction` | Swallows exceptions, returns `{}`, logs WARNING |
| `InferenceLogger.log` | Swallows serialisation exceptions, logs with `logger.exception` |
| `ReportGenerator.generate_report` (explainability save) | Swallows, logs WARNING |
| `run_inference.main()` | Catches all exceptions, logs, exits 1 |

### Known design constraints and fixes

| Tag | File | Description |
|---|---|---|
| **CRIT-1** | `inference_config.py` | Single `InferenceConfig` dataclass; loader re-exports engine's class |
| **CFG-7** | `inference_config.py` | Missing YAML keys skip type checks rather than raising; engine defaults apply |
| **CFG-2/5/6** | `constants.py` | Batch size, cache version, report version declared once and imported everywhere |
| **PP-3** | `postprocessing.py` | `process()` iterates `logits.keys()` not `label_maps.keys()` |
| **LAT-5** | `inference_cache.py` | Single-flight per-key lock prevents cache stampede |
| **REC-1** | `analyze_article.py` | `_run_prediction()` surfaces only available PredictionService keys |
| **REC-4** | `report_generator.py` | Aggregation must be pre-computed; re-computing in report raises ValueError |

### Deployment configuration

```toml
# .replit
[deployment]
build = "pip install 'torch>=2.1,<3.0' --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt"
run   = "gunicorn --workers 1 --timeout 120 -k uvicorn.workers.UvicornWorker api.app:app"
```

- CPU-only torch wheel keeps the image under the 8 GiB deploy limit.
- Single worker avoids duplicate model loads in memory.
- 120-second timeout accommodates first-request model warm-up.

### Performance guidelines

| Scenario | Recommendation |
|---|---|
| Single article, latency-critical | `PredictionService.predict` (cache-first) |
| Bulk offline scoring | `BatchInferenceEngine.run_batch` with `batch_size=64` |
| Research / evaluation | `InferenceEngine.predict_for_evaluation` → `run_inference.py` CLI |
| Full analysis with explainability | `ArticleAnalyzer.analyze` |
| Drift monitoring | Set baseline after warmup; call `DriftDetector.detect` every N batches |
