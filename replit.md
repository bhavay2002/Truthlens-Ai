# TruthLens AI

## Project Overview
TruthLens AI is a multi-layer AI platform for misinformation detection and news credibility analysis. It combines deep linguistic analysis, narrative extraction, propaganda detection, and graph-based reasoning to provide an interpretable "Credibility Score."

## Architecture
- **Backend**: FastAPI REST API (`api/app.py`) served via Uvicorn
- **Language**: Python 3.12
- **ML/NLP**: PyTorch, Hugging Face Transformers, spaCy, NLTK, LIME, SHAP
- **Port**: 5000

## Recent Refactors (audit fixes applied)
- **PERF-3**: All 14 singleton analyzers share one spaCy `en_core_web_sm` model via `get_shared_nlp()` in `src/analysis/_nlp.py`. All `disable_components` defaults unified to `()` so the cache key is always `("en_core_web_sm", ())`. Previous state: 4 separate pipeline instances.
- **ARCH-1**: `PredictionPipeline._compute_credibility_score()` and its dead private task methods (`_predict_bias`, `_predict_ideology`, `_predict_propaganda`, `_predict_emotion`) removed. Credibility computation is now exclusively owned by `AggregationPipeline`. `predict_with_aggregation()` reads `truthlens_credibility_score` directly from aggregation output.
- **ARCH-3**: `ExplainabilityLayer` (was in `prediction_pipeline.py`) and `explain_prediction_full`/`explain_fast` (was in `model_explainer.py`) consolidated into `ExplainabilityOrchestrator` in `src/explainability/orchestrator.py`. Single `explain()` method owns the full lifecycle: SHAP → LIME → bias/emotion → attention rollout → propaganda → aggregation → consistency. Backward-compat shims kept in both files.
- **CRIT-P2-1** (second-pass audit): `models/inference/predictor.py` `predict_batch` was collapsing N texts into 1 averaged result and returning a single dict — causing silent data loss and a `KeyError` crash in the `/batch-predict` fallback path. Fixed: per-sample tensor slicing after the batch forward pass; returns `[[real0,fake0], [real1,fake1], …]` as `app.py` expects.
- **CRIT-P2-2** (second-pass audit): `src/models/inference/predictor.py` `build_fake_real_output` called `probs.argmax(dim=-1).item()` on a (N,2) tensor — raises `RuntimeError` for N>1. Fixed: `probs.mean(dim=0)` collapses batch first; `argmax` and `max` are then safe for any batch size.

## Deep ML Systems Audit (April 2026)
- **Predictor key matching**: `_format_outputs` now uses `key.endswith("_logits")` instead of a loose `"logits" in key` substring match, so keys like `per_sample_logits_norm` are no longer mis-softmaxed.
- **Model wrapper precision**: Removed unconditional `.half()` and `torch.compile(mode="max-autotune")` side effects. Both are now opt-in via `use_half_precision` and `compile_mode` constructor kwargs. Autocast uses the device's actual `device.type` (not hardcoded `"cuda"`) with bf16/fp16 chosen by capability.
- **Trainer module-level side effects**: TF32/cuDNN precision flags moved into a `_configure_tf32()` helper called from `Trainer.__init__` (only when CUDA is present). Importing the module no longer mutates global PyTorch state. `torch.autograd.set_detect_anomaly(False)` removed.
- **Trainer per-step sync**: Replaced per-step `raw_loss.item()` accumulation with an on-device tensor accumulator (`loss_accum`) in both `_train_epoch` and `_validate_epoch`. CPU sync now happens once per epoch in the final return.
- **Propaganda explainer safety**: Removed `self.model.zero_grad()` on the shared singleton. Switched to `torch.autograd.grad` with per-sample softmax-max sum as the target. Eval/train mode preserved and restored. Duplicate merged tokens now accumulate instead of producing `_1/_2` suffix keys.
- **Emotion explainer isolation**: Integrated Gradients now runs on a local `copy.deepcopy` of the model, preventing any eval/train-state mutations from leaking into the serving singleton. Falls back to the shared model only if deepcopy fails.
- **Aggregation pipeline**: Removed the duplicated `compute_scores` call pattern (now one raw pass + one weighted pass). Applied weights logged at DEBUG. Pre-clip saturation warnings added so upstream scale bugs surface in logs instead of being silently clamped.
- **Predict API singleton**: Lazy loader now guarded by `threading.Lock` with double-checked locking, preventing duplicate model loads under concurrent request threads. Removed the `predict.batch_predict = batch_predict` function-attribute monkey-patch.
- **Explanation aggregator cleanup**: Dropped unused `_safe_corr` static method (dead code — `ExplanationConsistency` already handles correlation).
- **Missing model artifacts**: Generated `models/truthlens_model/config.json` (derived from `roberta-base` with Fake/Real `id2label`/`label2id`) and full tokenizer files (`vocab.json`, `merges.txt`, `tokenizer_config.json`, `special_tokens_map.json`) so the model registry boots cleanly. Actual `pytorch_model.bin` weights still need to be trained before `/predict` produces meaningful scores.

## Key API Endpoints

### Core prediction
- `POST /predict` — Predict fake/real for news text (1-hour memory cache)
- `POST /batch-predict` — Predict for up to 50 texts at once; cache-aware
- `POST /analyze` — Full deep analysis (bias, emotion, narrative, graph, LIME)
- `POST /report` — Lightweight structured report (bias + emotion + credibility)

### Calibration
- `GET /calibration/info` — Describe available calibration strategies (temperature scaling, isotonic regression)
- `POST /calibration/metrics` — Compute ECE, MCE, Brier Score, NLL from probability + label arrays

### Ensemble
- `GET /ensemble/info` — Describe available ensemble strategies
- `POST /ensemble/predict` — Combine probability vectors from multiple models (average / weighted_average / majority_vote)

### Export
- `GET /export/info` — Describe ONNX, TorchScript, and quantization options
- `POST /export/onnx` — Export loaded model to ONNX (requires trained model)
- `POST /export/torchscript` — Export loaded model to TorchScript (requires trained model)

### Inference meta
- `GET /inference/model-info` — InferenceEngine model metadata (params, device, label map)
- `POST /cache/clear` — Clear the in-memory prediction cache

### Utility
- `GET /` — Index; lists all available endpoints
- `GET /health` — Detailed health check (model availability, cache size)
- `GET /project-view` — Project structure and configuration info
- `GET /docs` — Interactive Swagger API documentation

## Project Structure
```
api/          - FastAPI application (app.py)
src/          - Core source code
  analysis/   - Bias, narrative, propaganda, discourse analysis
  features/   - Feature engineering (bias, emotion, lexical, semantic)
  models/
    calibration/  - CalibrationMetrics, TemperatureScaler, IsotonicCalibrator
    ensemble/     - EnsembleModel, WeightedEnsembleModel, StackingEnsembleModel
    export/       - ONNXExporter, TorchScriptExporter, QuantizationEngine
  inference/  - InferenceCache, InferenceEngine, InferenceLogger,
                ResultFormatter, ReportGenerator, FeaturePreparer,
                ModelLoader, PredictionPipeline
  explainability/ - SHAP/LIME explanations
  graph/      - Entity/narrative graph analysis
  utils/      - Configuration, logging, device, time, JSON utilities
models/       - Trained model artifacts
  inference/  - predictor.py (predict, predict_batch)
  registry/   - ModelRegistry
config/       - YAML configuration files
tests/        - Pytest test suite
```

## Integrated Subsystems (api/app.py singletons)

| Singleton | Class | Purpose |
|---|---|---|
| `INFERENCE_CACHE` | `InferenceCache` | 1-hour in-memory cache for /predict, /analyze, /report |
| `INFERENCE_LOGGER` | `InferenceLogger` | Structured JSON inference event logging |
| `RESULT_FORMATTER` | `ResultFormatter` | Validates /predict response schema |
| `REPORT_GENERATOR` | `ReportGenerator` | Powers /report endpoint |
| `CALIBRATION_METRICS` | `CalibrationMetrics` | Powers /calibration/metrics |
| `ONNX_EXPORTER` | `ONNXExporter` | Powers /export/onnx |
| `TORCHSCRIPT_EXPORTER` | `TorchScriptExporter` | Powers /export/torchscript |
| `_INFERENCE_ENGINE` | `InferenceEngine` | Lazy singleton for /batch-predict, /inference/model-info |
| `_QUANTIZATION_ENGINE` | `QuantizationEngine` | Lazy singleton (fbgemm backend guard) |

## Important Notes
- Model must be trained before `/predict`, `/analyze`, and export endpoints work fully
- Health endpoint shows "degraded" when no model is trained — this is expected
- The tokenizer requires `sentencepiece` or `protobuf` — a pre-existing environment issue
- `models/inference/predictor.py` contains `predict` and `predict_batch` (the RoBERTa backend)
- Export endpoints return 503 when InferenceEngine cannot be initialised (missing model)
- All `src/` subdirectories have populated `__init__.py` files

## Running
```
python -m uvicorn api.app:app --host 0.0.0.0 --port 5000 --reload \
  --reload-dir api --reload-dir src --reload-dir config --reload-dir models
```

## Configuration
Main config: `config/config.yaml` — model paths, training params, API settings

## Replit Migration Notes
The following files were created during Replit migration to resolve missing modules:
- `graph_hardening_patch.py` — Graph utility helpers (normalize_graph_adjacency, to_undirected, spectral_eigen_embedding, etc.) used by all `src/graph/` modules
- `src/models/ensemble/ensemble_model.py` — EnsembleModel / EnsembleConfig (average & majority-vote strategies)
- `src/models/ensemble/weighted_ensemble.py` — WeightedEnsembleModel / WeightedEnsembleConfig
- `src/models/ensemble/stacking_ensemble.py` — StackingEnsembleModel / StackingEnsembleConfig
