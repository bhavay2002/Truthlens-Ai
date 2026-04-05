# TruthLens AI

## Project Overview
TruthLens AI is a multi-layer AI platform for misinformation detection and news credibility analysis. It combines deep linguistic analysis, narrative extraction, propaganda detection, and graph-based reasoning to provide an interpretable "Credibility Score."

## Architecture
- **Backend**: FastAPI REST API (`api/app.py`) served via Uvicorn
- **Language**: Python 3.12
- **ML/NLP**: PyTorch, Hugging Face Transformers, spaCy, NLTK, LIME, SHAP
- **Port**: 5000

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
