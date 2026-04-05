# TruthLens AI

## Project Overview
TruthLens AI is a multi-layer AI platform for misinformation detection and news credibility analysis. It combines deep linguistic analysis, narrative extraction, propaganda detection, and graph-based reasoning to provide an interpretable "Credibility Score."

## Architecture
- **Backend**: FastAPI REST API (`api/app.py`) served via Uvicorn
- **Language**: Python 3.12
- **ML/NLP**: PyTorch, Hugging Face Transformers, spaCy, NLTK, LIME, SHAP
- **Port**: 5000

## Key API Endpoints
- `GET /` — Health check, lists all endpoints
- `GET /health` — Detailed health check (model availability)
- `POST /predict` — Predict fake/real for news text
- `POST /analyze` — Full analysis with bias, emotion, and explainability
- `GET /project-view` — Project structure and configuration info
- `GET /docs` — Interactive Swagger API documentation

## Project Structure
```
api/          - FastAPI application
src/          - Core source code
  aggregation/  - Credibility score calculation
  analysis/     - Bias, narrative, propaganda analysis
  features/     - Feature engineering (lexical, semantic, etc.)
    bias/       - Bias detection features
    emotion/    - Emotion analysis features
  models/       - Model implementations
  inference/    - Inference logic
  explainability/ - SHAP/LIME explanations
  graph/        - Entity/narrative graph analysis
  utils/        - Configuration, logging, utilities
models/       - Trained model artifacts (created during training)
  inference/  - Predictor module (predict, predict_batch)
  registry/   - Model registry
config/       - YAML configuration files
tests/        - Pytest test suite
```

## Important Notes
- Model must be trained before `/predict` and `/analyze` work fully
- Health endpoint shows "degraded" when no model is trained — this is expected
- `src/features/bias/bias_lexicon.py` and `src/features/emotion/emotion_lexicon.py` are wrapper modules created during import setup
- `models/inference/predictor.py` contains both `predict` and `predict_batch` functions
- All `src/` subdirectories have `__init__.py` files for Python package imports

## Running
The app runs via workflow: `python -m uvicorn api.app:app --host 0.0.0.0 --port 5000 --reload --reload-dir api --reload-dir src --reload-dir config --reload-dir models`
(reload dirs are scoped to avoid re-triggering on `.pythonlibs` package installs)

## Configuration
Main config: `config/config.yaml` — model paths, training params, API settings

## Bug Fixes Applied
- **`src/utils/config_loader.py`** — Fixed `load_app_config` to read from `data` (not `dataset`) key matching the actual config.yaml schema; fixed model name path to read from `model.encoder.name`; made `experiment` section optional with sensible defaults (it doesn't exist in config.yaml).
- **`models/inference/predictor.py`** — Added module-level model caching so the model is only loaded once per process instead of on every request; added `model.eval()` after loading; added device-aware input handling so tokenizer tensors are moved to the same device as the model before inference; moved `.cpu()` call to results to ensure safe return.
- **`src/features/emotion/emotion_lexicon.py`** — Fixed dominant emotion detection to correctly return `"neutral"` when all emotion scores are zero (previously returned the first emotion label arbitrarily).
- **`src/features/bias/bias_lexicon.py`** — Removed unused `field` import from `dataclasses`.
