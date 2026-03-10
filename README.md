# TruthLens AI

TruthLens AI is a fake-news detection project built around a RoBERTa classifier, with data cleaning, optional augmentation, engineered feature tokens, evaluation reporting, and a FastAPI inference service.

## What The Project Does

- Trains a binary classifier (`REAL` vs `FAKE`) using merged datasets.
- Applies NLP cleaning and optional data augmentation.
- Adds feature-engineered signal (metadata, source credibility, TF-IDF keyword tokens) into the model input text.
- Evaluates with classification metrics and confusion matrix output.
- Serves predictions through a REST API.
- Supports optional cross-validation and hyperparameter tuning from config.

## Current Capability Snapshot

- Training pipeline: `main.py`
- Model training/inference: `src/models/train_roberta.py`, `src/models/predict.py`
- Feature pipeline integration: `src/features/feature_pipeline.py`
- Optional CV: `src/training/cross_validation.py`
- Optional tuning: `src/training/hyperparameter_tuning.py` (Optuna backend if installed, fallback random search otherwise)
- API: `api/app.py`
- Evaluation: `evaluate.py` and `src/evaluation/evaluate_model.py`
- Typed settings/config management: `src/utils/settings.py`

## Architecture Overview

1. Data Ingestion Layer
- `src/data/merge_datasets.py` merges ISOT + LIAR + FakeNewsNet sources.

2. Data Quality Layer
- `src/data/clean_data.py` normalizes and filters text.
- `src/data/validate_data.py` validates schema, nulls, duplicates, and label distribution.

3. Feature Layer
- `src/features/feature_pipeline.py` combines:
  - source features (`src/features/source_features.py`)
  - metadata features (`src/features/metadata_features.py`)
  - TF-IDF signals (`src/features/text_features.py`)
- Produces `engineered_text` and persists TF-IDF vectorizer.

4. Training Layer
- `src/models/train_roberta.py` handles split/tokenize/train/save flow.
- `src/training/cross_validation.py` runs stratified CV against `train_model` compatible signatures.
- `src/training/hyperparameter_tuning.py` searches training params from dataframe inputs.

5. Evaluation Layer
- `src/evaluation/evaluate_model.py` computes metrics and saves report JSON.
- `src/visualization/visualize.py` plots confusion matrix.

6. Serving Layer
- `api/app.py` provides `/`, `/health`, and `/predict` endpoints.
- `src/models/predict.py` lazily loads model/tokenizer and predicts.

## Repository Layout

```text
Truthlens Ai/
  api/                  # FastAPI service
  config/               # YAML configuration
  data/                 # Raw/interim/processed datasets
  reports/              # Evaluation and EDA artifacts
  src/                  # Core application modules
  tests/                # Unit/integration tests
  main.py               # End-to-end training pipeline
  evaluate.py           # Standalone evaluation script
  run_eda.py            # EDA runner
  setup.py              # Environment setup helper
```

## Configuration

Primary config file: `config/config.yaml`

Important knobs:

- `model.*`: model name/path/max sequence length
- `training.*`: seed, epochs, batch size, learning rate
- `training.run_cross_validation`: enable/disable CV
- `training.run_hyperparameter_tuning`: enable/disable tuning
- `features.*`: TF-IDF settings for feature pipeline
- `data.*`: augmentation multiplier and dataset paths
- `paths.*`: output/log/report/model artifact paths

## Quick Start

### 1. Install

```bash
python -m venv venv
venv\Scripts\activate  # Windows PowerShell/CMD
pip install -r requirements.txt
```

### 2. Train

```bash
python main.py
```

### 3. Evaluate

```bash
python evaluate.py
```

### 4. Start API

```bash
uvicorn api.app:app --reload
```

### 5. Run Tests

```bash
python -B -m pytest -q
```

## API Example

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text":"Breaking news: Scientists discover new species in Amazon rainforest."}'
```

## Outputs

Main generated artifacts:

- Model: `models/roberta_model/`
- TF-IDF vectorizer: `models/tfidf_vectorizer.joblib`
- Training log: `logs/training.log`
- Evaluation JSON: `reports/evaluation_results.json`
- Confusion matrix: `reports/confusion_matrix.png`
- Cleaning report: `reports/data_cleaning_report.json`

## Documentation Map

- Quick usage: `QUICKSTART.md`
- Deep project knowledge and full file catalog: `KNOWLEDGE.md`
- Contribution process: `CONTRIBUTING.md`
- Current review/status notes: `PROJECT_REVIEW.md`

## License

MIT (see `LICENSE`).
