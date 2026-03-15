# TruthLens AI Repository Structure

This document is the canonical map of the project layout. It explains where code lives, where artifacts are generated, and where to add new files without creating drift.

## 1) Root Layout (Curated)

```text
Truthlens Ai/
|-- .github/
|   `-- workflows/
|       `-- ci.yml
|-- api/
|   `-- app.py
|-- config/
|   `-- config.yaml
|-- data/
|   |-- raw/
|   |   |-- isot/
|   |   |-- liar_dataset/
|   |   |-- FakeNewsNet/
|   |   `-- dataset.py
|   |-- interim/
|   |   `-- merged_dataset.csv
|   `-- processed/
|       |-- cleaned_dataset.csv
|       `-- test_set.csv
|-- models/
|   |-- roberta_model/
|   `-- 1/                           # currently empty (legacy/placeholder)
|-- logs/
|   `-- training.log
|-- reports/
|   |-- evaluation_results.json
|   |-- data_cleaning_report.json
|   |-- confusion_matrix.png
|   |-- eda_report.json
|   |-- eda_summary.json
|   |-- figures/
|   `-- _test_tmp/
|-- src/
|   |-- data/
|   |-- evaluation/
|   |-- explainability/
|   |-- features/
|   |-- models/
|   |-- training/
|   |-- utils/
|   `-- visualization/
|-- tests/
|   |-- test_smoke.py
|   |-- test_training_utils.py
|   |-- test_feature_pipeline_and_validation.py
|   `-- test_settings_and_utils.py
|-- architecture.md
|-- CONTRIBUTING.md
|-- Dockerfile
|-- docker-compose.yml
|-- evaluate.py
|-- KNOWLEDGE.md
|-- LICENSE
|-- main.py
|-- PROJECT_REVIEW.md
|-- QUICKSTART.md
|-- README.md
|-- requirements.txt
|-- run_eda.py
|-- setup.py
|-- source_scores.json
|-- structure.md
|-- test.py
|-- .env.example
`-- .gitignore
```

## 2) What Lives Where

- `src/`: all core Python implementation for data processing, features, training, evaluation, inference, utilities, and visualizations.
- `api/`: FastAPI service entrypoint (`app.py`) for online inference.
- `tests/`: automated tests for smoke checks, feature/validation behavior, settings/utilities, and training helpers.
- `config/`: runtime configuration (`config.yaml`) used by typed settings loader.
- `data/raw/`: source datasets (ISOT, LIAR, FakeNewsNet).
- `data/interim/`: temporary/merged data outputs from pipeline.
- `data/processed/`: cleaned/test datasets consumed by training/evaluation.
- `models/`: saved model artifacts and tokenizer files.
- `reports/`: evaluation and EDA outputs (JSON and plots).
- `logs/`: training/runtime logs.

## 3) Source vs Generated Files

Treat these as source-of-truth (edited by developers):

- `src/**`, `api/**`, `tests/**`, `config/config.yaml`
- root scripts/docs such as `main.py`, `evaluate.py`, `run_eda.py`, `README.md`

Treat these as generated/runtime artifacts (do not hand-edit unless debugging):

- `data/interim/*.csv`, `data/processed/*.csv`
- `models/roberta_model/*`
- `reports/*` (except if intentionally documenting example outputs)
- `logs/*`
- `__pycache__/`, `.pytest_cache/`

## 4) Module Ownership Map (`src/`)

- `src/data/`: load, merge, clean, validate, augment, and run EDA.
- `src/features/`: source/metadata/TF-IDF feature engineering and pipeline assembly.
- `src/models/`: RoBERTa training, model IO helpers, inference helpers.
- `src/training/`: cross-validation and hyperparameter tuning utilities.
- `src/evaluation/`: metric computation and reporting.
- `src/visualization/`: plots like confusion matrix and EDA charts.
- `src/explainability/`: SHAP/LIME explainers.
- `src/utils/`: config loading, typed settings, logging, validations, helper functions.

## 5) Entry Points

- `python main.py`: full training pipeline.
- `python evaluate.py`: evaluate saved model + test set.
- `python run_eda.py`: exploratory data analysis workflow.
- `uvicorn api.app:app --reload`: run inference API.
- `python -B -m pytest -q`: run test suite.

## 6) Placement Rules (Keep Structure Clean)

- New pipeline/data logic goes under `src/data/`.
- New engineered signals go under `src/features/`.
- New model/inference code goes under `src/models/`.
- New reusable support code goes under `src/utils/`.
- New tests must be added under `tests/` with `test_*.py` naming.
- New generated reports belong in `reports/`; never place generated files in `src/`.
- Large datasets and model binaries remain under `data/` and `models/`, not in docs or code folders.

## 7) Current State Notes

- `experiments/` and `notebooks/` exist but are currently empty.
- `models/1/` exists as an empty placeholder/legacy folder.
- `.env` exists locally; `.env.example` is the template that should stay in version control.
