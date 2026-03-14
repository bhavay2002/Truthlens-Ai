# TruthLens AI Architecture

## 1. Purpose

TruthLens AI is an end-to-end fake news detection system that:
- trains a RoBERTa classifier (`REAL` vs `FAKE`),
- enriches input text with engineered feature tokens,
- evaluates model quality and saves reports,
- serves online inference through FastAPI.

Primary entry points:
- Training pipeline: `main.py`
- Evaluation runner: `evaluate.py`
- EDA runner: `run_eda.py`
- Inference API: `api/app.py`

## 2. System Context

### Inputs
- Raw datasets from:
  - `data/raw/isot/`
  - `data/raw/liar_dataset/`
  - `data/raw/FakeNewsNet/`
- Runtime configuration from `config/config.yaml`

### Outputs
- Trained model + tokenizer in `models/roberta_model/`
- TF-IDF vectorizer in `models/tfidf_vectorizer.joblib`
- Processed/temporary datasets under `data/interim/` and `data/processed/`
- Reports and visual artifacts under `reports/`
- API predictions (`label`, `fake_probability`, `confidence`)

## 3. High-Level Architecture

```text
Raw Data Sources (ISOT + LIAR + FakeNewsNet)
                |
                v
        Merge Datasets (src/data/merge_datasets.py)
                |
                v
        Clean + Validate (src/data/clean_data.py, src/data/validate_data.py)
                |
                +--------------------------+
                |                          |
                v                          v
      Training/Eval Path            EDA Path (run_eda.py)
                |                    -> src/data/eda.py
                v
    Leakage-Safe Split (train/val/test)
                |
                v
      Train-only Augmentation
                |
                v
 Feature Pipeline (fit train, transform val/test)
  - source_features
  - metadata_features
  - text_features (TF-IDF)
                |
                v
      RoBERTa Training (HF Trainer)
                |
                v
     Metrics + Reports + Confusion Matrix
                |
                v
         Saved Artifacts (model/vectorizer/reports)
                |
                v
   Serving: FastAPI -> src/models/predict.py -> model output
```

## 4. Component View

### 4.1 Orchestration Layer
- `main.py`
  - Coordinates the full training lifecycle.
  - Creates required directories.
  - Runs merge, cleaning, split, augmentation, feature engineering, optional CV/tuning, final training, and evaluation.

### 4.2 Data Layer
- `src/data/merge_datasets.py`
  - Merges records from ISOT, LIAR, and FakeNewsNet into a common schema.
- `src/data/clean_data.py`
  - Normalizes text and removes unusable rows.
- `src/data/validate_data.py`
  - Validates structure/quality constraints.
- `src/data/data_augmentation.py`
  - Expands only the train split when enabled.
- `src/data/eda.py`
  - Generates analysis summaries and figures.

### 4.3 Feature Engineering Layer
- `src/features/feature_pipeline.py`
  - Builds `engineered_text` by composing multiple feature families.
  - Persists TF-IDF vectorizer for inference-time consistency.
- `src/features/source_features.py`
  - Adds source/domain credibility signals.
- `src/features/metadata_features.py`
  - Adds structural metadata features.
- `src/features/text_features.py`
  - TF-IDF extraction and top-term token generation.

### 4.4 Model Layer
- `src/models/train_roberta.py`
  - Tokenization, Trainer setup, training, and model serialization.
- `src/models/predict.py`
  - Lazy-loads model/tokenizer and optional TF-IDF vectorizer.
  - Applies the same text preparation strategy used during training.

### 4.5 Training Utilities Layer
- `src/training/cross_validation.py`
  - Optional stratified CV over dataframe-based flow.
- `src/training/hyperparameter_tuning.py`
  - Optional parameter search (Optuna or fallback logic).

### 4.6 Evaluation + Visualization Layer
- `src/evaluation/evaluate_model.py`
  - Computes classification metrics and serializes results.
- `src/visualization/visualize.py`
  - Confusion matrix plotting.
- `evaluate.py`
  - Standalone evaluation against saved model + test set.

### 4.7 Serving Layer
- `api/app.py`
  - REST API endpoints:
    - `GET /`
    - `GET /health`
    - `POST /predict`
  - Input validation via Pydantic request/response models.
  - Calls `src/models/predict.py` for inference.

### 4.8 Configuration + Utilities Layer
- `config/config.yaml`
  - Canonical runtime knobs for model, training, features, data, and paths.
- `src/utils/settings.py`
  - Typed dataclass mapping + cached settings loader.
- `src/utils/logging_utils.py`
  - Centralized logging setup.
- `src/utils/input_validation.py`
  - Shared guards for dataframe and input correctness.

## 5. Runtime Flows

### 5.1 Training Flow (`python main.py`)
1. Load typed settings from YAML.
2. Merge raw datasets.
3. Clean and validate data.
4. Save cleaned dataset and cleaning report.
5. Split into train/validation/test (leakage-safe).
6. Apply augmentation only on train split (optional).
7. Fit feature pipeline on train, transform val/test.
8. Save TF-IDF vectorizer.
9. Run optional cross-validation on train split.
10. Run optional hyperparameter tuning on train+val.
11. Train final RoBERTa model.
12. Evaluate on test split and save metrics/plots.

### 5.2 Inference Flow (`POST /predict`)
1. API validates request payload (`text`, min length constraints).
2. Prediction module lazy-loads tokenizer/model.
3. If configured for engineered text, load vectorizer and transform text.
4. Run model forward pass and softmax.
5. Return normalized response (`prediction`, `fake_probability`, `confidence`).

### 5.3 Offline Evaluation Flow (`python evaluate.py`)
1. Load persisted model and tokenizer.
2. Load persisted test set.
3. Infer predictions row-by-row.
4. Compute and save metrics + confusion matrix artifact.

### 5.4 EDA Flow (`python run_eda.py`)
1. Merge raw datasets.
2. Run EDA analysis class.
3. Save plots and JSON summary report.

## 6. Data and Artifact Contracts

### Datasets
- Merged dataset: `data/interim/merged_dataset.csv`
- Cleaned dataset: `data/processed/cleaned_dataset.csv`
- Test set: `data/processed/test_set.csv`

### Model Artifacts
- Model directory: `models/roberta_model/`
- Vectorizer: `models/tfidf_vectorizer.joblib`

### Reporting Artifacts
- Cleaning report: `reports/data_cleaning_report.json`
- Evaluation report: `reports/evaluation_results.json`
- Confusion matrix: `reports/confusion_matrix.png`
- EDA report: `reports/eda_report.json`
- EDA figures: `reports/figures/`

## 7. Key Architecture Decisions

1. Leakage-safe preprocessing order:
- Split occurs before augmentation and feature fitting.
- Feature pipeline is fit on train only, then reused for val/test.

2. Typed centralized settings:
- All runtime behavior is controlled via YAML + dataclass mapping.

3. Training-serving feature parity:
- Inference reuses saved TF-IDF vectorizer and feature pipeline logic.

4. Lazy model loading for API:
- Improves startup behavior and avoids repeated load overhead.

5. Optional advanced workflows:
- Cross-validation and hyperparameter tuning can be toggled without changing code paths.

## 8. Operational Interfaces

### CLI Commands
- Train: `python main.py`
- Evaluate: `python evaluate.py`
- Run EDA: `python run_eda.py`
- Serve API: `uvicorn api.app:app --reload`

### API Endpoints
- `GET /` basic service status
- `GET /health` model readiness and file completeness check
- `POST /predict` prediction endpoint

## 9. Testing Surface

Tests are organized under `tests/` and currently include:
- Feature pipeline + validation checks
- Settings and utilities checks
- Training utility checks (CV/tuning)
- Smoke/API tests

This gives baseline coverage for the critical training and serving integration points.
