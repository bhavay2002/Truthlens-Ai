# TruthLens AI Architecture

Last updated: 2026-04-02

## 1. System Purpose

TruthLens AI is a modular NLP system for misinformation analysis that supports two primary operating modes:

1. Binary classifier workflow (`REAL` vs `FAKE`) for production prediction endpoints.
2. Unified multi-task dataset workflow for structured NLP task training and experimentation.

## 2. Canonical Unified Dataset Contract

The unified dataset contract used by schema utilities and dataset builders is:

- Text: `title`, `text`
- Classification tasks: `bias_label`, `ideology_label`, `propaganda_label`, `frame`
- Narrative frame multi-label: `CO`, `EC`, `HI`, `MO`, `RE`
- Narrative role extraction: `hero`, `villain`, `victim`
- Narrative entities: `hero_entities`, `villain_entities`, `victim_entities`
- Emotion multi-label: `emotion_0` ... `emotion_19`
- Metadata: `dataset`

## 3. High-Level Architecture

```text
Data Sources / Split CSVs
        |
        +--> Unified schema normalization
        |     (src/data/unified_label_schema.py)
        |
        +--> Data pipeline orchestration
        |     (src/pipelines/data_pipeline.py)
        |
        +--> Feature engineering
        |     (src/features/feature_pipeline.py)
        |
        +--> Training paths
        |     - Binary training: src/models/train_roberta.py
        |     - Multi-task model: src/models/multitask/multitask_truthlens_model.py
        |
        +--> Evaluation
        |     (src/evaluation/evaluate_model.py,
        |      src/evaluation/visualize_metrics.py)
        |
        +--> Serving
              (api/app.py -> src/models/predict.py)
```

## 4. Core Components

### 4.1 Data Layer

- `src/data/load_data.py`: CSV loading and schema normalization helpers.
- `src/data/validate_data.py`: dataset quality validation (schema/nulls/duplicates/class balance).
- `src/data/unified_label_schema.py`: canonical unified label normalization and validation.
- `src/data/clean_data.py`: text cleaning primitives used in pipelines.

### 4.2 Feature Layer

- `src/features/feature_pipeline.py`: source/metadata/TF-IDF + semantic token enrichment.
- `src/features/bias/*`, `src/features/emotion/*`, `src/features/narrative/*`, `src/features/discourse/*`: domain-specific feature extraction.

### 4.3 Model Layer

- `src/models/train_roberta.py`: HuggingFace training pipeline with configurable `label_column` compatibility.
- `src/models/predict.py`: single/batch prediction helpers with model-config-aware label decoding.
- `src/models/inference.py`: enriched inference path with explainability and auxiliary feature signals.
- `src/models/multitask/multitask_truthlens_model.py`: shared encoder + task heads including `frame`, narrative-frame multi-label, roles, and emotion outputs.

### 4.4 Pipeline Layer

- `src/pipelines/data_pipeline.py`: supports both fake/real paired files and direct unified dataset file mode.
- `src/pipelines/feature_pipeline.py`, `src/pipelines/emotion_pipeline.py`, `src/pipelines/truthlens_pipeline.py`: orchestration utilities.

### 4.5 Training Utilities

- `src/training/cross_validation.py`: stratified CV with configurable label column support.
- `src/training/hyperparameter_tuning.py`: Optuna/fallback tuning with configurable label column support.

### 4.6 Evaluation and Visualization

- `src/evaluation/evaluate_model.py`: binary and multiclass-safe metric computation.
- `src/evaluation/visualize_metrics.py` and `src/visualization/visualize.py`: confusion matrix and metric plots with dynamic class label support.

### 4.7 API Serving

- `api/app.py`: health and prediction endpoints for deployed model usage.

## 5. Runtime Flows

### 5.1 Binary Training Flow

1. Load config/settings.
2. Merge/load data.
3. Validate + clean.
4. Feature engineering (optional engineered text path).
5. Train model with `train_roberta`.
6. Evaluate and persist artifacts.

### 5.2 Unified Dataset Build Flow

1. Read split files per task.
2. Standardize to canonical columns.
3. Merge into unified split CSV.
4. Validate with unified schema utilities.

Current builder entry point in repo:
- `ztest3 copy.py`

### 5.3 Inference/API Flow

1. Validate request text.
2. Lazy-load model/tokenizer (and vectorizer when needed).
3. Predict + optional explainability hooks.
4. Return normalized response payload.

## 6. Artifacts

Common outputs:

- Model directory: `models/roberta_model/`
- Vectorizer: `models/tfidf_vectorizer.joblib`
- Evaluation report: `reports/evaluation_results.json`
- Confusion matrix: `reports/confusion_matrix.png`
- Unified dataset files:
  - `data/unified_dataset_train.csv`
  - `data/unified_dataset_validation.csv`
  - `data/unified_dataset_test.csv`

## 7. Design Notes

- Backward compatibility remains for legacy binary workflows.
- New compatibility layers resolve non-`label` training columns for unified datasets.
- Evaluation stack is now multiclass-safe while preserving binary metrics and ROC behavior where applicable.
- Some root-level `ztest*.py` scripts are transitional utilities and not the long-term package API.
