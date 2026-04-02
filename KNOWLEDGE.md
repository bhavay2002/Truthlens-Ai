# KNOWLEDGE.md

Last updated: 2026-04-02

## 1. Project Snapshot

TruthLens AI is currently a hybrid codebase with:

- production-oriented binary misinformation classification
- unified multi-task dataset support for richer NLP task experimentation
- modular analysis, graph, explainability, and scoring subsystems

Latest local test status: `78 passed`.

## 2. Canonical Unified Dataset Schema

Use this schema as the authoritative contract across new data/model work:

- Text: `title`, `text`
- Classification: `bias_label`, `ideology_label`, `propaganda_label`, `frame`
- Narrative frame multi-label: `CO`, `EC`, `HI`, `MO`, `RE`
- Narrative role extraction: `hero`, `villain`, `victim`
- Narrative entities: `hero_entities`, `villain_entities`, `victim_entities`
- Emotion multi-label: `emotion_0` ... `emotion_19`
- Metadata: `dataset`

Main schema utility:
- `src/data/unified_label_schema.py`

## 3. Key Execution Paths

### 3.1 Training (`main.py`)

1. Load settings/config.
2. Merge/load dataset(s).
3. Validate and clean.
4. Optional augmentation/feature engineering.
5. Optional CV and hyperparameter tuning.
6. Train/evaluate model and save artifacts.

### 3.2 Unified dataset build

Current utility script in repo root:
- `ztest3 copy.py`

This script standardizes split CSVs into canonical unified columns and outputs:
- `data/unified_dataset_train.csv`
- `data/unified_dataset_validation.csv`
- `data/unified_dataset_test.csv`

### 3.3 Inference/API

- API: `api/app.py`
- Prediction helpers: `src/models/predict.py`, `src/models/inference.py`

## 4. Module Responsibility Map

### Data
- `src/data/load_data.py`: robust CSV loading and merge helpers.
- `src/data/validate_data.py`: schema and data-quality checks.
- `src/data/clean_data.py`: text cleaning.
- `src/data/unified_label_schema.py`: unified schema normalization/validation.

### Features
- `src/features/feature_pipeline.py`: engineered text construction.
- `src/features/bias/*`: bias, ideology, propaganda, framing utilities.
- `src/features/emotion/*`: emotion lexicon/modeling and signals.
- `src/features/narrative/*`: narrative feature extraction.
- `src/features/discourse/*`: discourse feature extraction.

### Models
- `src/models/train_roberta.py`: configurable-label training pipeline.
- `src/models/multitask/multitask_truthlens_model.py`: multi-head architecture.
- `src/models/predict.py`: single/batch prediction.
- `src/models/inference.py`: inference + explainability pathway.
- `src/models/model_registry.py`: artifact loading utility.

### Pipelines
- `src/pipelines/data_pipeline.py`: data orchestration (pair-file and unified-file modes).
- `src/pipelines/feature_pipeline.py`: non-emotion feature orchestration.
- `src/pipelines/emotion_pipeline.py`: emotion feature orchestration.
- `src/pipelines/truthlens_pipeline.py`: end-to-end analysis orchestration.

### Training Utilities
- `src/training/cross_validation.py`: CV with label-column compatibility.
- `src/training/hyperparameter_tuning.py`: tuning with label-column compatibility.

### Evaluation & Visualization
- `src/evaluation/evaluate_model.py`: metric computation (binary and multiclass-safe).
- `src/evaluation/visualize_metrics.py`: evaluation charts.
- `src/visualization/visualize.py`: confusion matrix helper.

### Analysis / Graph / Explainability / Aggregation
- `src/analysis/*`: argument, narrative, omission, emotion-target, and profile builders.
- `src/graph/*`: entity/narrative graph construction and analytics.
- `src/explainability/*`: SHAP/LIME and feature explainers.
- `src/aggregation/truthlens_score_calculator.py`: final scoring aggregation.

## 5. Known Boundaries and Practical Notes

- Production API semantics still center on binary-style response fields (`fake_probability`) for backward compatibility.
- Multi-task dataset support is strongest in schema utilities, dataset build scripts, multitask model, and configurable training utilities.
- Root-level exploratory scripts (`ztest*.py`) are transitional tools and may be refactored later into package-level CLIs.

## 6. Active Documentation Set

- Overview: `README.md`
- Architecture: `architecture.md`
- Structure map: `structure.md`
- Status report: `PROJECT_REVIEW.md`
