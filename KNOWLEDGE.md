# KNOWLEDGE.md

## 1. Project Overview

TruthLens AI is an end-to-end fake-news classification system.

Core outcomes:

- Train a RoBERTa classifier for binary classification (`REAL` vs `FAKE`).
- Build engineered textual context from source credibility, metadata, and TF-IDF keywords.
- Evaluate model quality with common classification metrics.
- Serve inference through a FastAPI application.

Main entry points:

- Training: `main.py`
- Evaluation: `evaluate.py`
- API: `api/app.py`
- EDA: `run_eda.py`

## 2. High-Level Architecture

```text
Raw Data (ISOT + LIAR + FakeNewsNet)
        |
        v
Dataset Merge (src/data/merge_datasets.py)
        |
        v
Cleaning + Quality Controls (src/data/clean_data.py, src/data/validate_data.py)
        |
        v
Feature Engineering (src/features/feature_pipeline.py)
  - source_features
  - metadata_features
  - text_features (TF-IDF)
        |
        v
Training Orchestration (main.py)
  - optional cross-validation
  - optional hyperparameter tuning
        |
        v
RoBERTa Training (src/models/train_roberta.py)
        |
        +--> Saved model + tokenizer
        +--> Saved reports/metrics/figures
        +--> Saved TF-IDF vectorizer

Serving Path:
Client -> FastAPI (api/app.py) -> src/models/predict.py -> model outputs
```

## 3. Execution Flows

### 3.1 Training Flow (`python main.py`)

1. Load global settings from `config/config.yaml` via `src/utils/settings.py`.
2. Create required directories and logging outputs.
3. Merge datasets (`src/data/merge_datasets.py`).
4. Clean text (`src/data/clean_data.py`).
5. Optionally augment data (`src/data/data_augmentation.py`).
6. Apply feature pipeline (`src/features/feature_pipeline.py`), creating `engineered_text`.
7. Optionally run CV (`src/training/cross_validation.py`).
8. Optionally run tuning (`src/training/hyperparameter_tuning.py`).
9. Train final RoBERTa model (`src/models/train_roberta.py`).
10. Evaluate and save metrics (`src/evaluation/evaluate_model.py`, `src/visualization/visualize.py`).

### 3.2 Evaluation Flow (`python evaluate.py`)

1. Load model/tokenizer from configured model path.
2. Load saved test set.
3. Generate predictions and probabilities.
4. Compute metrics and save reports.

### 3.3 API Flow (`uvicorn api.app:app --reload`)

1. `POST /predict` receives validated request text.
2. `src/models/predict.py` lazy-loads model/tokenizer.
3. Returns label, fake probability, confidence.

## 4. Configuration Model

All runtime settings come from `config/config.yaml` and are mapped into typed dataclasses in `src/utils/settings.py`.

Key setting groups:

- `model`: model name, sequence length, model save path.
- `training`: epochs, batch size, learning rate, CV/tuning toggles, search space.
- `features`: TF-IDF behavior for feature pipeline.
- `data`: augmentation and dataset paths.
- `paths`: logs, reports, artifact paths.

## 5. Key Module Responsibilities

### `src/data/*`

- `clean_data.py`: text normalization, deduplication, filtering.
- `data_augmentation.py`: synonym/deletion/swap augmentation.
- `eda.py`: exploratory analysis and figure/report generation.
- `load_data.py`: generic CSV loading/merging helper.
- `merge_datasets.py`: multi-source dataset merge logic.
- `validate_data.py`: dataset validation checks.

### `src/features/*`

- `feature_pipeline.py`: orchestrates metadata/source/TF-IDF features and builds `engineered_text`.
- `metadata_features.py`: text/title/author/source structural features.
- `source_features.py`: domain extraction and credibility scoring.
- `text_features.py`: TF-IDF vectorization utilities.

### `src/models/*`

- `train_roberta.py`: train/val/test split, tokenization, Trainer setup, save outputs.
- `predict.py`: lazy inference utilities for API and scripts.
- `model_utils.py`: generic model save/load wrappers.

### `src/training/*`

- `cross_validation.py`: stratified CV over dataframe training flows.
- `hyperparameter_tuning.py`: dataframe-first tuning (Optuna or fallback random search).

### `src/utils/*`

- `config_loader.py`: raw YAML loading helpers.
- `settings.py`: typed settings abstraction used by runtime modules.
- `logging_utils.py`: centralized logging setup.
- `input_validation.py`: reusable dataframe and numeric validators.
- `helper_functions.py`: small filesystem helper.

## 6. Full Tracked File Catalog

This section documents each tracked file and its purpose.

| File | Purpose | Use |
|---|---|---|
| `.env.example` | Example environment variables template | Copy to `.env` for local overrides if needed |
| `.github/workflows/ci.yml` | CI pipeline config | Runs automated checks in GitHub Actions |
| `.gitignore` | Git ignore rules | Prevents committing generated/local files |
| `CONTRIBUTING.md` | Contribution workflow | Guidance for contributors |
| `Dockerfile` | Container image definition | Build runnable API/training environment |
| `LICENSE` | License terms | Legal usage rights |
| `PROJECT_REVIEW.md` | Current project status summary | Implementation progress and roadmap |
| `QUICKSTART.md` | Fast command reference | Minimal setup/run guide |
| `README.md` | Primary project documentation | Overview, architecture, usage |
| `REQUIREMENTS_ADDITIONS.txt` | Supplemental requirements notes | Historical/additional dependency hints |
| `api/app.py` | FastAPI application | Exposes health and prediction endpoints |
| `config/config.yaml` | Canonical runtime configuration | Controls model/training/features/paths |
| `data/raw/FakeNewsNet/README.md` | Dataset source notes | Context for FakeNewsNet files |
| `data/raw/FakeNewsNet/gossipcop_fake.csv` | Raw labeled fake data | Training input source |
| `data/raw/FakeNewsNet/gossipcop_real.csv` | Raw labeled real data | Training input source |
| `data/raw/FakeNewsNet/politifact_fake.csv` | Raw labeled fake data | Training input source |
| `data/raw/FakeNewsNet/politifact_real.csv` | Raw labeled real data | Training input source |
| `data/raw/dataset.py` | Legacy dataset helper script | Historical/auxiliary data utility |
| `data/raw/isot/Fake.csv` | ISOT fake dataset | Training input source |
| `data/raw/isot/True.csv` | ISOT true dataset | Training input source |
| `data/raw/isot/dataset1info.txt` | ISOT dataset info | Dataset metadata reference |
| `data/raw/liar_dataset/README` | LIAR dataset notes | Dataset metadata reference |
| `data/raw/liar_dataset/test.tsv` | LIAR test split | Training input source (merge logic dependent) |
| `data/raw/liar_dataset/test_pos.csv` | LIAR processed variant | Auxiliary/raw asset |
| `data/raw/liar_dataset/train.tsv` | LIAR training split | Training input source |
| `data/raw/liar_dataset/train_pos.csv` | LIAR processed variant | Auxiliary/raw asset |
| `data/raw/liar_dataset/valid.tsv` | LIAR validation split | Training input source |
| `data/raw/liar_dataset/valid_pos.csv` | LIAR processed variant | Auxiliary/raw asset |
| `docker-compose.yml` | Multi-service container config | Local containerized run orchestration |
| `evaluate.py` | Standalone model evaluation runner | Re-evaluate saved model on test set |
| `main.py` | Main training orchestration script | Primary entry point for full pipeline |
| `reports/_test_tmp/cm_test.png` | Test artifact image | Generated during test/runtime checks |
| `reports/_test_tmp/cm_test_runtime_check.png` | Test artifact image | Generated during test/runtime checks |
| `reports/confusion_matrix.png` | Evaluation artifact | Confusion matrix output |
| `reports/data_cleaning_report.json` | Cleaning summary artifact | Rows removed/retention stats |
| `reports/eda_report.json` | EDA summary artifact | High-level exploratory analysis report |
| `reports/eda_summary.json` | EDA details artifact | EDA statistics output |
| `reports/evaluation_results.json` | Evaluation metrics artifact | Accuracy/precision/recall/f1/etc |
| `reports/figures/2gram_frequency.png` | EDA figure | Bigram frequency visualization |
| `reports/figures/3gram_frequency.png` | EDA figure | Trigram frequency visualization |
| `reports/figures/correlation_matrix.png` | EDA figure | Feature correlation visualization |
| `reports/figures/doc_length_by_label.png` | EDA figure | Document length by class |
| `reports/figures/fake_wordcloud.png` | EDA figure | Wordcloud for fake class |
| `reports/figures/label_distribution.png` | EDA figure | Label balance plot |
| `reports/figures/real_wordcloud.png` | EDA figure | Wordcloud for real class |
| `reports/figures/text_length_distribution.png` | EDA figure | Text length histogram |
| `reports/figures/text_length_outliers.png` | EDA figure | Outlier visualization |
| `reports/figures/text_length_vs_label.png` | EDA figure | Length vs label comparison |
| `reports/figures/word_frequency.png` | EDA figure | Top word frequency plot |
| `reports/figures/wordcloud.png` | EDA figure | Generic wordcloud output |
| `requirements.txt` | Python dependency manifest | Install runtime/dev dependencies |
| `run_eda.py` | EDA launcher script | Runs EDA workflow on dataset |
| `setup.py` | Environment bootstrap helper | Creates dirs, checks deps/tests |
| `source_scores.json` | Source credibility mapping | Used by source feature logic/context |
| `src/data/clean_data.py` | Text cleaning and dataframe cleaning | Preprocesses training text |
| `src/data/data_augmentation.py` | NLP augmentation utilities | Expands dataset when enabled |
| `src/data/eda.py` | EDA engine class | Produces analysis figures/reports |
| `src/data/load_data.py` | CSV/data loading utilities | Generic loader and merge helper |
| `src/data/merge_datasets.py` | Multi-dataset merge logic | Builds canonical merged dataframe |
| `src/data/validate_data.py` | Dataset validation class | Schema/null/duplicate/quality checks |
| `src/evaluation/evaluate_model.py` | Metric computation and save utilities | Training/evaluation reporting |
| `src/explainability/lime_explainer.py` | LIME explanation utilities | Local model interpretability |
| `src/explainability/shap_explainer.py` | SHAP explanation utilities | Model interpretability |
| `src/features/feature_pipeline.py` | Integrated feature engineering pipeline | Builds engineered text + saves vectorizer |
| `src/features/metadata_features.py` | Metadata feature extraction | Structural text/source/author signals |
| `src/features/source_features.py` | Source/domain credibility features | Domain extraction + credibility flags |
| `src/features/text_features.py` | TF-IDF utilities | Text vectorization helpers |
| `src/models/model_utils.py` | Generic serialization helpers | Save/load arbitrary model objects |
| `src/models/predict.py` | Inference functions | Single/batch prediction APIs |
| `src/models/train_roberta.py` | RoBERTa training module | Trainer setup, train, save outputs |
| `src/training/cross_validation.py` | CV utilities | Fold-based evaluation with train callbacks |
| `src/training/hyperparameter_tuning.py` | Hyperparameter search utilities | Optuna/fallback search over training params |
| `src/utils/config_loader.py` | Low-level YAML config helpers | Base config path/value utilities |
| `src/utils/helper_functions.py` | Simple helper utilities | Filesystem helper |
| `src/utils/input_validation.py` | Validation helper functions | Guard clauses for data/params |
| `src/utils/logging_utils.py` | Logging setup helper | Central logger configuration |
| `src/utils/settings.py` | Typed settings layer | Canonical config accessor |
| `src/visualization/visualize.py` | Plotting helpers | Confusion matrix rendering |
| `test_model.py` | Manual/quick inference script | Ad-hoc model sanity checking |
| `tests/test_feature_pipeline_and_validation.py` | Unit tests for features + validators | Verifies augmentation/feature behavior |
| `tests/test_settings_and_utils.py` | Unit tests for settings/utilities | Verifies settings and helper contracts |
| `tests/test_smoke.py` | Broad smoke tests | API/data/feature/evaluation sanity checks |
| `tests/test_training_utils.py` | Unit tests for CV/tuning utilities | Verifies training utility behavior |

## 7. Notes On Generated vs Source Files

- Files under `reports/` are runtime-generated artifacts and may change between runs.
- Files under `data/raw/` are dataset inputs, not code.
- Most operational logic resides in `src/`, plus entry scripts in repository root.

## 8. How To Extend The Project Safely

1. Add config keys in `config/config.yaml`.
2. Map them in `src/utils/settings.py`.
3. Consume settings in runtime modules.
4. Add input validation via `src/utils/input_validation.py`.
5. Add/update tests in `tests/`.
6. Update `README.md`, `QUICKSTART.md`, and this file.
