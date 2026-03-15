# Test Suite Guide

This document describes each file under `tests/`, what it validates, and why it exists.

## Shared Setup

| File | What it does | What it tests |
|---|---|---|
| `conftest.py` | Shared pytest fixtures and project-path bootstrap. | Ensures `src` and `api` imports work consistently and provides reusable sample DataFrames. |

## API Tests

| File | What it does | What it tests |
|---|---|---|
| `test_api.py` | Tests FastAPI endpoints with a stubbed prediction function. | Health endpoint availability, `/predict` response schema, and request validation behavior. |

## Configuration Tests

| File | What it does | What it tests |
|---|---|---|
| `test_config_integrity.py` | Loads typed app settings. | Required training/model settings are present and sane. |
| `test_config_loading.py` | Loads YAML configuration. | Core config sections (`model`, `training`, `data`) exist. |

## Data Tests

| File | What it does | What it tests |
|---|---|---|
| `test_data_augmentation.py` | Validates augmentation interface behavior. | `multiplier=1` no-op and invalid multipliers raise `ValueError`. |
| `test_data_leakage.py` | Checks train/test text overlap after splitting. | No duplicate texts leak across split boundaries. |
| `test_data_processing.py` | Tests cleaning utilities. | URL/lowercasing cleanup and DataFrame text-column sanitation flow. |
| `test_data_validation.py` | Tests schema validator. | Required columns are accepted by `DataValidator`. |
| `test_dataset_schema.py` | Minimal schema contract check. | Dataset contains required `text` and `label` columns. |
| `test_dataset_split_integrity.py` | Split integrity test on synthetic data. | Train/test split has no overlapping text rows. |

## Evaluation & Explainability Tests

| File | What it does | What it tests |
|---|---|---|
| `test_evaluation.py` | Runs evaluation on small label arrays. | Accuracy metric is returned and bounded in `[0, 1]`. |
| `test_explainability.py` | Tests explainability pipeline with monkeypatched internals. | `explain_bias` returns the full expected result schema. |

## Feature Engineering Tests

| File | What it does | What it tests |
|---|---|---|
| `test_feature_pipeline.py` | Runs feature engineering on fixture data. | Engineered text column creation, row-count preservation, and vectorizer output. |

## Inference & Input Tests

| File | What it does | What it tests |
|---|---|---|
| `test_inference_speed.py` | Lightweight performance sanity check. | Basic inference-like loop completes under a simple threshold. |
| `test_input_validation.py` | Tests integer validator utility. | Valid positive ints pass, invalid values raise `ValueError`. |
| `test_prediction_pipeline.py` | Tests prediction function with mocked tokenizer/model. | Prediction output schema and empty-input validation. |
| `test_prediction_stability.py` | Determinism check with dummy model. | Same input returns identical prediction twice. |

## Logging & Project Structure Tests

| File | What it does | What it tests |
|---|---|---|
| `test_logging.py` | Logger initialization smoke test. | Logger instance is available and callable. |
| `test_project_structure.py` | Filesystem contract tests. | Required project directories and core files exist. |

## Training Tests

| File | What it does | What it tests |
|---|---|---|
| `test_model_training.py` | Tests training split/validation helpers (no heavy model training). | Disjoint train/val/test partitions and split DataFrame validation errors. |
| `test_training_pipeline.py` | Tests CV and tuning wrappers with dummy trainer. | Cross-validation summary contract and hyperparameter tuning result contract. |
| `test_tokenization.py` | Tests tokenization helper with stub tokenizer. | `tokenize_function` output keys and shape consistency. |

## Utility Tests

| File | What it does | What it tests |
|---|---|---|
| `test_model_utils.py` | Serialization utility tests. | Saving and loading model payloads through filesystem. |
| `test_reproducibility.py` | Randomness control test. | Same random seed reproduces identical NumPy outputs. |
| `test_utils.py` | Utility integration tests. | Typed settings, folder creation behavior, and model utils with `Path` objects. |

## How To Run

```bash
pytest -q
```

Current baseline in this repository: `37 passed`.
