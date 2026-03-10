# TruthLens AI - QuickStart

This guide is for running the project quickly with the current architecture.

## 1. Environment Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Optional setup helper:

```bash
python setup.py
```

## 2. Data Expectations

The training pipeline uses merged datasets from:

- `data/raw/isot/`
- `data/raw/liar_dataset/`
- `data/raw/FakeNewsNet/`

If files are missing, add them under those folders before training.

## 3. Configuration (Before Training)

Edit `config/config.yaml`.

Common settings:

- `data.augmentation_multiplier`
- `training.epochs`
- `training.batch_size`
- `training.learning_rate`
- `training.text_column` (default: `engineered_text`)
- `training.run_cross_validation` (true/false)
- `training.run_hyperparameter_tuning` (true/false)
- `training.optuna_trials`

## 4. Train

```bash
python main.py
```

Pipeline steps:

1. Merge datasets
2. Clean dataset
3. Optional augmentation
4. Feature engineering (source + metadata + TF-IDF tokens)
5. Optional CV
6. Optional tuning
7. Final training
8. Evaluation + reports

## 5. Evaluate Saved Model

```bash
python evaluate.py
```

## 6. Run API

```bash
uvicorn api.app:app --reload
```

Endpoints:

- `GET /`
- `GET /health`
- `POST /predict`

## 7. Test Suite

```bash
python -B -m pytest -q
```

## 8. Useful Commands

```bash
# Run only training utility tests
python -B -m pytest tests/test_training_utils.py -q

# Run only feature/validation tests
python -B -m pytest tests/test_feature_pipeline_and_validation.py -q

# Run API tests
python -B -m pytest tests/test_smoke.py::TestAPI -q
```

## 9. Main Artifacts

- `models/roberta_model/`
- `models/tfidf_vectorizer.joblib`
- `logs/training.log`
- `reports/evaluation_results.json`
- `reports/confusion_matrix.png`
- `reports/data_cleaning_report.json`

## 10. Troubleshooting

- Model not found in API:
  - Run `python main.py` first.
- Slow training:
  - Lower `training.epochs`, use smaller batch size, or use GPU.
- Tuning fails due missing Optuna:
  - Install Optuna or rely on built-in fallback search.

For full architecture + file-by-file details, read `KNOWLEDGE.md`.
