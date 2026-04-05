
# TruthLens AI

![Python](https://img.shields.io/badge/Python-3.12+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![FastAPI](https://img.shields.io/badge/API-FastAPI-green)
![Tests](https://img.shields.io/badge/Tests-236%20passing-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

**TruthLens AI** is a **multi-layer AI system for misinformation detection and credibility analysis**.

The platform combines:

* Multi-task RoBERTa transformer models with shared encoder and task-specific heads
* Linguistic feature engineering and narrative discourse analysis
* Propaganda and ideological framing detection
* Bias and emotion classification (20-label multi-label)
* Graph-based entity and claim reasoning
* Explainable AI via SHAP, LIME, and attention rollout
* Weighted credibility score aggregation with risk assessment

TruthLens evaluates news articles using **dozens of analytical signals** and produces a **structured credibility score with explanations**.

---

# Key Capabilities

### Fake News Detection

Binary classification of news articles:

```
REAL vs FAKE
```

Using multi-task transformer models and engineered features.

---

### Linguistic & Narrative Analysis

TruthLens performs deep linguistic analysis including:

* Bias profiling
* Emotion targeting (20-class multi-label)
* Narrative structure extraction (hero / villain / victim)
* Narrative frame detection (RE / HI / CO / MO / EC)
* Rhetorical device detection
* Context omission detection
* Information density analysis

---

### Propaganda & Ideology Detection

The system identifies propaganda techniques and ideological framing:

* Manipulation patterns
* Ideologically loaded language (left / center / right)
* Framing strategies
* Persuasion techniques

---

### Graph-Based Analysis

TruthLens constructs graphs representing relationships between:

* Entities
* Claims
* Narratives
* Sources

Graph reasoning enables **propagation and narrative conflict detection**.

---

### Explainable AI

Predictions are accompanied by explanations using:

* SHAP token importance
* LIME local interpretability
* Attention rollout
* Integrated gradients
* Explanation consistency checks

---

### Credibility Score Aggregation

Outputs a final **TruthLens credibility score** using weighted signals from multiple modules.

Example output:

```
Fake News Probability: 0.82
Bias Score: 0.61
Propaganda Score: 0.72
Narrative Manipulation Score: 0.58

TruthLens Credibility Score: 0.24
Risk Level: HIGH
```

---

# System Architecture

TruthLens follows a **multi-stage ML pipeline**.

```
Article Input
      ↓
Preprocessing
      ↓
Feature Engineering
      ↓
Multi-Task RoBERTa Encoder
      ↓
Task Heads (Bias / Ideology / Propaganda / Narrative / Emotion)
      ↓
Graph Analysis
      ↓
Explainability Layer (SHAP / LIME / Attention)
      ↓
Aggregation Engine
      ↓
TruthLens Credibility Score
```

---

# Repository Structure

```
TruthLens-AI/

api/                     FastAPI inference service
config/                  YAML configuration files
data/                    Root-level data pipeline orchestration
  data_pipeline.py       Pipeline entry point and config validator
experiments/             Experimental results
logs/                    Training & inference logs
models/                  Saved model artifacts
notebooks/               Research notebooks
reports/                 Evaluation reports and EDA
training/                Root-level training compatibility shim

src/

  aggregation/           Credibility scoring and risk assessment
  analysis/              Linguistic & narrative analysis modules
  data/                  Data ingestion & preprocessing
  evaluation/            Evaluation metrics and dashboards
  explainability/        SHAP, LIME, bias explainer, and explanation tools
  features/              Feature engineering pipelines
  graph/                 Graph construction & analysis
  inference/             Production inference engine
  models/
    encoder/             Shared transformer encoder
    heads/               Classification and multi-label task heads
    multitask/           MultiTaskTruthLensModel and config
    inference/           Predictor (predict_batch for LIME)
  pipelines/             End-to-end ML pipelines
  training/              Training utilities, cross-validation, hyperparameter tuning
  utils/                 Config, logging, seed control, helper utilities
  visualization/         Plotting and evaluation visualization

tests/                   37 test modules, 236 tests (all passing)

main.py                  Training pipeline entry point
evaluate.py              Evaluation script
run_eda.py               Exploratory data analysis
```

---

# Installation

**Python 3.12+ required.**

Create a virtual environment:

```bash
python -m venv venv
```

Activate it:

```bash
# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

Install dependencies (CPU-only PyTorch recommended to reduce disk usage):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Download the spaCy language model:

```bash
python -m spacy download en_core_web_sm
```

---

# Training

Run the training pipeline:

```bash
python main.py
```

The pipeline performs:

1. Dataset merging
2. Data cleaning
3. Feature engineering
4. Multi-task model training
5. Evaluation report generation

---

# Evaluation

Evaluate trained models:

```bash
python evaluate.py
```

Evaluation outputs include:

* Precision, Recall, F1 Score
* Calibration analysis
* Confusion matrices
* Task correlation analysis
* Uncertainty quantification

Reports are stored in:

```
reports/
```

---

# Inference API

Start the FastAPI server:

```bash
uvicorn api.app:app --host 0.0.0.0 --port 5000 --reload
```

The API is available at:

```
http://localhost:5000
```

Interactive docs (Swagger UI):

```
http://localhost:5000/docs
```

---

# API Example

```
POST /predict
```

Example request:

```bash
curl -X POST http://localhost:5000/predict \
-H "Content-Type: application/json" \
-d '{"text":"Breaking news: Scientists discover new species in the Amazon rainforest."}'
```

Example response:

```json
{
  "prediction": "REAL",
  "confidence": 0.91,
  "truthlens_score": 0.72
}
```

---

# Datasets

TruthLens integrates multiple datasets including:

| Task       | Dataset                 |
| ---------- | ----------------------- |
| Fake News  | ISOT, LIAR, FakeNewsNet |
| Bias       | BABE, BASIL, MBIC       |
| Emotion    | GoEmotions, SemEval     |
| Narrative  | FrameNet                |
| Propaganda | PTC Propaganda          |
| Ideology   | AllSides                |

Datasets are unified using a **shared label schema**.

---

# Testing

Run the full test suite:

```bash
pytest
```

**236 tests across 37 modules — all passing.**

Coverage includes:

| Area | Test Modules |
|------|-------------|
| API endpoints & error paths | `test_api.py`, `test_api_error_paths.py` |
| Aggregation & risk scoring | `test_aggregation.py` |
| Evaluation metrics & uncertainty | `test_evaluation.py`, `test_evaluation_metrics.py` |
| Explainability (SHAP / LIME / bias) | `test_explainability.py`, `test_shap_explainer.py` |
| Emotion lexicon analysis | `test_emotion.py` |
| Input validation | `test_input_validation.py` |
| Model architecture & registry | `test_model_subpackage_imports.py`, `test_model_registry.py`, `test_multitask_label_helpers.py` |
| Model training & tokenization | `test_model_training.py`, `test_tokenization.py` |
| Training pipeline & cross-validation | `test_training_pipeline.py` |
| Inference speed & prediction stability | `test_inference_speed.py`, `test_prediction_stability.py` |
| Data pipelines & schema | `test_data_pipeline_module.py`, `test_dataset_schema.py` |
| Configuration loading | `test_config_loading.py`, `test_config_integrity.py` |
| Reproducibility (seed control) | `test_reproducibility.py` |
| Utility functions | `test_utils.py` |
| Project structure | `test_project_structure.py` |

---

# Future Work

Planned extensions:

* Multilingual misinformation detection
* Real-time news monitoring
* Knowledge graph integration
* Cross-source narrative tracking
* Browser credibility extension

---

# License

MIT License
