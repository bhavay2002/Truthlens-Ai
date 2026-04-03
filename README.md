
# TruthLens AI

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![FastAPI](https://img.shields.io/badge/API-FastAPI-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

**TruthLens AI** is a  **multi-layer AI system for misinformation detection and credibility analysis** .

The platform combines:

* Transformer-based NLP models
* Linguistic feature engineering
* Narrative and discourse analysis
* Propaganda detection
* Bias and ideological language detection
* Graph-based reasoning
* Explainable AI techniques

TruthLens evaluates news articles using **dozens of analytical signals** and produces a  **structured credibility score with explanations** .

---

# Key Capabilities

### Fake News Detection

Binary classification of news articles:

```
REAL vs FAKE
```

Using transformer models and engineered features.

---

### Linguistic & Narrative Analysis

TruthLens performs deep linguistic analysis including:

* Bias profiling
* Emotion targeting
* Narrative structure extraction
* Rhetorical device detection
* Context omission detection
* Information density analysis

---

### Propaganda & Ideology Detection

The system identifies propaganda techniques and ideological framing:

* Manipulation patterns
* Ideologically loaded language
* Framing strategies
* Persuasion techniques

---

### Graph-Based Analysis

TruthLens constructs graphs representing relationships between:

* Entities
* Claims
* Narratives
* Sources

Graph reasoning enables  **propagation and narrative conflict detection** .

---

### Explainable AI

Predictions are accompanied by explanations using:

* SHAP
* LIME
* Attention rollout
* Feature importance
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

TruthLens follows a  **multi-stage ML pipeline** .

```
Article Input
      ↓
Preprocessing
      ↓
Feature Engineering
      ↓
Transformer Model
      ↓
Analysis Modules
      ↓
Graph Analysis
      ↓
Explainability Layer
      ↓
Aggregation Engine
      ↓
TruthLens Credibility Score
```

---

# Repository Structure

Simplified project structure:

```
TruthLens-AI/

api/                     FastAPI inference service
config/                  YAML configuration files
data/                    Raw / processed datasets
experiments/             Experimental results
logs/                    Training & inference logs
models/                  Saved model artifacts
notebooks/               Research notebooks
reports/                 Evaluation reports and EDA

src/

  aggregation/           Credibility scoring system
  analysis/              Linguistic & narrative analysis modules
  data/                  Data ingestion & preprocessing
  evaluation/            Evaluation metrics and dashboards
  explainability/        SHAP, LIME, and explanation tools
  features/              Feature engineering pipelines
  graph/                 Graph construction & analysis
  inference/             Production inference engine
  models/                Transformer and multitask models
  pipelines/             End-to-end ML pipelines
  training/              Training utilities
  utils/                 Config, logging, helper utilities
  visualization/         Plotting and evaluation visualization

tests/                   Unit and integration tests

main.py                  Training pipeline
evaluate.py              Evaluation script
run_eda.py               Exploratory data analysis
```

This architecture enables  **modular experimentation and scalable inference pipelines** .

---

# Installation

Create virtual environment

```bash
python -m venv venv
```

Activate environment

Windows

```
venv\Scripts\activate
```

Linux / macOS

```
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
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
4. Model training
5. Evaluation report generation

---

# Evaluation

Evaluate trained models:

```bash
python evaluate.py
```

Evaluation outputs include:

* Precision
* Recall
* F1 Score
* Calibration analysis
* Confusion matrices
* Task correlation analysis

Reports are stored in:

```
reports/
```

---

# Inference API

Start the FastAPI server:

```bash
uvicorn api.app:app --reload
```

Default address:

```
http://localhost:8000
```

---

# API Example

```
POST /predict
```

Example request:

```bash
curl -X POST http://localhost:8000/predict \
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

Datasets are unified using a  **shared label schema** .

---

# Testing

Run the full test suite:

```bash
pytest
```

Tests cover:

* data pipelines
* model training
* inference pipelines
* explainability modules
* configuration integrity

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
