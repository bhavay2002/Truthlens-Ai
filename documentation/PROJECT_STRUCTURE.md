
# PROJECT_STRUCTURE.md

This document explains the **directory structure and architectural organization** of the TruthLens AI repository.

TruthLens AI is organized as a **modular machine learning platform** designed for:

* misinformation detection
* credibility analysis
* linguistic signal extraction
* explainable AI
* scalable inference pipelines

The repository follows a **layered architecture** separating data processing, feature extraction, modeling, inference, and evaluation.

---

# Root Directory Overview

```text
TruthLens-AI/
```

Top-level components of the repository.

| Directory        | Description                              |
| ---------------- | ---------------------------------------- |
| `.github/`     | GitHub workflows and CI/CD configuration |
| `api/`         | FastAPI service for prediction endpoints |
| `config/`      | YAML configuration files                 |
| `data/`        | Raw, processed, and split datasets       |
| `logs/`        | Training and inference logs              |
| `models/`      | Saved trained models and vectorizers     |
| `reports/`     | Evaluation reports and EDA outputs       |
| `src/`         | Core application source code             |
| `tests/`       | Unit and integration tests               |
| `notebooks/`   | Research notebooks                       |
| `experiments/` | Experiment artifacts                     |

Additional project documentation:

* `README.md`
* `QUICKSTART.md`
* `KNOWLEDGE.md`
* `architecture.md`

---

# Continuous Integration

```text
.github/
 └── workflows/
     └── ci.yml
```

Defines automated pipelines for:

* running tests
* validating configuration
* ensuring code integrity

---

# API Layer

```text
api/
 └── app.py
```

Provides a **FastAPI-based REST service** for article analysis and model inference.

Example endpoints:

* `/predict`
* `/health`

---

# Configuration

```text
config/
 ├── config.yaml
 └── data_config.yaml
```

Stores system configuration parameters including:

* model settings
* dataset paths
* feature pipeline parameters
* training configuration

---

# Data Layer

```text
data/
```

Contains datasets used for training and evaluation.

Structure:

```text
data/
 ├── raw/
 ├── interim/
 ├── processed/
 ├── splits/
 ├── unified_dataset_train.csv
 ├── unified_dataset_validation.csv
 └── unified_dataset_test.csv
```

Dataset categories include:

* bias datasets
* emotion datasets
* narrative datasets
* propaganda datasets
* ideology datasets
* fake news datasets

These datasets are unified using a  **shared label schema** .

---

# Logs

```text
logs/
 ├── training.log
 ├── uvicorn_test.err
 └── uvicorn_test.out
```

Stores logs generated during training and inference.

---

# Models

```text
models/
```

Stores trained model artifacts and vectorizers.

```text
models/
 ├── roberta_model/
 └── tfidf_vectorizer.joblib
```

The `roberta_model` directory contains:

* transformer configuration
* tokenizer files
* trained model weights

---

# Reports

```text
reports/
```

Contains outputs from evaluation and exploratory analysis.

Examples include:

* confusion matrices
* dataset statistics
* word clouds
* feature distributions
* evaluation results

---

# Source Code

```text
src/
```

The `src` directory contains the  **core implementation of TruthLens AI** .

Main subsystems include:

* aggregation
* analysis
* data
* evaluation
* explainability
* features
* graph
* inference
* models
* pipelines
* training
* utilities
* visualization

---

# Aggregation System

```text
src/aggregation/
```

Responsible for computing the  **final TruthLens credibility score** .

Key modules:

* `truthlens_score_calculator.py`
* `score_normalizer.py`
* `risk_assessment.py`
* `weight_manager.py`
* `score_explainer.py`

---

# Linguistic Analysis

```text
src/analysis/
```

Implements advanced analysis of article content.

Capabilities include:

* bias profiling
* narrative extraction
* propaganda detection
* rhetorical device detection
* discourse coherence analysis
* context omission detection

---

# Data Processing

```text
src/data/
```

Handles dataset ingestion, cleaning, and preprocessing.

Important modules:

* `load_data.py`
* `merge_datasets.py`
* `clean_data.py`
* `data_split.py`
* `validate_data.py`
* `data_augmentation.py`

---

# Evaluation

```text
src/evaluation/
```

Responsible for model performance evaluation.

Includes:

* metrics computation
* calibration analysis
* uncertainty estimation
* evaluation dashboards
* PDF report generation

---

# Explainability

```text
src/explainability/
```

Provides explainable AI tools for interpreting model predictions.

Techniques implemented:

* SHAP
* LIME
* attention visualization
* token alignment
* explanation consistency metrics

---

# Feature Engineering

```text
src/features/
```

Generates structured features for the models.

Submodules include:

```text
features/
 ├── base/
 ├── bias/
 ├── discourse/
 ├── emotion/
 ├── narrative/
 ├── propaganda/
 ├── graph/
 ├── text/
 ├── fusion/
 ├── importance/
 └── pipelines/
```

Feature types include:

* lexical features
* semantic features
* syntactic features
* ideological features
* narrative signals
* propaganda patterns

---

# Graph Analysis

```text
src/graph/
```

Builds and analyzes graphs representing relationships between:

* entities
* narratives
* claims
* interactions

---

# Inference System

```text
src/inference/
```

Production inference pipeline for analyzing articles.

Main components:

* `inference_engine.py`
* `prediction_pipeline.py`
* `batch_inference.py`
* `model_loader.py`
* `report_generator.py`

---

# Model Architecture

```text
src/models/
```

Contains model implementations and utilities.

Key submodules:

```text
models/
 ├── encoder/
 ├── multitask/
 ├── narrative/
 ├── propaganda/
 ├── ideology/
 ├── ensemble/
 ├── training/
 ├── calibration/
 └── registry/
```

Supports:

* transformer encoders
* multitask learning
* ensemble models
* model calibration

---

# ML Pipelines

```text
src/pipelines/
```

Coordinates end-to-end machine learning workflows.

Examples:

* preprocessing pipelines
* feature pipelines
* prediction pipelines
* TruthLens analysis pipeline

---

# Training

```text
src/training/
```

Utilities for model training and optimization.

Includes:

* cross validation
* hyperparameter tuning
* optimizer configuration
* scheduler configuration

---

# Utilities

```text
src/utils/
```

Shared utilities used across the project.

Examples:

* configuration loading
* logging utilities
* device management
* reproducibility tools

---

# Visualization

```text
src/visualization/
```

Provides visualization tools for evaluation and analysis.

Examples include:

* confusion matrices
* ROC curves
* performance charts

---

# Testing

```text
tests/
```

Contains the project's automated test suite.

Tests cover:

* data processing
* model training
* inference pipelines
* explainability modules
* configuration validation
* reproducibility

---

# System Pipeline

TruthLens follows a  **multi-stage analysis pipeline** .

```text
Article Input
      ↓
Preprocessing
      ↓
Feature Engineering
      ↓
Transformer Models
      ↓
Linguistic Analysis Modules
      ↓
Graph Analysis
      ↓
Explainability
      ↓
Aggregation Engine
      ↓
TruthLens Credibility Score
```
