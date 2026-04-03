
# ARCHITECTURE.md

This document describes the  **system architecture of TruthLens AI** .

TruthLens AI is a **multi-layer machine learning platform** designed for:

* misinformation detection
* credibility analysis
* linguistic signal extraction
* explainable AI
* scalable inference pipelines

The architecture follows a **modular layered design** that separates data processing, feature engineering, model inference, linguistic analysis, graph reasoning, explainability, and scoring.

---

# High-Level Architecture

TruthLens processes articles through multiple analytical layers.

```text
Article Input
      ↓
Preprocessing
      ↓
Feature Engineering
      ↓
Transformer Model
      ↓
Linguistic Analysis
      ↓
Graph Analysis
      ↓
Explainability
      ↓
Score Aggregation
      ↓
TruthLens Credibility Score
```

Each layer contributes signals used to evaluate  **information credibility and manipulation risk** .

---

# System Components

The system consists of several major subsystems.

| Layer                | Purpose                              |
| -------------------- | ------------------------------------ |
| Data Layer           | Dataset ingestion and preprocessing  |
| Feature Layer        | Feature extraction from text         |
| Model Layer          | Transformer and multitask models     |
| Analysis Layer       | Linguistic and narrative analysis    |
| Graph Layer          | Entity and narrative graph reasoning |
| Explainability Layer | Interpretable predictions            |
| Aggregation Layer    | Credibility scoring                  |
| Inference Layer      | Production inference pipeline        |

---

# Data Layer

Responsible for  **dataset ingestion, validation, and preprocessing** .

Location:

```text
src/data/
```

Key responsibilities:

* load datasets
* merge datasets
* clean text
* perform data augmentation
* validate schema
* split datasets

Typical flow:

```text
Raw Dataset
      ↓
Cleaning
      ↓
Validation
      ↓
Unified Dataset
      ↓
Train / Validation / Test Splits
```

---

# Feature Engineering Layer

Transforms raw article text into  **structured features** .

Location:

```text
src/features/
```

Feature categories include:

* lexical features
* semantic features
* syntactic features
* ideological signals
* narrative features
* propaganda indicators
* emotion trajectories

Feature pipeline:

```text
Article Text
      ↓
Tokenization
      ↓
Feature Extractors
      ↓
Feature Fusion
      ↓
Unified Feature Representation
```

These features are used by the  **machine learning models** .

---

# Model Layer

The model layer implements the  **core predictive models** .

Location:

```text
src/models/
```

Supported model types:

* Transformer encoders
* Multitask learning models
* Ensemble models
* Task-specific classifiers

Example tasks:

* fake news classification
* bias detection
* emotion classification
* ideology detection
* propaganda detection
* narrative detection

Model pipeline:

```text
Input Features
      ↓
Transformer Encoder
      ↓
Task Heads
      ↓
Predictions
```

---

# Linguistic Analysis Layer

Performs deeper analysis of article structure.

Location:

```text
src/analysis/
```

Capabilities include:

* bias profiling
* rhetorical device detection
* narrative extraction
* context omission detection
* information density analysis
* discourse coherence analysis

These modules generate  **linguistic credibility signals** .

---

# Graph Reasoning Layer

Constructs graphs representing relationships between entities and narratives.

Location:

```text
src/graph/
```

Graph types include:

* entity graphs
* narrative graphs
* temporal graphs

Graph analysis enables detection of:

* narrative propagation
* narrative conflicts
* interaction patterns

Graph pipeline:

```text
Extracted Entities
      ↓
Graph Construction
      ↓
Graph Embeddings
      ↓
Graph Features
```

---

# Explainability Layer

Provides interpretable explanations for model predictions.

Location:

```text
src/explainability/
```

Explanation methods include:

* SHAP
* LIME
* attention visualization
* token attribution

Explanation outputs help users understand  **why an article was classified as misleading** .

---

# Aggregation Layer

Combines signals from all modules into a  **single credibility score** .

Location:

```text
src/aggregation/
```

Aggregation components:

* score normalization
* risk assessment
* signal weighting
* score explanation

Example output:

```text
Fake News Score: 0.82
Bias Score: 0.63
Propaganda Score: 0.71
Narrative Manipulation Score: 0.55

TruthLens Credibility Score: 0.24
Risk Level: HIGH
```

---

# Inference Layer

Handles  **real-time article analysis** .

Location:

```text
src/inference/
```

Responsibilities:

* model loading
* feature preparation
* article analysis
* batch processing
* result formatting

Inference pipeline:

```text
Article
      ↓
Preprocessing
      ↓
Feature Extraction
      ↓
Model Prediction
      ↓
Analysis Modules
      ↓
Score Aggregation
      ↓
Formatted Output
```

---

# API Layer

The system exposes inference capabilities through a  **FastAPI service** .

Location:

```text
api/app.py
```

Example endpoint:

```text
POST /predict
```

Input:

```json
{
  "text": "Article content"
}
```

Output:

```json
{
  "prediction": "FAKE",
  "confidence": 0.87,
  "truthlens_score": 0.32
}
```

---

# Training Architecture

Training pipeline:

```text
Dataset
      ↓
Data Cleaning
      ↓
Feature Generation
      ↓
Model Training
      ↓
Evaluation
      ↓
Checkpoint Storage
```

Training utilities are located in:

```text
src/training/
```

---

# Evaluation System

Responsible for model performance analysis.

Location:

```text
src/evaluation/
```

Evaluation capabilities include:

* classification metrics
* calibration analysis
* uncertainty estimation
* task correlation analysis
* evaluation dashboards

---

# System Design Principles

TruthLens architecture follows several design principles:

### Modularity

Each subsystem is independent and replaceable.

### Scalability

Pipelines support batch processing and large datasets.

### Interpretability

Explainability modules provide transparency.

### Reproducibility

Configurations and tests ensure reproducible experiments.

### Production Readiness

Inference pipelines and API services support deployment.

---

# Complete System Flow

The complete processing pipeline:

```text
News Article
      ↓
Preprocessing
      ↓
Feature Extraction
      ↓
Transformer Models
      ↓
Linguistic Analysis
      ↓
Graph Analysis
      ↓
Explainability
      ↓
Score Aggregation
      ↓
TruthLens Credibility Score
      ↓
API Response
```

---
