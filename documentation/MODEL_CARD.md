
# MODEL_CARD.md

# TruthLens AI Model Card

This document describes the  **TruthLens AI model** , including its architecture, training data, intended uses, evaluation results, and limitations.

TruthLens AI is designed for  **misinformation detection and credibility analysis of news articles** .

---

# Model Overview

| Attribute         | Description                                                                   |
| ----------------- | ----------------------------------------------------------------------------- |
| Model Name        | TruthLens AI                                                                  |
| Model Type        | Transformer-based multi-task NLP system                                       |
| Base Architecture | RoBERTa Transformer Encoder                                                   |
| Tasks             | Fake news detection, bias detection, propaganda detection, narrative analysis |
| Framework         | PyTorch                                                                       |
| Interface         | FastAPI inference service                                                     |
| Language          | English                                                                       |

TruthLens combines **machine learning predictions with linguistic analysis and credibility scoring** to evaluate the reliability of news content.

---

# Model Architecture

The TruthLens system uses a **multi-layer architecture** composed of:

1. Transformer encoder
2. Task-specific prediction heads
3. Linguistic analysis modules
4. Graph reasoning modules
5. Explainability tools
6. Score aggregation engine

Simplified architecture:

```text
Article Text
      ↓
Tokenizer
      ↓
Transformer Encoder (RoBERTa)
      ↓
Task Heads
   ├── Fake News Classifier
   ├── Bias Detector
   ├── Emotion Classifier
   ├── Ideology Detector
   ├── Propaganda Detector
   └── Narrative Detector
      ↓
Linguistic Analysis Modules
      ↓
Graph Reasoning
      ↓
Explainability Layer
      ↓
TruthLens Credibility Score
```

---

# Training Data

TruthLens uses multiple datasets covering  **misinformation, bias, emotion, narrative framing, and propaganda** .

| Task                   | Dataset                 |
| ---------------------- | ----------------------- |
| Fake News Detection    | ISOT, LIAR, FakeNewsNet |
| Bias Detection         | BABE, BASIL, MBIC       |
| Emotion Classification | GoEmotions, SemEval     |
| Ideology Detection     | AllSides                |
| Narrative Analysis     | FrameNet                |
| Propaganda Detection   | PTC Propaganda Dataset  |

These datasets are merged using a  **unified label schema** .

---

# Training Procedure

Training pipeline:

```text
Dataset Loading
      ↓
Data Cleaning
      ↓
Feature Engineering
      ↓
Train / Validation Split
      ↓
Transformer Training
      ↓
Model Evaluation
      ↓
Checkpoint Saving
```

Training configuration includes:

* optimizer: AdamW
* learning rate: configurable
* batch size: configurable
* scheduler: linear warmup

Hyperparameter tuning and cross-validation are supported.

---

# Input Format

The model accepts **news article text** as input.

Example:

```json
{
  "text": "Breaking news: Scientists discover a new species in the Amazon rainforest."
}
```

---

# Output Format

Example model output:

```json
{
  "prediction": "REAL",
  "confidence": 0.91,
  "bias_score": 0.42,
  "propaganda_score": 0.37,
  "narrative_manipulation_score": 0.28,
  "truthlens_score": 0.76
}
```

Where:

| Field                        | Description                          |
| ---------------------------- | ------------------------------------ |
| prediction                   | Fake vs real classification          |
| confidence                   | Model confidence                     |
| bias_score                   | Estimated ideological bias           |
| propaganda_score             | Probability of propaganda techniques |
| narrative_manipulation_score | Narrative manipulation indicator     |
| truthlens_score              | Final credibility score              |

---

# Evaluation

Evaluation metrics include:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Calibration metrics

Evaluation reports are stored in:

```text
reports/
```

---

# Intended Use

TruthLens AI is intended for:

* misinformation detection research
* media credibility analysis
* journalism tools
* academic NLP research
* news monitoring systems

Possible deployment scenarios:

* fact-checking systems
* misinformation monitoring dashboards
* news aggregation platforms

---

# Out-of-Scope Use

TruthLens should  **not be used as the sole authority for determining factual truth** .

The system:

* cannot replace human fact-checking
* may misclassify satire or opinion pieces
* should not be used for censorship decisions without human oversight

---

# Ethical Considerations

Because TruthLens analyzes political and media content, several ethical considerations apply:

### Bias in Training Data

Training datasets may contain political or cultural biases.

### Misuse Risk

Automated credibility scoring could be misused for censorship or political manipulation.

### Transparency

Explainability modules are included to improve transparency of model decisions.

---

# Limitations

Known limitations include:

* primarily trained on **English-language datasets**
* may struggle with **sarcasm or satire**
* performance depends on **training dataset quality**
* limited ability to verify factual claims directly

TruthLens focuses on  **linguistic and structural signals** , not factual verification.

---

# Explainability

TruthLens includes multiple explanation methods:

* SHAP explanations
* LIME explanations
* attention visualization
* feature importance analysis

These tools help users understand  **why a prediction was made** .

---

# Versioning

Model versions are managed using:

* checkpoint manager
* model registry
* model metadata files

Each model version includes:

* configuration
* training parameters
* dataset information
* evaluation results

---

# Future Improvements

Planned improvements include:

* multilingual misinformation detection
* improved narrative analysis models
* knowledge graph integration
* real-time news monitoring pipelines
* improved factual verification modules
