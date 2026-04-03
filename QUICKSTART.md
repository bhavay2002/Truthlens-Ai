
# TruthLens AI — Quick Start

This guide explains how to  **run TruthLens AI quickly using the current multi-model architecture** .

TruthLens is an AI system designed to analyze news articles and detect:

* Fake news signals
* Political and media bias
* Propaganda techniques
* Narrative framing
* Ideological positioning
* Emotional manipulation

The system combines **multiple models and feature pipelines** to produce a **comprehensive credibility and narrative analysis** of an article.

---

# 1. Environment Setup

Create a virtual environment.

```bash
python -m venv venv
```

Activate the environment.

### Windows

```bash
venv\Scripts\activate
```

### Linux / Mac

```bash
source venv/bin/activate
```

Install dependencies.

```bash
pip install -r requirements.txt
```

Optional environment setup helper:

```bash
python setup.py
```

This script may:

* verify environment configuration
* create required directories
* validate dataset paths
* prepare cache directories

---

# 2. Dataset Requirements

TruthLens supports  **multi-task learning** , so datasets may contain labels for multiple signals.

Datasets should be placed in:

```
data/raw/isot/
data/raw/liar_dataset/
data/raw/FakeNewsNet/
```

These datasets are merged during training.

### Dataset Roles

| Dataset        | Purpose                          |
| -------------- | -------------------------------- |
| ISOT Fake News | Fake vs real news classification |
| LIAR Dataset   | political truthfulness labels    |
| FakeNewsNet    | misinformation + social signals  |

Optional additional datasets may support:

* propaganda detection
* ideology labeling
* narrative framing
* emotion detection

---

# 3. Configuration

Edit the configuration file:

```
config/config.yaml
```

Important settings:

### Data Configuration

| Parameter                        | Description                |
| -------------------------------- | -------------------------- |
| `data.augmentation_multiplier` | Dataset augmentation level |
| `data.merge_strategy`          | Dataset merging strategy   |

### Training Configuration

| Parameter                  | Description                |
| -------------------------- | -------------------------- |
| `training.epochs`        | Number of training epochs  |
| `training.batch_size`    | Training batch size        |
| `training.learning_rate` | Optimizer learning rate    |
| `training.text_column`   | Text column used by models |

### Multi-Model Configuration

TruthLens supports multiple model heads.

| Parameter                    | Description                  |
| ---------------------------- | ---------------------------- |
| `models.enable_fake_news`  | Enable fake news classifier  |
| `models.enable_bias`       | Enable bias detection model  |
| `models.enable_propaganda` | Enable propaganda classifier |
| `models.enable_narrative`  | Enable narrative analysis    |
| `models.enable_ideology`   | Enable ideology classifier   |
| `models.enable_emotion`    | Enable emotion analysis      |

### Training Options

| Parameter                              | Description                  |
| -------------------------------------- | ---------------------------- |
| `training.run_cross_validation`      | Enable cross-validation      |
| `training.run_hyperparameter_tuning` | Enable hyperparameter search |
| `training.optuna_trials`             | Number of tuning trials      |

---

# 4. Train the Models

Run the full training pipeline.

```bash
python main.py
```

Training pipeline steps:

1. Dataset loading and merging
2. Data cleaning and normalization
3. Optional dataset augmentation
4. Feature engineering
   * source credibility features
   * metadata signals
   * linguistic features
   * TF-IDF tokens
5. Feature pipeline transformation
6. Multi-task model training
7. Optional cross-validation
8. Optional hyperparameter tuning
9. Final training of all model heads
10. Evaluation and report generation

---

# 5. Multi-Model Architecture

TruthLens uses  **multiple specialized models** .

| Model            | Purpose                         |
| ---------------- | ------------------------------- |
| Fake News Model  | Detect misinformation           |
| Bias Model       | Detect political/media bias     |
| Propaganda Model | Identify propaganda techniques  |
| Narrative Model  | Detect narrative framing        |
| Ideology Model   | Predict ideological orientation |
| Emotion Model    | Detect emotional manipulation   |

Outputs from these models are combined to generate a  **comprehensive article analysis report** .

---

# 6. Evaluate Trained Models

Run evaluation.

```bash
python evaluate.py
```

Evaluation includes:

* classification metrics
* confusion matrices
* model-specific performance
* multi-task evaluation reports

Metrics may include:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

---

# 7. Run the API Server

TruthLens exposes a REST API for inference.

Start the API server.

```bash
uvicorn api.app:app --reload
```

Default address:

```
http://127.0.0.1:8000
```

---

# 8. API Endpoints

| Endpoint     | Method | Description     |
| ------------ | ------ | --------------- |
| `/`        | GET    | API info        |
| `/health`  | GET    | health check    |
| `/predict` | POST   | analyze article |

Example request:

```json
{
  "text": "Example news article content..."
}
```

Example response:

```json
{
  "fake_news_probability": 0.78,
  "bias_score": 0.42,
  "propaganda_score": 0.36,
  "narrative_type": "political conflict",
  "ideology": "left-leaning",
  "emotion_profile": {
    "anger": 0.32,
    "fear": 0.18,
    "joy": 0.05
  }
}
```

---

# 9. Run the Test Suite

Execute all tests.

```bash
python -B -m pytest -q
```

Tests verify:

* feature pipelines
* data validation
* training utilities
* inference pipeline
* API endpoints

---

# 10. Useful Testing Commands

### Training utility tests

```bash
python -B -m pytest tests/test_training_utils.py -q
```

### Feature pipeline tests

```bash
python -B -m pytest tests/test_feature_pipeline_and_validation.py -q
```

### API tests

```bash
python -B -m pytest tests/test_smoke.py::TestAPI -q
```

---

# 11. Generated Artifacts

After training, the following artifacts are generated.

| Artifact                              | Description                |
| ------------------------------------- | -------------------------- |
| `models/fake_news_model/`           | fake news classifier       |
| `models/bias_model/`                | bias detection model       |
| `models/propaganda_model/`          | propaganda classifier      |
| `models/narrative_model/`           | narrative analysis model   |
| `models/ideology_model/`            | ideology classifier        |
| `models/emotion_model/`             | emotion classifier         |
| `models/tfidf_vectorizer.joblib`    | TF-IDF vectorizer          |
| `logs/training.log`                 | training logs              |
| `reports/evaluation_results.json`   | evaluation metrics         |
| `reports/confusion_matrix.png`      | confusion matrix           |
| `reports/data_cleaning_report.json` | dataset processing summary |

---

# 12. Troubleshooting

### Model not found in API

Run training first.

```bash
python main.py
```

---

### Training is slow

Possible solutions:

* reduce `training.epochs`
* reduce `training.batch_size`
* disable hyperparameter tuning
* run training on GPU

---

### Hyperparameter tuning fails

Install Optuna.

```bash
pip install optuna
```

If Optuna is unavailable, TruthLens will fall back to  **basic parameter search** .
