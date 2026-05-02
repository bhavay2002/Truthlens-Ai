# Training Guide

This document walks through **training the TruthLens AI model** from data preparation to a fully trained and ready-to-use model artifact.

---

## Prerequisites

Ensure the following are in place before training:

1. Python 3.12 environment with all dependencies installed (`pip install -r requirements.txt`)
2. Raw datasets placed in `data/raw/` under the appropriate task subdirectory
3. `config/config.yaml` reviewed and adjusted if needed
4. At least 8 GB of RAM (16 GB recommended for default settings)
5. A GPU with CUDA is strongly recommended for full training with `amp_dtype: bf16` — CPU training is significantly slower and BF16 AMP will be skipped automatically

---

## Quick Start

To run the complete training pipeline with default settings:

```bash
python main.py
```

This will:
1. Run the 8-stage data pipeline (`src/data_processing/data_pipeline.py`)
2. Validate dataset quality and schemas against `data_contracts.py`
3. Apply cleaning, augmentation, and leakage checking
4. Build PyTorch datasets and dataloaders
5. Train the MultiTask RoBERTa model
6. Evaluate on the validation set using `weighted_composite_score`
7. Save the checkpoint to `models/checkpoints/checkpoint.pt`

---

## Step-by-Step

### Step 1 — Prepare Raw Datasets

Place your datasets in the appropriate subdirectories:

```
data/raw/
├── bias/           # CSV with columns: text, bias_label
├── emotion/        # CSV with columns: text, emotion_0, emotion_1, ..., emotion_10
├── ideology/       # CSV with columns: text, ideology_label
├── narrative/      # CSV with columns: text, hero, villain, victim
│                   #   + frame columns: CO, EC, HI, MO, RE
└── propaganda/     # CSV with columns: text, propaganda_label
```

**Important — emotion column format:**
The EMOTION-11 schema uses positional integer suffixes. Column names are `emotion_0` through `emotion_10`, mapped to:

| Column      | Label        |
|-------------|--------------|
| `emotion_0` | `neutral`    |
| `emotion_1` | `admiration` |
| `emotion_2` | `approval`   |
| `emotion_3` | `gratitude`  |
| `emotion_4` | `annoyance`  |
| `emotion_5` | `amusement`  |
| `emotion_6` | `curiosity`  |
| `emotion_7` | `disapproval`|
| `emotion_8` | `love`       |
| `emotion_9` | `optimism`   |
| `emotion_10`| `anger`      |

Legacy column names from the 20-label GoEmotions set (`emotion_joy`, `emotion_fear`, etc.) are **rejected** by the data contracts validator.

**Task schemas** are defined in `src/data_processing/data_contracts.py` — consult this file as the authoritative reference for column names and types.

**Minimum dataset requirements (from `data_config.yaml`):**
- Each row must have at least 30 words
- No more than 10% null values per column
- No more than 15% duplicate rows
- Each class label must appear in at least 10% of rows

**Supported datasets:**

| Task                   | Recommended datasets             |
|------------------------|----------------------------------|
| Bias detection         | BABE, BASIL, MBIC                |
| Emotion classification | GoEmotions (11-label subset)     |
| Ideology detection     | AllSides                         |
| Narrative analysis     | FrameNet                         |
| Propaganda detection   | PTC Propaganda Corpus            |
| Fake news (optional)   | ISOT, LIAR, FakeNewsNet          |

---

### Step 2 — Review Configuration

Open `config/config.yaml` and verify the key training settings:

```yaml
model:
  encoder: roberta-base          # Change to roberta-large for better performance
  hidden_dim: 768
  dropout: 0.1
  gradient_checkpointing: true
  flash_attention: true
  torch_compile: true            # Disable if your environment doesn't support it
  compile_mode: "default"

data:
  batch_size: 32                 # Reduce to 16 or 8 if GPU memory is tight
  num_workers: 8
  shuffle: true

training:
  epochs: 10                     # Upper bound; early stopping typically fires by epoch 4–6
  min_epochs: 4                  # Model runs at least this many epochs before early stop
  gradient_accumulation_steps: 2 # Effective batch = batch_size × accumulation = 64
  max_grad_norm: 1.0
  amp_dtype: "bf16"              # BF16 AMP — requires CUDA; disabled automatically on CPU
  early_stopping_patience: 2
  early_stopping_min_delta: 0.003
```

**Tip:** On CPU-only environments (e.g. Replit), `amp_dtype: bf16` is inert — the trainer detects the absence of CUDA and falls back to float32. Training will be slower but functionally correct.

---

### Step 3 — Run the Training Pipeline

```bash
python main.py
```

Training logs are written to `logs/training.log`. You can monitor progress with:

```bash
tail -f logs/training.log
```

Example log output:
```
2026-05-02 | INFO | Epoch 1/10 | Loss: 0.8234 | Val composite: 0.6812
2026-05-02 | INFO | Epoch 2/10 | Loss: 0.6451 | Val composite: 0.7034
2026-05-02 | INFO | Epoch 3/10 | Loss: 0.5102 | Val composite: 0.7198
2026-05-02 | INFO | Epoch 4/10 | Loss: 0.4308 | Val composite: 0.7241
2026-05-02 | INFO | Epoch 5/10 | Loss: 0.4187 | Val composite: 0.7244
2026-05-02 | INFO | Epoch 6/10 | Loss: 0.4103 | Val composite: 0.7240
2026-05-02 | INFO | Early stopping triggered — best epoch: 5
2026-05-02 | INFO | Checkpoint saved to models/checkpoints/checkpoint.pt
```

---

### Step 4 — Verify the Trained Model

After training, verify the checkpoint exists:

```bash
ls models/checkpoints/
# checkpoint.pt  checkpoint.meta.json
```

Check the API health endpoint:

```bash
curl http://localhost:5000/health
# → { "status": "healthy", ... }
```

Send a test prediction:

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Scientists confirm that regular exercise reduces cardiovascular disease risk in a large-scale longitudinal study published in a peer-reviewed journal."}'
```

---

## Data Pipeline Internals

The training pipeline runs eight stages via `src/data_processing/data_pipeline.py`:

```
Stage 1: Path resolution
  src/data_processing/data_resolver.py
  - Resolves raw data paths from config

Stage 2: Load + validate + clean
  src/data_processing/data_loader.py      — reads CSV files from data/raw/
  src/data_processing/data_validator.py   — null ratio, duplicate, class balance checks
  src/data_processing/data_cleaning.py    — unicode, URL, HTML, lowercasing, min word count

Stage 3: Multi-task validation + label analysis
  src/analysis/multitask_validator.py
  src/analysis/label_analysis.py
  - Validates column presence against data_contracts.py
  - Analyzes per-task label distributions

Stage 4: Leakage check (raw splits, before augmentation)
  src/data_processing/leakage_checker.py
  - Detects cross-split text contamination

Stage 5: Data augmentation (train split only)
  src/data_processing/data_augmentation.py
  - Synonym replacement, random swap, random deletion

Stage 6: Cache write
  src/data_processing/data_cache.py
  - Cache key incorporates tokenizer + max_length + cleaning + augmentation config

Stage 7: Data profiling
  src/data_processing/data_profiler.py
  - Computes distribution statistics for the final processed splits

Stage 8: Build datasets + dataloaders
  src/data_processing/dataset_factory.py
  src/data_processing/dataloader_factory.py
  → PyTorch Datasets ready for the training loop
```

---

## Generating EDA Reports

Run exploratory data analysis on your datasets before training:

```bash
python run_eda.py
```

Reports are saved to `reports/`:
- `data_cleaning_report.json` — cleaning statistics
- `figures/label_distribution.png` — class frequency charts
- `figures/text_length_distribution.png` — article length histogram
- `figures/tfidf_top_terms.png` — most informative terms

---

## Hyperparameter Tuning

Enable automatic hyperparameter search using Optuna:

```yaml
# config/config.yaml
hyperparameter_tuning:
  enabled: true
  trials: 10
  direction: minimize
  metric: eval_loss
  search_space:
    learning_rate:
      min: 1e-6
      max: 5e-5
    batch_size:
      - 16
      - 32
    epochs:
      - 4
      - 8
```

Then run: `python main.py`

Optuna will run the specified number of trials and report the best hyperparameters found.

---

## Cross-Validation

Enable k-fold cross-validation:

```yaml
cross_validation:
  enabled: true
  splits: 5
  metric: eval_loss
```

---

## Resuming Training from a Checkpoint

If training is interrupted, resume from the latest checkpoint:

```yaml
training:
  resume_from_checkpoint: true
```

Checkpoints are saved every `checkpoint_every: 500` steps and up to `max_checkpoints: 3` are retained.

---

## Training on GPU

**CUDA (NVIDIA GPU):**
```yaml
model:
  torch_compile: true
  flash_attention: true

training:
  amp_dtype: "bf16"    # BF16 is preferred over FP16 — avoids overflow issues
```

Verify CUDA is available:
```python
import torch
print(torch.cuda.is_available())  # Should print True
```

**Apple Silicon (MPS):**
```yaml
training:
  amp_dtype: "bf16"    # BF16 is supported on MPS from PyTorch 2.1+
```

**CPU-only (e.g. Replit):**
```yaml
data:
  batch_size: 8        # Reduce batch size to avoid memory pressure
training:
  gradient_accumulation_steps: 4   # Maintain effective batch = 32
  torch_compile: false              # torch.compile may be unstable without CUDA
```

---

## Training from a Different Base Model

To use a larger or different pre-trained model:

```yaml
model:
  encoder: roberta-large            # Or: bert-base-uncased, distilroberta-base
  hidden_dim: 1024                  # roberta-large uses 1024 hidden dim
```

Note: `roberta-large` uses 24 transformer layers vs 12 for `roberta-base` — expect roughly 3× the training time and memory.

---

## Training Only Selected Task Heads

To train on a subset of tasks, remove unused head configurations from `config.yaml`:

```yaml
model:
  heads:
    bias_detection:
      num_labels: 3
      loss: cross_entropy
      label_column: bias_label
    # (remove ideology_detection, propaganda_detection, etc.)
```

Also ensure your dataset only contains the relevant task columns. The `data_contracts.py` schemas control which columns are required per task.

---

## Model Artifacts After Training

After successful training, the following files are created:

| Path                                     | Description                              |
|------------------------------------------|------------------------------------------|
| `models/checkpoints/checkpoint.pt`       | Trained model weights (PyTorch state dict)|
| `models/checkpoints/checkpoint.meta.json`| Training metadata and hyperparameters    |
| `models/tfidf_vectorizer.joblib`         | Fitted TF-IDF vectorizer                 |
| `logs/training.log`                      | Full training log                        |
| `reports/evaluation_results.json`        | Final evaluation metrics per task        |
| `reports/confusion_matrix.png`           | Confusion matrix visualization           |

---

## Evaluation After Training

Evaluate the trained model on the test set:

```bash
python -c "
from src.evaluation.metrics import evaluate_model
results = evaluate_model('data/splits/test.csv')
print(results)
"
```

Metrics computed per task: accuracy, precision, recall, F1. For multi-label tasks (emotion, narrative frames): micro F1, macro F1, ROC-AUC. A `weighted_composite_score` is computed as the task-balanced average of per-task scores.

Results are also written to `reports/evaluation_results.json` automatically after training.
