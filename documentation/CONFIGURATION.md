# Configuration Reference

This document describes all configuration options available in TruthLens AI.

Configuration is split across two YAML files:
- `config/config.yaml` — model architecture, training, API, and inference settings
- `config/data_config.yaml` — data pipeline, preprocessing, augmentation, and EDA settings

Settings are loaded via `src/utils/settings.py` (primary interface used by the API and training) and `src/utils/config_loader.py` (low-level loader with dataclass conversion).

---

## `config/config.yaml`

### `model`

Controls the base transformer encoder and architectural settings.

```yaml
model:
  encoder: roberta-base              # HuggingFace model ID or local path
  hidden_dim: 768                    # Hidden dimension (768 for roberta-base)
  dropout: 0.1                       # Dropout rate applied in task heads
  gradient_checkpointing: true       # Trade compute for memory during training
  flash_attention: true              # Use flash attention when available
  torch_compile: true                # torch.compile at training time
  compile_mode: "default"            # torch.compile mode ("default", "reduce-overhead", "max-autotune")
```

| Key                      | Default        | Description                                     |
|--------------------------|----------------|-------------------------------------------------|
| `encoder`                | `roberta-base` | Encoder model identifier                        |
| `hidden_dim`             | `768`          | Encoder hidden dimension                        |
| `dropout`                | `0.1`          | Dropout applied in classification heads         |
| `gradient_checkpointing` | `true`         | Reduces VRAM at cost of extra forward passes    |
| `flash_attention`        | `true`         | Faster attention kernel (requires compatible GPU)|
| `torch_compile`          | `true`         | Enables `torch.compile` for training speed      |
| `compile_mode`           | `"default"`    | Compilation strategy for torch.compile          |

**Note on CPU-only environments:** `flash_attention` and `torch_compile` require CUDA for full benefit. On CPU-only hosts (e.g. Replit), these settings are accepted by the config loader but their effects are environment-dependent.

---

### `model.heads`

Defines the task-specific classification heads attached to the shared encoder.

```yaml
model:
  heads:
    bias_detection:
      num_labels: 3          # left / center / right
      loss: cross_entropy
      label_column: bias_label

    ideology_detection:
      num_labels: 3
      loss: cross_entropy
      label_column: ideology_label

    propaganda_detection:
      num_labels: 2          # Binary: propaganda vs not
      loss: cross_entropy
      label_column: propaganda_label

    emotion_detection:
      num_labels: 11         # EMOTION-11 schema (emotion_0 … emotion_10)
      type: multi_label
      loss: binary_cross_entropy
      label_prefix: emotion_ # Positional columns: emotion_0, emotion_1, ..., emotion_10

    narrative_roles:
      hero: hero             # Binary column for hero label
      villain: villain
      victim: victim
      loss: binary_cross_entropy

    frame_detection:
      labels:
        - CO                 # Conflict
        - EC                 # Economic
        - HI                 # Human Interest
        - MO                 # Moral
        - RE                 # Resolution
      loss: binary_cross_entropy
```

**Emotion column naming:** The `label_prefix: emotion_` combined with `num_labels: 11` means the dataset must have columns named `emotion_0` through `emotion_10`. Legacy column names like `emotion_joy` or `emotion_fear` are rejected by the data contracts validator. See `src/data_processing/data_contracts.py` for the authoritative schema.

---

### `model.path`

Path where the trained model artifacts are saved and loaded from.

```yaml
model:
  path: models/truthlens_model
```

---

### `checkpoint`

Checkpoint lifecycle settings.

```yaml
checkpoint:
  dir: "checkpoints/"          # Directory for checkpoint files
  max_checkpoints: 3           # Maximum number of checkpoints to retain
```

Checkpoint files: `checkpoint.pt` (weights) and `checkpoint.meta.json` (metadata).

---

### `data`

Dataset loading and batching settings.

```yaml
data:
  batch_size: 32               # Per-device batch size
  num_workers: 8               # DataLoader worker processes
  pin_memory: true             # Pin tensors to page-locked memory (GPU only)
  shuffle: true                # Shuffle training data each epoch
```

| Key          | Default | Description                                     |
|--------------|---------|-------------------------------------------------|
| `batch_size` | `32`    | Samples per gradient step                       |
| `num_workers`| `8`     | Parallel DataLoader workers                     |
| `pin_memory` | `true`  | GPU memory transfer optimization                |
| `shuffle`    | `true`  | Randomize training sample order                 |

---

### `training`

Controls all training hyperparameters.

```yaml
training:
  epochs: 10                         # Maximum training epochs (upper bound)
  min_epochs: 4                      # Minimum epochs before early stopping activates
  log_every: 50                      # Log metrics every N steps
  eval_every: 1                      # Evaluate every N epochs
  checkpoint_every: 500              # Save checkpoint every N steps

  gradient_accumulation_steps: 2     # Effective batch = batch_size × accumulation = 64
  max_grad_norm: 1.0                 # Hard L2 norm cap on gradient before optimizer step

  amp_dtype: "bf16"                  # AMP dtype: "bf16" | "fp16" (bf16 preferred; CPU = no-op)
  allow_tf32: true                   # Enable TF32 on Ampere+ GPUs

  grad_scaler_init_scale: 1024       # Initial loss scale for GradScaler (fp16 only)

  early_stopping_patience: 2        # Stop after N epochs without improvement
  early_stopping_min_delta: 0.003   # Minimum improvement threshold to reset patience counter

  resume_from_checkpoint: false      # Resume training from latest checkpoint
```

| Key                          | Default     | Description                                          |
|------------------------------|-------------|------------------------------------------------------|
| `epochs`                     | `10`        | Maximum training epochs                              |
| `min_epochs`                 | `4`         | Early stopping not active before this epoch          |
| `gradient_accumulation_steps`| `2`         | Effective batch = `data.batch_size × steps` = 64    |
| `max_grad_norm`              | `1.0`       | Gradient clipping norm                               |
| `amp_dtype`                  | `"bf16"`    | Automatic mixed precision dtype                      |
| `early_stopping_patience`    | `2`         | Epochs without improvement before stopping           |
| `early_stopping_min_delta`   | `0.003`     | Min score change counted as improvement              |

**Early stopping metric:** `weighted_composite_score` — a task-balanced weighted average of per-task validation scores, injected into `val_metrics` by the Trainer after each evaluation pass. This ensures early stopping tracks multi-task generalization rather than the loss of the dominant head.

---

### `features`

Feature engineering settings.

```yaml
features:
  engineered_text_column: engineered_text    # Column name for TF-IDF engineered text

  tfidf:
    enabled: true
    max_features: 5000         # Maximum TF-IDF vocabulary size
    top_terms_per_doc: 4       # Top TF-IDF terms to prepend to each text
```

---

### `cross_validation`

Cross-validation settings (disabled by default).

```yaml
cross_validation:
  enabled: false
  splits: 5
  metric: eval_loss
```

---

### `hyperparameter_tuning`

Optuna-based hyperparameter search (disabled by default).

```yaml
hyperparameter_tuning:
  enabled: false
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

---

### `evaluation`

Metrics computed during and after training.

```yaml
evaluation:
  metrics:
    classification:
      - accuracy
      - precision
      - recall
      - f1
    multi_label:
      - micro_f1
      - macro_f1
      - roc_auc
```

---

### `logging`

Logging and checkpoint configuration.

```yaml
logging:
  log_level: INFO
  training_log_path: logs/training.log

  save_steps: 500           # Save checkpoint every N steps
  eval_steps: 500           # Run validation every N steps
  save_total_limit: 3       # Keep only last N checkpoints
```

---

### `paths`

Output directories and artifact file paths.

```yaml
paths:
  models_dir: models
  logs_dir: logs
  reports_dir: reports

  tfidf_vectorizer_path: models/tfidf_vectorizer.joblib

  evaluation_results_path: reports/evaluation_results.json
  confusion_matrix_path: reports/confusion_matrix.png
  cleaning_report_path: reports/data_cleaning_report.json
```

---

### `api`

FastAPI application metadata.

```yaml
api:
  title: TruthLens AI API
  description: Multi-task NLP system for bias, ideology, propaganda, emotion, and narrative analysis
  version: 2.0.0
  text_preview_chars: 100    # Characters of input text to show in API responses
```

---

### `inference`

Runtime inference settings.

```yaml
inference:
  batch_size: 16              # Batch size for batch inference calls
  device: auto                # Device for inference ("auto", "cpu", "cuda", "mps")
  allow_raw_text_fallback: true   # Use raw text if TF-IDF vectorizer is unavailable

  return_outputs:             # Toggle which outputs to include in full analysis
    bias: true
    ideology: true
    propaganda: true
    emotion: true
    narrative_roles: true
    frames: true
```

---

### `distributed`

Distributed training settings (DDP).

```yaml
distributed:
  use_ddp: true
  backend: "gloo"              # "gloo" works on CPU and GPU; switch to "nccl" for multi-GPU CUDA
  find_unused_parameters: false
```

**Note:** `"gloo"` is required for CPU-only environments. `"nccl"` provides higher bandwidth on multi-GPU CUDA hardware but will fail on CPU.

---

## `config/data_config.yaml`

### `project`

Project metadata (informational).

```yaml
project:
  name: TruthLens AI
  version: 2.0
  description: Multi-task NLP pipeline for bias, ideology, propaganda, emotion, narrative roles, and frame detection.
```

---

### `dataset.unified_schema`

Defines the column mapping used when merging datasets.

```yaml
dataset:
  unified_schema:
    text_fields:
      - title
      - text
    label_fields:
      bias_label: bias_label
      ideology_label: ideology_label
      propaganda_label: propaganda_label
      frame: frame
      hero: hero
      villain: villain
      victim: victim
      emotion_prefix: emotion_    # Positional columns: emotion_0, emotion_1, ..., emotion_10
      CO: conflict
      EC: economic
      HI: human_interest
      MO: moral
      RE: resolution
```

**Emotion column convention:** `emotion_prefix: emotion_` produces columns `emotion_0` through `emotion_10` (one per EMOTION-11 label in canonical order). These are integer-indexed, not name-indexed. See `src/data_processing/data_contracts.py` for the enforced schema.

---

### `dataset.datasets`

Per-task dataset source directories.

```yaml
dataset:
  datasets:
    bias:
      path: data/raw/bias
      text_column: text
      label_column: bias_label

    ideology:
      path: data/raw/ideology
      text_column: text
      label_column: ideology_label

    propaganda:
      path: data/raw/propaganda
      text_column: text
      label_column: propaganda_label

    narrative:
      path: data/raw/narrative
      text_column: text
      hero_entities: hero
      villain_entities: villain
      victim_entities: victim

    emotion:
      path: data/raw/emotion
      text_column: text
      emotion_columns_prefix: emotion_   # Columns: emotion_0 … emotion_10
```

---

### `validation`

Data quality validation thresholds applied before training.

```yaml
validation:
  required_columns:
    - text
  max_null_ratio: 0.10        # Reject dataset if >10% of values are null
  max_duplicate_ratio: 0.15   # Reject dataset if >15% of rows are duplicates
  min_text_length: 20         # Minimum character count per row
  min_word_count: 30          # Minimum word count per row
  label_checks:
    min_class_ratio: 0.10     # Each class must make up at least 10% of labels
    allow_missing_labels: true
```

---

### `cleaning`

Text cleaning operations applied to all datasets.

```yaml
cleaning:
  normalize_unicode: true    # Convert unicode characters to ASCII equivalents
  normalize_numbers: true    # Normalize numeric formats
  remove_emojis: true        # Strip emoji characters
  remove_urls: true          # Remove http/https URLs
  remove_html: true          # Strip HTML tags
  expand_contractions: true  # Expand "don't" → "do not", etc.
  lowercase: true
  strip_whitespace: true
  min_word_count: 30         # Drop rows with fewer than N words after cleaning
```

---

### `balancing`

Class imbalance handling.

```yaml
balancing:
  enabled: true
  method: oversample    # "oversample", "undersample", or "smote"
  random_state: 42
```

---

### `augmentation`

Text augmentation techniques to expand the training set.

```yaml
augmentation:
  enabled: true
  multiplier: 2    # Generate N augmented copies per original sample

  techniques:
    synonym_replacement: true
    random_swap: true
    random_deletion: true
    back_translation: false    # Disabled by default (requires translation API)
```

---

### `split`

Dataset split ratios.

```yaml
split:
  train_ratio: 0.70
  validation_ratio: 0.15
  test_ratio: 0.15
  stratified: true          # Maintain class proportions in each split
  random_state: 42
```

---

### `eda`

Exploratory Data Analysis settings.

```yaml
eda:
  enabled: true
  figures_dir: reports/figures
  top_words: 30
  max_tfidf_features: 10000
  ngrams:
    - 1
    - 2
    - 3
  plots:
    label_distribution: true
    text_length_distribution: true
    tfidf_top_terms: true
    emotion_distribution: true
```

Run EDA with: `python run_eda.py`

---

### `output`

Output file paths for processed datasets.

```yaml
output:
  unified_dataset: data/processed/unified_dataset.csv
  processed_data_dir: data/processed
  splits_dir: data/splits
  logs_dir: logs
  save_formats:
    - csv
    - parquet
```

---

## Accessing Configuration in Code

### Primary interface — `src/utils/settings.py`

Used by the API and main training pipeline:

```python
from src.utils.settings import load_settings

settings = load_settings()
print(settings.model.path)       # Path object to model directory
print(settings.training.epochs)  # Integer
print(settings.api.title)        # "TruthLens AI API"
```

### Low-level interface — `src/utils/config_loader.py`

Direct YAML access with nested key retrieval:

```python
from src.utils.config_loader import load_config, get_config_value

config = load_config()
max_len = get_config_value(config, "model", "encoder", "max_length", default=512)
```

### Structured dataclass — `load_app_config()`

Converts YAML to typed dataclasses:

```python
from src.utils.config_loader import load_app_config

app_config = load_app_config()
print(app_config.model.name)           # "roberta-base"
print(app_config.training.batch_size)  # 32
```

Both loaders use `@lru_cache` — the YAML file is read from disk once per process.
