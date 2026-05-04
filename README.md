# TruthLens AI

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-red)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-4.46.3-yellow)
![FastAPI](https://img.shields.io/badge/API-FastAPI-green)
![Tests](https://img.shields.io/badge/Tests-344%20passing-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

**Multi-task NLP misinformation detection platform powered by RoBERTa.**

TruthLens AI analyses news articles and text passages across six simultaneous detection tasks — media bias, political ideology, propaganda, emotional framing, narrative roles, and narrative frames — using a single shared transformer encoder with independent classification heads. The system is production-deployed via a lightweight HuggingFace Inference API backend, with a full PyTorch training and evaluation stack available for offline research.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Detection Tasks](#detection-tasks)
4. [API Reference](#api-reference)
5. [Quick Start](#quick-start)
6. [Project Structure](#project-structure)
7. [Model Details](#model-details)
8. [Training Pipeline](#training-pipeline)
9. [Feature Engineering](#feature-engineering)
10. [Evaluation](#evaluation)
11. [Explainability](#explainability)
12. [Configuration](#configuration)
13. [Deployment](#deployment)
14. [Testing](#testing)

---

## Overview

TruthLens AI is a research-grade, production-deployed platform for automated credibility assessment of news content. It combines:

- **Transformer-based multi-task learning** — one RoBERTa encoder fine-tuned jointly across six detection objectives, sharing contextual representations while maintaining task-specific classification heads.
- **150+ hand-crafted linguistic features** — nine semantic domains (bias, discourse, emotion, narrative, propaganda, rhetoric, text statistics, graph, psychological) feeding an auxiliary feature branch in the hybrid model variant.
- **5-stage aggregation engine** — rule-based, ML-based, graph-based, temporal, and weighted ensemble aggregation that fuses all detection signals into a single credibility score with uncertainty bounds.
- **Full explainability stack** — LIME token attribution, SHAP values, Integrated Gradients, attention rollout, and multi-method consistency scoring with faithfulness metrics.
- **Production inference layer** — 20-module serving stack with LRU + gzip-disk two-tier caching, latency monitoring, distribution drift detection, and structured audit logging.

**HuggingFace model checkpoints:**
- Primary: [`bhavaygupta2002/truthlens_v1`](https://huggingface.co/bhavaygupta2002/truthlens_v1)
- Secondary: [`bhavaygupta2002/truthlens2`](https://huggingface.co/bhavaygupta2002/truthlens2)

---

## Architecture

```
                     ┌──────────────────────────────────┐
                     │         FastAPI Application        │
                     │    api/app.py  ·  api/main.py      │
                     └──────────────┬───────────────────┘
                                    │ POST /predict
                                    ▼
                     ┌──────────────────────────────────┐
                     │    Inference Layer  (20 modules)   │
                     │  PredictionService                 │
                     │    ├─ InferenceCache (LRU + disk)  │
                     │    ├─ InferenceEngine              │
                     │    │    ├─ AutoTokenizer           │
                     │    │    ├─ AMP autocast            │
                     │    │    └─ Temperature scaling     │
                     │    ├─ PostProcessor                │
                     │    ├─ InferenceMonitor             │
                     │    └─ InferenceLogger              │
                     └──────────────┬───────────────────┘
                                    │
                                    ▼
                     ┌──────────────────────────────────┐
                     │    MultiTaskTruthLensModel         │
                     │  ┌───────────────────────────┐   │
                     │  │   TransformerEncoder        │   │
                     │  │   roberta-base  (125 M)     │   │
                     │  │   pooled_output: (B, 768)   │   │
                     │  └─────────────┬─────────────┘   │
                     │                │                   │
                     │  ┌─────────────┴─────────────┐   │
                     │  │        Task Heads           │   │
                     │  │  bias           (2-class)   │   │
                     │  │  ideology       (3-class)   │   │
                     │  │  propaganda     (2-class)   │   │
                     │  │  narrative      (3-label)   │   │
                     │  │  narrative_frame (5-label)  │   │
                     │  │  emotion        (11-label)  │   │
                     │  └────────────────────────────┘   │
                     └──────────────────────────────────┘
                                    │
                       ┌────────────┴─────────────┐
                       ▼                           ▼
          ┌────────────────────┐     ┌──────────────────────┐
          │  Feature Pipeline  │     │  Aggregation Engine   │
          │  150+ signals      │     │  5-stage fusion       │
          │  9 semantic domains│     │  credibility score    │
          └────────────────────┘     └──────────────────────┘
```

### Core Design Invariants

| Tag | Invariant |
|-----|-----------|
| G4 | Calibration temperature scalars are isolated from the main optimizer and never updated during training |
| N1 | All entropy computations go through `log_softmax` or `logsigmoid` — never `log(p + eps)` |
| P1 | Probabilities, confidence, and entropy are only materialised at inference time, eliminating dead compute in the autograd graph during training |
| C1.3 | All `torch.load` calls use `weights_only=True` |
| A3.4 | Every task head must return a `dict` with at minimum a `"logits"` key |

---

## Detection Tasks

| Task | Head Type | Labels | Description |
|------|-----------|--------|-------------|
| `bias` | ClassificationHead | `non_bias`, `bias` | Media bias presence (2-class) |
| `ideology` | ClassificationHead | `left`, `center`, `right` | Political lean (3-class) |
| `propaganda` | ClassificationHead | `non_propaganda`, `propaganda` | Propaganda detection (2-class) |
| `narrative` | MultiLabelHead | `hero`, `villain`, `victim` | Narrative role assignment (3-label) |
| `narrative_frame` | MultiLabelHead | `RE`, `HI`, `CO`, `MO`, `EC` | Episodic/thematic frames — Realistic, Human-Interest, Conflict, Morality, Economic (5-label) |
| `emotion` | MultiLabelHead | `neutral`, `admiration`, `approval`, `gratitude`, `annoyance`, `amusement`, `curiosity`, `disapproval`, `love`, `optimism`, `anger` | EMOTION-11 schema (11-label) |

All six heads run simultaneously on the same encoded representation. Task subset selection is supported at request time via the `task_types` field.

---

## API Reference

### POST `/predict`

Classify a single text passage across all six tasks.

**Request:**
```json
{
  "text": "Article or passage to analyse.",
  "task_types": ["bias", "emotion"],
  "article_id": "optional-caller-id"
}
```

`task_types` is optional — omitting it runs all six tasks. `article_id` is passed through to logs and cache keys.

**Response:**
```json
{
  "label": "bias",
  "confidence": 0.87,
  "fake_probability": 0.73,
  "task_outputs": {
    "bias": {
      "predictions": [1],
      "probabilities": [[0.13, 0.87]],
      "labels": ["non_bias", "bias"],
      "confidence": 0.87
    },
    "emotion": {
      "probabilities": ["..."],
      "predictions": ["..."],
      "labels": ["..."]
    }
  },
  "article_id": "optional-caller-id",
  "cached": false,
  "processing_time_ms": 142.3
}
```

### POST `/batch_predict`

Run inference on up to 50 texts in one request.

```json
{
  "texts": ["Article one.", "Article two."],
  "task_types": null
}
```

### GET `/health`

Returns system status, model load state, and cache statistics.

### POST `/analyze`

Full article analysis pipeline — runs the complete feature extraction, entity graph analysis, bias profile construction, aggregation, and report generation. Returns a structured `TruthLensAPIResponse` with per-task explanations.

### GET `/tasks`

Returns the list of supported task names and their label vocabularies.

### Additional Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Home — lists all endpoints and confirms status |
| GET | `/project-view` | API metadata and directory structure |
| GET | `/docs` | Interactive Swagger documentation |
| POST | `/report` | Structured credibility report |
| POST | `/cache/clear` | Clear the inference result cache |
| GET | `/calibration/info` | Calibration method descriptions |
| POST | `/calibration/metrics` | Compute ECE, MCE, Brier score, NLL |
| GET | `/ensemble/info` | Ensemble strategy descriptions |
| POST | `/ensemble/predict` | Ensemble prediction (average / weighted / vote) |
| GET | `/export/info` | Export format descriptions |
| POST | `/export/onnx` | Export model to ONNX format |
| POST | `/export/torchscript` | Export model to TorchScript format |

---

## Quick Start

### Development server

```bash
pip install -r requirements.txt
uvicorn api.app:app --host 0.0.0.0 --port 5000 --reload
```

Interactive docs at `http://localhost:5000/docs`.

### Single prediction (curl)

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Breaking: Study shows alarming rise in partisan news coverage."}'
```

### Batch prediction (curl)

```bash
curl -X POST http://localhost:5000/batch_predict \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Article one.", "Article two."]}'
```

### Python SDK

```python
from src.inference.inference_engine import InferenceEngine, InferenceConfig
from src.inference.prediction_service import PredictionService

config = InferenceConfig(model_path="saved_models/truthlens_v1", device="auto")
service = PredictionService(InferenceEngine(config))

result = service.predict("Breaking: Study shows alarming rise in partisan news coverage.")
bias_out = result["task_outputs"]["bias"]
print(bias_out["labels"][bias_out["predictions"][0]])
```

### CLI evaluation

```bash
python -m src.inference.run_inference \
    --model-path saved_models/truthlens_v1 \
    --input data/test/bias.csv \
    --tasks bias ideology propaganda \
    --output reports/
```

---

## Project Structure

```
truthlens/
├── app.py                           # Lightweight production entry (Gunicorn / Render)
├── api/
│   ├── app.py                       # Development FastAPI app (port 5000)
│   └── main.py                      # Full unified API (PyTorch, for research)
├── config/
│   ├── config.yaml                  # Primary runtime configuration
│   └── data_config.yaml             # Data pipeline configuration
├── src/
│   ├── models/                      # Neural network layer (~111 files, 30 sub-packages)
│   │   ├── multitask/               # MultiTaskTruthLensModel (flagship)
│   │   ├── architectures/           # HybridTruthLensModel (encoder + feature fusion)
│   │   ├── base/                    # BaseModel, BaseClassifier, MultiTaskBaseModel
│   │   ├── encoder/                 # TransformerEncoder (HuggingFace AutoModel wrapper)
│   │   ├── heads/                   # ClassificationHead, MultiLabelHead
│   │   ├── loss/                    # MultiTaskLoss, TaskLossRouter, EMA normaliser
│   │   ├── calibration/             # Temperature scaling, isotonic, ECE/MCE metrics
│   │   ├── checkpointing/           # SHA-256 integrity, atomic save, schema migration
│   │   ├── ensemble/                # Stacking, weighted, voting ensembles
│   │   ├── uncertainty/             # MC-Dropout, ensemble disagreement, learned head
│   │   ├── export/                  # ONNX, TorchScript, INT8 quantisation
│   │   ├── adapters/                # LoRA, bottleneck adapters (PEFT)
│   │   └── optimization/            # OptimizerFactory, LR scheduler factory
│   ├── inference/                   # Production serving layer (20 modules)
│   │   ├── inference_engine.py      # AMP inference + calibration core
│   │   ├── prediction_service.py    # Cache + monitor + logger wrapper
│   │   ├── inference_cache.py       # LRU memory + gzip disk two-tier cache
│   │   ├── batch_inference.py       # Bulk throughput engine
│   │   ├── drift_detection.py       # Distribution drift detector
│   │   ├── monitoring.py            # Latency / confidence monitor
│   │   ├── result_formatter.py      # API / dashboard / research output formats
│   │   └── analyze_article.py       # Full-article orchestration
│   ├── training/                    # Complete training engine (15 modules, ~4,700 LOC)
│   │   ├── trainer.py               # Outer epoch loop, early stopping, checkpointing
│   │   ├── training_step.py         # Per-batch forward/backward/clip/step
│   │   ├── loss_engine.py           # Multi-task loss with EMA normalisation
│   │   ├── evaluation_engine.py     # Streaming metrics with DDP-correct reduction
│   │   ├── distributed_engine.py    # DDP init, NCCL/gloo fallback
│   │   ├── hyperparameter_tuning.py # Optuna TPE study, Pareto-front support
│   │   └── cross_validation.py      # StratifiedKFold CV runner
│   ├── data_processing/             # 8-stage data pipeline
│   │   ├── data_pipeline.py         # Orchestrator
│   │   ├── data_loader.py           # Multi-source loader (CSV, JSON, HuggingFace)
│   │   ├── data_cleaner.py          # Unicode normalisation, dedup, length filter
│   │   ├── data_validator.py        # Schema checks, label distribution audit
│   │   ├── data_augmenter.py        # Back-translation, paraphrase, synonym swap
│   │   ├── feature_integrator.py    # Merges hand-crafted features into training rows
│   │   ├── label_processor.py       # Multi-task label alignment and encoding
│   │   └── data_splitter.py         # Stratified / temporal train/val/test splits
│   ├── features/                    # 150+ signals across 70+ files
│   │   ├── bias/                    # Partisan language, source credibility
│   │   ├── discourse/               # Framing, urgency, hedging
│   │   ├── emotion/                 # Sentiment, valence, arousal, EMOTION-11
│   │   ├── narrative/               # Role detection, story structure
│   │   ├── propaganda/              # Rhetorical devices, logical fallacies
│   │   ├── graph/                   # Entity co-occurrence, centrality
│   │   └── text/                    # Readability, TF-IDF, n-gram statistics
│   ├── analysis/                    # 14 linguistic analysers
│   ├── aggregation/                 # 5-stage signal fusion engine
│   ├── evaluation/                  # 23-module evaluation framework
│   │   ├── metrics_engine.py        # Accuracy, F1, calibration, uncertainty metrics
│   │   ├── calibration.py           # Temperature / Platt / isotonic post-hoc scaling
│   │   ├── fairness_auditor.py      # Demographic parity, equalised odds
│   │   ├── threshold_optimizer.py   # Per-task decision threshold search
│   │   └── evaluation_pipeline.py   # End-to-end offline evaluation orchestrator
│   ├── explainability/              # 22-module explainability stack
│   │   ├── explainability_orchestrator.py
│   │   ├── lime_explainer.py
│   │   ├── shap_explainer.py
│   │   ├── integrated_gradients.py
│   │   ├── attention_explainer.py
│   │   └── aggregation.py           # Multi-method fusion + consistency scoring
│   └── utils/                       # Settings, logging, config loading, device helpers
├── data/
│   ├── raw/                         # Per-task raw datasets
│   └── train/ val/ test/            # Processed splits (data/<split>/<task>.csv)
├── saved_models/                    # Checkpointed model artifacts
├── saved_eval/                      # Evaluation artifacts (32 files, gold metrics)
├── reports/                         # Evaluation results, confusion matrices
├── artifacts/                       # Runtime cache, logs, model outputs
├── documentation/                   # Architecture, API, training, deployment guides (11 files)
└── doc2/                            # Deep-dive technical documentation per subsystem (15 files)
```

---

## Model Details

### MultiTaskTruthLensModel

The flagship production model. Built on `roberta-base` (125 M parameters, hidden dimension 768).

**Three construction paths:**

```python
# 1. Raw module injection (testing / research)
model = MultiTaskTruthLensModel(encoder=encoder, task_heads=heads)

# 2. Convenience dataclass (fast unit tests, default heads)
model = MultiTaskTruthLensModel(config=MultiTaskTruthLensConfig(...))

# 3. Full YAML-driven path (model registry, inference engine)
model = MultiTaskTruthLensModel.from_model_config(MultiTaskModelConfig)
```

**Encoder pooling strategies:**

| Mode | Formula |
|------|---------|
| `cls` (default) | `hidden[:, 0]` — CLS token representation |
| `mean` | Masked average over non-padding tokens |
| `attention` | Softmax-weighted sum over the sequence |

**Head output (inference mode):**

| Key | Shape | Description |
|-----|-------|-------------|
| `logits` | `(B, C)` | Raw pre-softmax scores |
| `probabilities` | `(B, C)` | Softmax / sigmoid probabilities |
| `predictions` | `(B,)` or `(B, C)` | Argmax or threshold-based binary labels |
| `confidence` | `(B,)` | Max probability (multiclass) or mean probability (multilabel) |
| `entropy` | `(B,)` | Shannon entropy via `log_softmax` (invariant N1) |

### HybridTruthLensModel

An auxiliary variant that fuses the 768-dim encoder output with a projected hand-crafted feature vector through a learned fusion layer before the task heads.

```
encoder_output (768) ─┐
                       ├─ fusion_linear (GELU) ──► task heads
feature_proj (proj_dim)┘
```

Xavier initialisation is applied to `feature_proj` and `fusion`. Default task head sizes are identical to `MultiTaskTruthLensModel`.

### Post-Hoc Calibration

Applied before logit decoding:

| Method | Description |
|--------|-------------|
| Temperature scaling | Learned scalar `T` divides logits; `T > 1` softens distributions |
| Platt scaling | Linear transform `a·logit + b` fitted on a held-out calibration set |
| Isotonic regression | Non-parametric monotone mapping fitted per-class |

Calibration parameters are isolated from training via `BaseModel.get_calibration_parameters()` (invariant G4).

### Model Export

- **ONNX** — `src/models/export/onnx_export.py` with verification
- **TorchScript** — tracing and scripting via `torchscript_export.py`
- **INT8 quantisation** — dynamic and static quantisation via `quantization.py`

---

## Training Pipeline

### Single-task training

```bash
python main.py --task bias --config config/config.yaml
```

### Multi-task training (production)

```bash
python main.py --multitask --config config/config.yaml
```

The multi-task factory (`create_multitask_trainer_fn`) wires a single shared-encoder model across all six tasks simultaneously, with:

- **EMA loss normalisation** — prevents any one task from dominating the gradient signal.
- **Coverage tracking** — down-weights tasks where label coverage is sparse in a given batch.
- **Adaptive task scheduling** — samples tasks proportionally to their current EMA loss.
- **Three-layer imbalance strategy** — task-level sampling frequency (`TaskScheduler`), within-task data exposure (`WeightedRandomSampler`), and within-task gradient signal (`LossBalancer` with focal loss).

### Key Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| Encoder | `roberta-base` |
| Optimizer | AdamW |
| Learning rate | `3e-5` |
| Weight decay | `0.01` |
| Batch size | `16` |
| Gradient accumulation | `2` |
| Epochs | `4` |
| Scheduler | Linear with 1000 warmup steps |
| AMP dtype | `bfloat16` |
| Max grad norm | `1.0` |
| Early stopping patience | `5` |
| Checkpointing interval | every 500 steps |

### Distributed Training

DDP is supported via `DistributedEngine` with automatic NCCL → gloo fallback on CPU-only hosts. The single-device contract (GPU-1) enforces exactly one `model.to(device)` call before the optimizer is constructed — no parameter re-placement after optimizer creation.

### Hyperparameter Tuning

Optuna TPE sampler with MedianPruner. Supports multi-objective Pareto-front optimisation across learning rate, batch size, epochs, dropout, and weight decay.

### 8-Stage Data Pipeline

| Stage | Module | Responsibility |
|-------|--------|----------------|
| 1 | `data_loader.py` | Multi-source ingestion (CSV, JSON, HuggingFace Hub) |
| 2 | `data_cleaner.py` | Unicode normalisation, dedup, length filter |
| 3 | `data_validator.py` | Schema checks, label distribution audit |
| 4 | `data_augmenter.py` | Back-translation, paraphrase, synonym swap (×1.5) |
| 5 | `feature_integrator.py` | Merges 150+ engineered signals into training rows |
| 6 | `label_processor.py` | Multi-task label alignment and encoding |
| 7 | `data_splitter.py` | Stratified / temporal train/val/test splits |
| 8 | `collate.py` | Dynamic padding via `DataCollatorWithPadding` |

### Training Source Datasets

| Task | Dataset |
|------|---------|
| Fake news | ISOT, LIAR, FakeNewsNet |
| Bias | BABE, BASIL, MBIC |
| Emotion | GoEmotions, SemEval |
| Narrative | FrameNet |
| Propaganda | PTC Propaganda |
| Ideology | AllSides |

---

## Feature Engineering

150+ numeric signals extracted across nine semantic domains:

| Domain | Example Features |
|--------|-----------------|
| **Bias** | Partisan language score, source credibility index, loaded word density |
| **Discourse** | Hedging ratio, urgency markers, evidentiality, concession phrases |
| **Emotion** | Valence, arousal, dominance, per-label EMOTION-11 probabilities |
| **Narrative** | Hero/villain/victim entity counts, story arc completeness |
| **Propaganda** | Rhetorical device inventory, logical fallacy presence, appeal-to-authority markers |
| **Graph** | Entity co-occurrence centrality, community structure, hub entity prominence |
| **Text** | Flesch–Kincaid readability, type-token ratio, sentence length distribution, TF-IDF |
| **Psychological** | Moral foundation scores, epistemic certainty, cognitive complexity |
| **Rhetoric** | Sentiment polarity, subjectivity, framing intensity |

Features are produced by `src/features/` (70+ files across sub-packages) and consumed by:
- `FeaturePreparer` in the inference layer — bridges features into the auxiliary tensor for `HybridTruthLensModel`
- `FeatureIntegrator` in the data pipeline — merges features into training DataFrames
- `AggregationPipeline` — fuses signals into the final credibility score

---

## Evaluation

The 23-module evaluation framework in `src/evaluation/` provides:

### Metrics by Task Type

| Task Type | Metrics |
|-----------|---------|
| Multiclass (`bias`, `ideology`, `propaganda`) | Accuracy, F1-macro, Precision, Recall |
| Multilabel (`narrative`, `narrative_frame`, `emotion`) | F1-micro, F1-macro, Hamming loss, Subset accuracy |

### Calibration Assessment

- Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)
- Reliability diagrams
- Post-hoc recalibration comparison across temperature scaling, Platt scaling, and isotonic regression

### Uncertainty Quantification

- MC-Dropout stochastic forward passes
- Ensemble member disagreement
- Shannon entropy per task (invariant N1)

### Fairness Auditing

- Demographic parity across content-type subgroups
- Equalised odds audit

### Experiment Tracking

`ExperimentTracker` provides a unified facade over MLflow and Weights & Biases, with rank-0-safe distributed logging and automatic run lifecycle management (always closed in `try/finally`).

### Evaluation Artifacts

The `saved_eval/` directory contains 32 pre-computed artefacts including gold metrics, confusion matrices, calibration curves, and per-task threshold optimisation results.

---

## Explainability

The 22-module `src/explainability/` stack provides four complementary attribution methods, fused by a multi-method aggregator with consistency and faithfulness scoring.

| Method | Technique |
|--------|-----------|
| **LIME** | Occlusion-based token importance via linear surrogate models |
| **SHAP** | Shapley value attribution over transformer attention |
| **Integrated Gradients** | Path-integrated gradient attribution from a zero-embedding baseline |
| **Attention rollout** | Layerwise attention propagation across all encoder layers |

**Aggregation:** Per-token importance scores from all methods are fused using configurable weights. A consistency metric (rank correlation between methods) and a faithfulness metric (sufficiency / comprehensiveness) are computed for every explanation.

**Output formats:**

| Format | Use case |
|--------|----------|
| `TruthLensAPIResponse` | Per-task explanation tokens for API consumers |
| `TruthLensDashboardReport` | Visual-ready token highlights for Streamlit dashboard |
| `TruthLensResearchExport` | Full attribution tensors and metadata for offline analysis |

---

## Configuration

The primary configuration file is `config/config.yaml`. The data pipeline reads `config/data_config.yaml`.

### Key YAML Sections

```yaml
model:
  encoder: "roberta-base"
  dropout: 0.1

training:
  epochs: 4
  gradient_accumulation_steps: 2
  early_stopping_patience: 5
  checkpoint_every: 500

optimizer:
  name: "adamw"
  lr: 3.0e-5
  weight_decay: 0.01

scheduler:
  name: "linear"
  warmup_steps: 1000

precision:
  use_amp: true
  amp_dtype: "bf16"

tasks:
  bias: "multiclass"
  ideology: "multiclass"
  propaganda: "multiclass"
  narrative: "multilabel"
  narrative_frame: "multilabel"
  emotion: "multilabel"

task_weights:
  bias: 1.0
  ideology: 1.0
  propaganda: 1.0
  narrative: 1.0
  narrative_frame: 1.0
  emotion: 1.0

tracking:
  backend: "wandb"
  project_name: "truthlens"
```

### Runtime Path Resolution

All output paths resolve under `artifacts/` by default and can be overridden with environment variables:

| Variable | Default |
|----------|---------|
| `TRUTHLENS_MODELS_DIR` | `artifacts/models` |
| `TRUTHLENS_LOGS_DIR` | `artifacts/logs` |
| `TRUTHLENS_CACHE_DIR` | `artifacts/cache` |

### Inference Config (YAML override)

```yaml
# config/inference.yaml
model_path: saved_models/truthlens_v1
device: auto
batch_size: 16
max_length: 512
use_amp: true
calibrate: true
temperature: 1.2
warmup_steps: 3
```

---

## Deployment

### Production (Render / Gunicorn)

The root `app.py` forwards requests to the HuggingFace Inference API, keeping the container under 30 MB with no PyTorch imports at startup.

```bash
gunicorn app:app \
  --workers 1 \
  --worker-class uvicorn.workers.UvicornWorker \
  --timeout 120 \
  --bind 0.0.0.0:$PORT
```

### Self-Hosted (Full PyTorch)

```bash
pip install -r requirements.txt
uvicorn api.app:app --host 0.0.0.0 --port 5000 --reload \
  --reload-dir api --reload-dir src --reload-dir config --reload-dir models
```

### Extending with a New Detection Task

1. Add the task label vocabulary to the relevant sub-package under `src/features/`.
2. Add a head entry to `MultiTaskTruthLensModel._DEFAULT_TASK_SPEC`.
3. Register the task type (`"multiclass"` / `"multilabel"`) in `config/config.yaml::tasks`.
4. Add a dataset entry in `config/data_config.yaml` with the corresponding label column name.

The training factory, loss engine, evaluation engine, and task scheduler all pick up new tasks automatically from the configuration — no code changes required in those layers.

---

## Testing

```bash
pytest tests/ -v
```

**344 tests across 38 modules — all passing.**

| Area | Test Modules |
|------|-------------|
| End-to-end dataset & API | `test_e2e_dataset.py` (108 tests, 13 classes) |
| API endpoints & error paths | `test_api.py`, `test_api_error_paths.py` |
| Aggregation & risk scoring | `test_aggregation.py` |
| Evaluation metrics & uncertainty | `test_evaluation.py`, `test_evaluation_metrics.py` |
| Explainability (SHAP / LIME) | `test_explainability.py`, `test_shap_explainer.py` |
| Emotion lexicon analysis | `test_emotion.py` |
| Input validation | `test_input_validation.py` |
| Model architecture & registry | `test_model_subpackage_imports.py`, `test_model_registry.py` |
| Model training & tokenisation | `test_model_training.py`, `test_tokenization.py` |
| Training pipeline & cross-validation | `test_training_pipeline.py` |
| Inference speed & prediction stability | `test_inference_speed.py`, `test_prediction_stability.py` |
| Data pipelines & schema | `test_data_pipeline_module.py`, `test_dataset_schema.py` |
| Configuration loading | `test_config_loading.py`, `test_config_integrity.py` |
| Reproducibility (seed control) | `test_reproducibility.py` |
| Utility functions | `test_utils.py` |
| Project structure | `test_project_structure.py` |

---

## License

MIT License

---

## Citation

If you use TruthLens AI in your research, please cite:

```bibtex
@software{truthlens2025,
  title  = {TruthLens AI: Multi-Task NLP Misinformation Detection},
  author = {Bhavay Gupta},
  year   = {2025},
  url    = {https://huggingface.co/bhavaygupta2002/truthlens_v1}
}
```
