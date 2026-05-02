# Architecture

This document describes the **system architecture of TruthLens AI**.

TruthLens AI is a **multi-layer machine learning platform** designed for misinformation detection, credibility analysis, linguistic signal extraction, explainable AI, and scalable inference via REST API.

---

## High-Level Architecture

TruthLens processes news articles through eight analytical layers that each contribute signals to the final credibility evaluation:

```
News Article Input
       ↓
┌─────────────────────────────────────────────────────┐
│  1. Preprocessing & Text Cleaning                   │
└─────────────────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────────────────┐
│  2. Feature Engineering                             │
│     Lexical · Bias · Emotion · Narrative ·          │
│     Propaganda · Discourse · Graph                  │
└─────────────────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────────────────┐
│  3. MultiTask Transformer Model (RoBERTa)           │
│     Bias · Ideology · Propaganda ·                  │
│     Emotion (11-label) · Narrative Roles ·          │
│     Narrative Frames                                │
└─────────────────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────────────────┐
│  4. Linguistic Analysis Modules                     │
│     14 analyzers registered via AnalyzerRegistry   │
│     Rhetoric · Argument · Discourse · Framing ·     │
│     Ideology · Narrative · Source Attribution       │
└─────────────────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────────────────┐
│  5. Graph Reasoning                                 │
│     Entity Graphs · Narrative Graphs ·              │
│     Propagation Detection                           │
└─────────────────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────────────────┐
│  6. Explainability Layer                            │
│     SHAP · LIME · Attention Rollout ·               │
│     Token Attribution                               │
└─────────────────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────────────────┐
│  7. Score Aggregation Engine                        │
│     FeatureMapper · WeightManager ·                 │
│     TruthLensScoreCalculator · RiskAssessment ·     │
│     ScoreExplainer                                  │
└─────────────────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────────────────┐
│  8. API Response                                    │
│     FastAPI · JSON output · Swagger docs            │
└─────────────────────────────────────────────────────┘
```

---

## System Components

| Layer              | Location                  | Purpose                                |
|--------------------|---------------------------|----------------------------------------|
| Data Layer         | `src/data_processing/`    | Dataset ingestion and preprocessing    |
| Feature Layer      | `src/features/`           | Structured feature extraction          |
| Model Layer        | `src/models/`             | Transformer and multitask models       |
| Analysis Layer     | `src/analysis/`           | Deep linguistic and narrative analysis |
| Graph Layer        | `src/graph/`              | Entity and narrative graph reasoning   |
| Explainability     | `src/explainability/`     | Interpretable prediction explanations  |
| Aggregation Layer  | `src/aggregation/`        | Credibility scoring and risk levels    |
| Inference Layer    | `src/inference/`          | Production inference pipeline          |
| API Layer          | `api/app.py`              | FastAPI REST service                   |

---

## Data Layer

Responsible for **dataset ingestion, validation, and preprocessing**.

```
Raw Datasets (bias, emotion, ideology, narrative, propaganda)
       ↓
Stage 1: Path resolution (data_resolver.py)
       ↓
Stage 2: Load + validate + clean raw CSVs (data_loader, data_validator, data_cleaning)
       ↓
Stage 3: Multi-task validation + label analysis
       ↓
Stage 4: Leakage check on raw splits (before augmentation)
       ↓
Stage 5: Data augmentation (train split only)
       ↓
Stage 6: Cache write
       ↓
Stage 7: Data profiling
       ↓
Stage 8: Build datasets + dataloaders
       ↓
Unified Dataset → Train / Validation / Test Splits (70% / 15% / 15%)
```

Key modules in `src/data_processing/`:
- `data_pipeline.py` — 8-stage orchestrator (single entry point)
- `data_contracts.py` — canonical task schemas (single source of truth for label columns)
- `data_cleaning.py` — unicode normalization, URL removal, HTML stripping
- `data_validator.py` — null ratio, duplicate ratio, class balance checks
- `data_augmentation.py` — synonym replacement, random swap, random deletion
- `leakage_checker.py` — cross-split contamination detection

---

## Feature Engineering Layer

Transforms raw article text into **structured, interpretable features**.

```
Article Text
       ↓
Tokenization
       ↓
┌──────────────────────────────────────────────┐
│  Parallel Feature Extractors                 │
│  ├── Lexical (token counts, diversity)       │
│  ├── Semantic (contextual meaning)           │
│  ├── Syntactic (grammar, structure)          │
│  ├── Bias (lexicon density, framing)         │
│  ├── Emotion (11-label intensity scoring)    │
│  ├── Narrative (hero/villain/victim roles)   │
│  ├── Propaganda (manipulation patterns)      │
│  └── Graph (entity interaction signals)      │
└──────────────────────────────────────────────┘
       ↓
Feature Fusion & Scaling
       ↓
Unified Feature Representation
```

All feature extractors inherit from `BaseFeature` and receive a `FeatureContext` object as input. The `FeatureContext` class (defined in `src/analysis/feature_context.py`) is spaCy-backed for NER and syntactic annotation.

---

## Model Layer — MultiTask Architecture

TruthLens uses a **shared RoBERTa encoder with six task-specific heads** (`MultiTaskTruthLensModel`).

```
Article Text
       ↓
Tokenizer (roberta-base, max_length=512)
       ↓
Shared RoBERTa Encoder (roberta-base, 12 layers, 768 hidden dim)
       ↓
┌──────────────────────────────────────────────────────────────┐
│  Task-Specific Heads (nn.ModuleDict)                         │
│  ├── Bias Detection Head      (3-class: left/center/right)   │
│  ├── Ideology Detection Head  (3-class)                      │
│  ├── Propaganda Head          (binary)                       │
│  ├── Emotion Head             (11-label multi-label)         │
│  │     emotion_0…emotion_10 in dataset                       │
│  ├── Narrative Roles Head     (hero/villain/victim)          │
│  └── Narrative Frame Head     (CO/EC/HI/MO/RE)              │
└──────────────────────────────────────────────────────────────┘
       ↓
Softmax / Sigmoid per head
       ↓
Per-task Probabilities
```

**Multi-task learning advantages:**
- Single forward pass produces outputs for all six tasks
- Shared encoder learns unified semantic representations
- Reduced training cost compared to six separate models
- Improved generalization from cross-task regularization

**Inference path** (`models/inference/predictor.py`):
- Model is loaded once and cached in memory
- Device-aware tensor routing (CPU / CUDA / MPS)
- `model.eval()` ensures dropout is disabled during inference

---

## Linguistic Analysis Layer

Performs deeper structural analysis beyond model probabilities. The `AnalyzerRegistry` holds 14 named analyzers, constructed via `build_default_registry()` in `src/analysis/analysis_registry.py`. The `AnalysisPipeline` runs them in registration order against a shared `FeatureContext`.

| Order | Registry Key             | Analyzer Class                     | Signal Type                  |
|-------|--------------------------|------------------------------------|------------------------------|
| 1     | `rhetorical`             | `RhetoricalDeviceDetector`         | Hyperbole, loaded phrases    |
| 2     | `argument`               | `ArgumentMiningAnalyzer`           | Claim–evidence structure     |
| 3     | `context`                | `ContextOmissionDetector`          | Missing/selective facts      |
| 4     | `discourse`              | `DiscourseCoherenceAnalyzer`       | Argument consistency         |
| 5     | `emotion`                | `EmotionTargetAnalyzer`            | Emotion target entities      |
| 6     | `framing`                | `FramingAnalyzer`                  | Narrative frame strategy     |
| 7     | `information`            | `InformationDensityAnalyzer`       | Lexical density signals      |
| 8     | `information_omission`   | `InformationOmissionDetector`      | Omitted context signals      |
| 9     | `ideology`               | `IdeologicalLanguageDetector`      | Political ideology markers   |
| 10    | `narrative_role`         | `NarrativeRoleExtractor`           | Hero/villain/victim roles    |
| 11    | `narrative_conflict`     | `NarrativeConflictAnalyzer`        | Adversarial framing          |
| 12    | `narrative_propagation`  | `NarrativePropagationAnalyzer`     | Propagation patterns         |
| 13    | `narrative_temporal`     | `NarrativeTemporalAnalyzer`        | Timeline consistency         |
| 14    | `source`                 | `SourceAttributionAnalyzer`        | Source credibility signals   |

**Critical:** analyzer registry keys must match exactly. Use `build_default_registry()` — do not instantiate `AnalyzerRegistry` manually.

---

## Graph Reasoning Layer

Constructs graphs representing relationships between entities and narrative elements.

```
Entity Extraction (via spaCy NER through FeatureContext)
       ↓
Graph Construction (NetworkX)
       ↓
Graph Embeddings
       ↓
Topological Feature Extraction
       ↓
Graph-Based Credibility Signals
```

Graph types:
- **Entity graphs** — person/organization/location relationships
- **Narrative graphs** — story arc propagation and conflict detection
- **Temporal graphs** — event ordering and timeline consistency

---

## Explainability Layer

Provides interpretable explanations for every prediction.

| Method               | What it explains                              |
|----------------------|-----------------------------------------------|
| SHAP                 | Global feature importance across predictions  |
| LIME                 | Local token-level explanation per article     |
| Attention Rollout    | Transformer attention attribution             |
| Emotion Explainer    | Lexicon-based emotional signal breakdown      |
| Bias Explainer       | Per-sentence bias heatmap                     |

All explanations are returned as structured JSON in the `/analyze` endpoint response.

---

## Aggregation and Scoring

The aggregation engine combines all signals into a **single normalized credibility score** through a five-stage pipeline:

```
Stage 1: FeatureMapper
  Maps raw model outputs + analysis signals to scored feature groups
       ↓
Stage 2: WeightManager (adaptive)
  Applies WEIGHT_GROUPS (defined in aggregation_config.py — single source of truth)
  Groups: "manipulation", "credibility", "final"
  Weights are confidence + entropy-driven at inference time
       ↓
Stage 3: TruthLensScoreCalculator
  Combines weighted group scores into TruthLens Credibility Score
       ↓
Stage 4: RiskAssessment
  Low (0.0–0.33) · Medium (0.34–0.66) · High (0.67–1.0)
       ↓
Stage 5: ScoreExplainer
  Generates human-readable explanation of score components
```

---

## API Layer

The system exposes all inference capabilities through a **FastAPI service** running on port 5000.

**Quick predict:**
```
POST /predict
{ "text": "Breaking news: ..." }

→ { "text": "...", "prediction": "FAKE", "fake_probability": 0.87, "confidence": 0.87 }
```

**Full analysis:**
```
POST /analyze
{ "text": "Breaking news: ..." }

→ {
    "prediction": "FAKE",
    "fake_probability": 0.87,
    "confidence": 0.87,
    "bias": { "bias_score": 0.12, "media_bias": "lean", "biased_tokens": [...], "sentence_heatmap": [...] },
    "emotion": { "dominant_emotion": "annoyance", "emotion_scores": {...}, "emotion_distribution": {...} },
    "explainability": { "emotion_explanation": {...}, "lime": {...} }
  }
```

See [API_REFERENCE.md](API_REFERENCE.md) for the complete endpoint specification.

---

## Training Architecture

```
Training Dataset (train split from data_pipeline.py)
       ↓
MultiTaskTruthLensModel (roberta-base encoder + 6 heads)
       ↓
Combined Multi-Task Loss (cross-entropy / binary_cross_entropy per head)
       ↓
AdamW Optimizer + Linear Warmup Scheduler
       ↓
BF16 Automatic Mixed Precision (amp_dtype: bf16)
       ↓
Gradient Clipping (max_grad_norm=1.0)
       ↓
Early Stopping (patience=2, min_delta=0.003, metric=weighted_composite_score)
  — runs at least min_epochs=4, then up to epochs=10 if improving
       ↓
Checkpoint saved to models/checkpoints/checkpoint.pt
```

Key training hyperparameters (from `config/config.yaml`):
- `data.batch_size: 32`
- `training.gradient_accumulation_steps: 2` → effective batch = 64
- `training.max_grad_norm: 1.0`
- `model.gradient_checkpointing: true`
- `model.torch_compile: true`

Configuration: `config/config.yaml` · Entry point: `python main.py`

---

## Inference Architecture

```
Incoming Request → FastAPI (api/app.py)
       ↓
Input Validation (min 10 chars, max 10,000 chars)
       ↓
ModelRegistry.load_model() [cached after first call]
       ↓
Tokenizer → RoBERTa Encoder → Softmax → Probabilities
       ↓
Parallel: BiasLexiconFeatures · EmotionLexiconAnalyzer · LIME
       ↓
JSON Response Assembly
       ↓
HTTP Response
```

---

## Deployment Architecture

TruthLens is deployed on **Replit Autoscale**.

```
Client Request
       ↓
Replit Autoscale (HTTPS, mTLS proxy)
       ↓
Gunicorn (UvicornWorker, port 5000, --workers 1 --timeout 120)
       ↓
FastAPI Application
       ↓
Inference Pipeline
       ↓
JSON Response
```

Production run command:
```
gunicorn --bind=0.0.0.0:5000 --reuse-port --workers 1 --timeout 120 \
  -k uvicorn.workers.UvicornWorker api.app:app
```

Development run command:
```
python -m uvicorn api.app:app --host 0.0.0.0 --port 5000 --reload \
  --reload-dir api --reload-dir src --reload-dir config --reload-dir models
```

---

## Design Principles

**Modularity** — Every subsystem is independently replaceable. Feature extractors, model heads, and analysis modules all plug into shared base classes.

**Scalability** — Batch inference, feature caching, and configurable pipeline steps support large-scale processing.

**Interpretability** — SHAP, LIME, and attention-based explanations are built into the core inference path, not added as afterthoughts.

**Reproducibility** — Fixed seeds, YAML configuration, and a comprehensive test suite ensure consistent results across runs.

**Fail-Safe Inference** — The API degrades gracefully: LIME errors are caught and returned as structured error objects; the `/health` endpoint reports model availability without crashing.
