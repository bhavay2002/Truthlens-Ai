# TruthLens AI — Project Report

---

## 1. Title Page

| Field        | Detail                                                                 |
|--------------|------------------------------------------------------------------------|
| **Project**  | TruthLens AI — Multi-layer Misinformation Detection System             |
| **Type**     | Full-Stack Machine Learning Application                                |
| **Stack**    | Python 3.12 · PyTorch · HuggingFace Transformers · FastAPI · React 19 |
| **Version**  | 2.0.0                                                                  |

---

## 2. Abstract

TruthLens AI is a multi-layer misinformation detection system that performs simultaneous structural and linguistic analysis of news articles across six orthogonal detection dimensions: political bias, emotional manipulation, propaganda patterns, ideological framing, narrative role assignment, and narrative frame classification.

The core model is a multi-task fine-tuned RoBERTa-base encoder (~125M parameters) with six task-specific classification heads trained jointly on five domain-specific datasets (BABE/BASIL/MBIC for bias, GoEmotions 11-label subset for emotion, PTC for propaganda, AllSides for ideology, and FrameNet for narrative). Training uses AdamW optimisation with BF16 automatic mixed precision, gradient accumulation to an effective batch size of 64, and task-balanced early stopping via a `weighted_composite_score` metric rather than raw loss — preventing the dominant (easier) heads from masking under-convergence in harder tasks such as ideology classification.

Beyond the transformer, TruthLens adds a 14-analyzer linguistic pipeline (`AnalyzerRegistry`) covering argument mining, rhetorical devices, discourse coherence, contextual omission, source attribution, narrative propagation, and temporal consistency. Signals from both layers are combined by a five-stage adaptive aggregation engine into a final Credibility Score with risk classification.

The system is served through a FastAPI REST API (port 5000) and consumed by a React 19 / Vite 8 frontend that renders a full six-panel explainability dashboard. All inference is CPU-compatible and deployed to Replit Autoscale via a single-worker Gunicorn + UvicornWorker configuration.

---

## 3. Introduction

The proliferation of online misinformation represents one of the most consequential challenges of the information age. Studies estimate that false news stories spread six times faster than true ones on social media, and readers frequently lack the linguistic tools to identify the structural hallmarks of manipulative content — emotionally loaded vocabulary, selective omission of context, partisan framing, and fear-based rhetoric.

Existing fact-checking tools address this inadequately. Most offer a binary "true/false" label with no explanation of *why* or *how* content is misleading. Rule-based systems catch surface patterns but miss contextual nuance; single-task classifiers trained only on fake/real labels ignore the rich orthogonal signals — bias, emotion, propaganda, ideology — that together constitute the full picture of a misleading article.

TruthLens AI was designed to close this gap. Rather than returning a verdict, it returns an *explanation*: a structured, interpretable breakdown of every manipulative signal detected in the text, from individual bias-loaded tokens to narrative arc classification and LIME-derived token importance weights. The platform is aimed at journalists, researchers, educators, and technically literate general users who need not just a label but a comprehensible rationale.

---

## 4. Problem Statement

**Given a natural-language news article (English, 10–10,000 characters), classify its credibility along six simultaneous dimensions and produce a ranked, human-readable explanation of the signals that drove the assessment.**

Formal input–output specification:

```
Input:  text  →  str  (news article or claim)

Output:
  prediction           →  REAL | FAKE
  fake_probability     →  float  [0.0, 1.0]
  confidence           →  float  [0.0, 1.0]
  bias_score           →  float  [0.0, 1.0] + media_bias label + sentence heatmap
  emotion              →  dominant label (EMOTION-11) + per-label scores
  ideology             →  3-class distribution
  propaganda           →  binary + intensity
  narrative_roles      →  hero / villain / victim multi-label
  narrative_frames     →  CO / EC / HI / MO / RE multi-label
  explainability       →  LIME token importance + SHAP feature attribution
  credibility_score    →  float  [0.0, 1.0] + risk level (Low / Medium / High)
```

The challenge extends beyond standard text classification: it requires multi-task joint learning, interpretable output at token level, reliable performance without GPU at inference time, and real-time throughput sufficient for a live web application.

---

## 5. Objectives

- Build and train a multi-task RoBERTa model covering six classification tasks simultaneously in a single forward pass
- Design an 8-stage data pipeline with strict schema contracts, leakage detection, augmentation, and caching
- Implement 14 rule-based and hybrid linguistic analyzers as a complementary second analytical layer
- Develop an adaptive five-stage aggregation engine that converts heterogeneous signals into a single interpretable credibility score
- Integrate LIME and SHAP explainability at the inference layer so every prediction is traceable to specific tokens and features
- Expose all capabilities through a FastAPI REST service with sub-second response on `/predict` and graceful degradation on LIME failure
- Build a React frontend that renders a full six-panel interactive analysis dashboard
- Achieve CPU-compatible production deployment on Replit Autoscale with health monitoring

---

## 6. System Overview / Architecture

### 6.1 High-Level Data Flow

```
User Input (news article text)
        │
        ▼
┌─────────────────────────────────────────┐
│  React 19 Frontend (Vite, port 5000)   │
│  InputForm → AnalysisPage → Dashboard  │
└──────────────┬──────────────────────────┘
               │  POST /predict or /analyze
               ▼
┌─────────────────────────────────────────┐
│  FastAPI REST API  (Python, port 5000) │
│  api/app.py                            │
└──────────────┬──────────────────────────┘
               │
       ┌───────┴──────────────────────────┐
       │                                  │
       ▼                                  ▼
┌─────────────────────┐     ┌─────────────────────────────┐
│  ModelRegistry      │     │  Feature Engineering        │
│  RoBERTa-base       │     │  BiasLexiconFeatures        │
│  + 6 Task Heads     │     │  EmotionLexiconAnalyzer     │
│  → Probabilities    │     │  LIME Explainability        │
└──────────┬──────────┘     └────────────┬────────────────┘
           │                             │
           └──────────┬──────────────────┘
                      ▼
        ┌─────────────────────────────────┐
        │  Aggregation Engine             │
        │  FeatureMapper                  │
        │  → WeightManager (adaptive)     │
        │  → TruthLensScoreCalculator     │
        │  → RiskAssessment               │
        │  → ScoreExplainer               │
        └─────────────────────────────────┘
                      │
                      ▼
             JSON API Response
```

### 6.2 ML Pipeline Architecture

```
Raw Datasets
      │
      ▼  src/data_processing/data_pipeline.py  (8 stages)
[1] Path resolution  →  [2] Load/validate/clean  →  [3] Multi-task validation
      │
      ▼
[4] Leakage check  →  [5] Augmentation (train only)  →  [6] Cache write
      │
      ▼
[7] Profiling  →  [8] Dataset/DataLoader build
      │
      ▼
MultiTaskTruthLensModel
  Shared RoBERTa-base encoder (12L, 12H, 768D, ~125M params)
      │
      ├── Head: bias_detection      (3-class cross_entropy)
      ├── Head: ideology_detection  (3-class cross_entropy)
      ├── Head: propaganda          (binary  cross_entropy)
      ├── Head: emotion             (11-label binary_cross_entropy)
      ├── Head: narrative_roles     (3 binary binary_cross_entropy)
      └── Head: narrative_frame     (5 binary binary_cross_entropy)
      │
      ▼
Combined multi-task loss  →  AdamW + BF16 AMP  →  Checkpoint
      │
      ▼  src/analysis/analysis_registry.py  (14 analyzers, run post-inference)
AnalysisPipeline
  [1]  RhetoricalDeviceDetector     [8]  InformationOmissionDetector
  [2]  ArgumentMiningAnalyzer       [9]  IdeologicalLanguageDetector
  [3]  ContextOmissionDetector      [10] NarrativeRoleExtractor
  [4]  DiscourseCoherenceAnalyzer   [11] NarrativeConflictAnalyzer
  [5]  EmotionTargetAnalyzer        [12] NarrativePropagationAnalyzer
  [6]  FramingAnalyzer              [13] NarrativeTemporalAnalyzer
  [7]  InformationDensityAnalyzer   [14] SourceAttributionAnalyzer
      │
      ▼
Aggregation Engine  →  Credibility Score  →  API Response
```

---

## 7. Technology Stack

### ML / Backend

| Layer                  | Technology                          | Version  | Purpose                                              |
|------------------------|-------------------------------------|----------|------------------------------------------------------|
| Language               | Python                              | 3.12     | Entire ML and API stack                              |
| Deep Learning          | PyTorch                             | ≥2.1     | Model training, inference, tensor operations         |
| Transformer Encoder    | HuggingFace Transformers            | latest   | RoBERTa-base tokenizer and encoder weights           |
| API Framework          | FastAPI                             | latest   | REST service, request validation, Swagger docs       |
| API Server (dev)       | Uvicorn                             | latest   | ASGI development server with hot-reload              |
| API Server (prod)      | Gunicorn + UvicornWorker            | latest   | Production ASGI server (1 worker, 120s timeout)      |
| NLP Utilities          | spaCy                               | latest   | NER, POS tagging, dependency parsing in FeatureContext|
| Explainability         | LIME                                | latest   | Local token-level prediction explanations            |
| Explainability         | SHAP                                | latest   | Global Shapley value feature attribution             |
| Vectorization          | scikit-learn (TF-IDF)               | latest   | TF-IDF feature engineering + vectorizer artifact     |
| Data Handling          | pandas                              | latest   | Dataset loading, cleaning, merging                   |
| Hyperparameter Tuning  | Optuna                              | latest   | Automated hyperparameter search (optional)           |
| Configuration          | PyYAML                              | latest   | YAML config loading with typed dataclasses           |

### Frontend

| Technology     | Version | Purpose                                         |
|----------------|---------|-------------------------------------------------|
| React          | 19      | Component-based UI framework                    |
| Vite           | 8       | Build tool and development server               |
| Tailwind CSS   | v4      | Utility-first styling                           |
| Framer Motion  | 12      | Animations and panel transitions                |
| Recharts       | 3       | Data visualisation (bias heatmap, emotion chart)|
| React Router   | v7      | Client-side routing                             |
| Lucide React   | latest  | Icon library                                    |
| Axios          | 1       | HTTP client for API calls                       |

### Node.js Middleware Backend

| Technology | Version | Purpose                                    |
|------------|---------|--------------------------------------------|
| Node.js    | 18+     | JavaScript runtime                         |
| Express    | 4       | HTTP server and routing layer              |

---

## 8. Dataset Description

### 8.1 Data Sources

TruthLens AI is trained on five public benchmark datasets, each covering a distinct detection task:

| Task                   | Dataset(s)                    | Description                                          |
|------------------------|-------------------------------|------------------------------------------------------|
| Bias Detection         | BABE, BASIL, MBIC             | Sentence- and article-level media bias annotations   |
| Emotion Classification | GoEmotions (11-label subset)  | Reddit comments with fine-grained emotion labels     |
| Ideology Detection     | AllSides                      | News articles labelled by outlet political lean      |
| Propaganda Detection   | PTC Propaganda Corpus         | News articles with 18 propaganda technique labels    |
| Narrative Analysis     | FrameNet / narrative datasets | Frame and role annotations for news stories          |

### 8.2 Unified Schema

All datasets are merged into a single unified schema by `src/data_processing/data_contracts.py` — the authoritative source of truth for column names and task types. Key columns in the unified dataset:

| Column            | Type            | Task                    |
|-------------------|-----------------|-------------------------|
| `text`            | string          | All tasks               |
| `bias_label`      | int (0/1/2)     | Bias detection          |
| `ideology_label`  | int (0/1/2)     | Ideology detection      |
| `propaganda_label`| int (0/1)       | Propaganda detection    |
| `emotion_0`       | float binary    | Emotion (neutral)       |
| `emotion_1`       | float binary    | Emotion (admiration)    |
| `emotion_2`–`10`  | float binary    | Emotion (approval … anger)|
| `hero`            | int binary      | Narrative role          |
| `villain`         | int binary      | Narrative role          |
| `victim`          | int binary      | Narrative role          |
| `CO` / `EC` / `HI` / `MO` / `RE` | int binary | Narrative frame    |

### 8.3 Emotion Label Schema (EMOTION-11)

The emotion task uses a reduced 11-label schema derived from GoEmotions. Labels from the original 20-label set that showed poor inter-annotator agreement or semantic overlap were moved to a `_LEGACY_EMOTION_LABELS` constant kept for audit visibility only.

| Index | Label        | Index | Label        |
|-------|--------------|-------|--------------|
| 0     | neutral      | 6     | curiosity    |
| 1     | admiration   | 7     | disapproval  |
| 2     | approval     | 8     | love         |
| 3     | gratitude    | 9     | optimism     |
| 4     | annoyance    | 10    | anger        |
| 5     | amusement    |       |              |

### 8.4 Dataset Splits

| Split      | Ratio | File                         |
|------------|-------|------------------------------|
| Train      | 70%   | `data/splits/train.csv`      |
| Validation | 15%   | `data/splits/validation.csv` |
| Test       | 15%   | `data/splits/test.csv`       |

Splits are stratified by task label to preserve class distributions. A cross-split leakage check is run before augmentation using text hashing.

---

## 9. Methodology

### 9.1 Data Preprocessing

The preprocessing pipeline (`src/data_processing/data_pipeline.py`) runs **8 stages** in strict order:

**Stage 1 — Path Resolution** (`data_resolver.py`)
Resolves raw dataset paths from `config/data_config.yaml`. Fails early with a descriptive error if configured paths do not exist, preventing silent downstream failures.

**Stage 2 — Load / Validate / Clean**
- `data_loader.py`: reads CSV files and checks for required columns against `data_contracts.py` schemas before any processing
- `data_validator.py`: enforces quality thresholds — null ratio ≤10%, duplicate ratio ≤15%, minimum word count ≥30 words per row, class balance ≥10% per label
- `data_cleaning.py`: applies a configurable cleaning stack — unicode normalization, URL and HTML removal, contraction expansion ("don't" → "do not"), lowercasing, whitespace normalization

**Stage 3 — Multi-task Validation + Label Analysis**
`multitask_validator.py` checks that all required label columns are present across the merged dataset, then `label_analysis.py` computes per-task class distributions and asserts label health (catches silent NaN or missing label issues that would otherwise corrupt the loss computation).

**Stage 4 — Leakage Check** (`leakage_checker.py`)
Compares raw splits by text hash before augmentation. Surfaces exact contamination counts between train/validation and train/test. Fails loudly rather than silently producing inflated evaluation metrics.

**Stage 5 — Data Augmentation** (train split only)
`data_augmentation.py` applies three techniques:
- *Synonym replacement* — substitutes non-stopword tokens with WordNet synonyms
- *Random swap* — transposes random adjacent word pairs
- *Random deletion* — drops tokens with a low probability

Augmentation multiplier is configurable (`augmentation.multiplier: 2`). Back-translation is supported but disabled by default.

**Stage 6 — Cache Write** (`data_cache.py`)
Serializes the processed splits with a cache key derived from tokenizer identity, max token length, cleaning configuration, and augmentation settings. Any change to these parameters automatically invalidates the cache.

**Stage 7 — Data Profiling** (`data_profiler.py`)
Computes distribution statistics (mean/std/min/max/percentiles) for text length, label frequency, and emotion intensity across all three splits. Written to `reports/data_cleaning_report.json`.

**Stage 8 — Dataset and DataLoader Build** (`dataset_factory.py`, `dataloader_factory.py`)
Produces a `MultiTaskDataset` per task that returns tokenized `input_ids`, `attention_mask`, and per-task label tensors. `DataLoader` is constructed with `pin_memory=true`, `num_workers=8`, and per-epoch shuffle on the training split.

---

### 9.2 Model Architecture

#### 9.2.1 Shared Encoder

The encoder is RoBERTa-base loaded from HuggingFace (`roberta-base`), providing:

| Property              | Value            |
|-----------------------|------------------|
| Architecture          | Transformer encoder |
| Layers                | 12               |
| Attention heads       | 12               |
| Hidden dimension      | 768              |
| Intermediate dim      | 3,072            |
| Parameters            | ~125M            |
| Max token length      | 512              |
| Tokenizer             | BPE, vocab 50,265|

**Architectural enhancements applied:**
- **Gradient checkpointing** (`gradient_checkpointing: true`) — recomputes activations during the backward pass instead of storing them, reducing VRAM by ~40% at the cost of ~30% extra compute. Essential for training on hardware with limited GPU memory.
- **Flash Attention** (`flash_attention: true`) — replaces the standard O(n²) attention kernel with a memory-efficient fused CUDA kernel where supported.
- **`torch.compile`** (`torch_compile: true`, `compile_mode: "default"`) — ahead-of-time compilation of the model graph for faster training throughput.

#### 9.2.2 Task-Specific Heads

Each head is a linear classification layer over the `[CLS]` token representation from the shared encoder. All heads are stored in a `nn.ModuleDict` keyed by task name, allowing independent per-task loss scaling and enabling selective task training via config.

```
CLS token representation  (dim: 768)
        │
    Dropout(p=0.1)
        │
    Linear(768 → num_classes)
        │
    Softmax (single-label) or Sigmoid (multi-label)
```

| Head                  | Output classes | Loss function            | Output activation |
|-----------------------|----------------|--------------------------|-------------------|
| `bias_detection`      | 3              | CrossEntropyLoss         | Softmax           |
| `ideology_detection`  | 3              | CrossEntropyLoss         | Softmax           |
| `propaganda`          | 2              | CrossEntropyLoss         | Softmax           |
| `emotion`             | 11             | BCEWithLogitsLoss        | Sigmoid           |
| `narrative_roles`     | 3              | BCEWithLogitsLoss        | Sigmoid           |
| `narrative_frame`     | 5              | BCEWithLogitsLoss        | Sigmoid           |

#### 9.2.3 HybridTruthLensModel (Variant)

In addition to `MultiTaskTruthLensModel`, the codebase includes `HybridTruthLensModel` (`src/models/architectures/hybrid_truthlens_model.py`). This variant augments the encoder CLS representation with a projection of hand-engineered features before the classification heads:

```
CLS token (768) + Engineered feature vector
                │
          Linear (feature_proj)  →  Fused representation
                │
          Fusion layer
                │
          Task heads
```

Xavier initialization (`CFG1`) is applied to the projection weights. This model is used when auxiliary TF-IDF or lexicon features are expected to carry signal not captured by the transformer.

---

### 9.3 Training Process

#### Hyperparameters

| Parameter                         | Value                          | Rationale                                              |
|-----------------------------------|--------------------------------|--------------------------------------------------------|
| Base encoder                      | roberta-base                   | Balance of capacity vs CPU inference feasibility       |
| Max sequence length               | 512 tokens                     | Full article context                                   |
| Batch size (`data.batch_size`)    | 32                             | Stable gradient signal on single GPU                  |
| Gradient accumulation steps       | 2                              | Effective batch = 64; reduces per-step variance       |
| Max epochs                        | 10                             | Upper bound; early stopping typically fires by epoch 5–6 |
| Min epochs                        | 4                              | Prevents premature stopping in early convergence phases|
| Gradient clipping (`max_grad_norm`)| 1.0                           | Hard L2 norm cap; prevents gradient explosion          |
| AMP dtype                         | bf16                           | BF16 preferred over FP16 — larger dynamic range, avoids overflow |
| Optimizer                         | AdamW                          | Weight decay regularization without biasing embeddings |
| Learning rate (post-convergence)  | 1.75e-6                        | Tuned post epoch-4 stagnation analysis                |
| LR schedule                       | Linear warmup                  | Stable early training before peak LR                  |
| Early stopping patience           | 2 epochs                       | Tight; relies on `min_epochs` for safety margin        |
| Early stopping min_delta          | 0.003                          | Noise floor above ±0.001 per-epoch validation oscillation |
| Early stopping metric             | `weighted_composite_score`     | Task-balanced; prevents dominant heads masking underfit |
| Checkpoint retention              | 3                              | Saves `checkpoint.pt` + `checkpoint.meta.json`        |

#### Early Stopping Design

A critical design decision was replacing `eval_loss` with `weighted_composite_score` as the early stopping monitor. Profiling of early training runs showed that `eval_loss` was dominated by the easier heads (propaganda, emotion) which saturated quickly, while the ideology head — substantially harder — continued to improve beyond epoch 4 before regressing. The `weighted_composite_score` is a per-task accuracy-weighted aggregate injected into `val_metrics` by the Trainer after each evaluation step, ensuring that all six heads contribute equally to the stop decision.

#### Multi-Task Loss

The total loss at each step is the unweighted sum of per-task losses:

```
L_total = L_bias + L_ideology + L_propaganda + L_emotion + L_narrative_roles + L_narrative_frame
```

All tasks are treated equally in the loss sum. Task-specific loss weighting is available via config for future experimentation but not used in the default configuration.

#### BF16 Automatic Mixed Precision

Training uses PyTorch's `torch.amp` with `dtype=torch.bfloat16`. BF16 was chosen over FP16 because:
- BF16 has the same exponent range as FP32 (8 bits) — immune to the overflow spikes that corrupted earlier FP16 training runs at this learning rate
- The dynamic loss scaler required by FP16 (`GradScaler`) introduces complexity; BF16 runs without a scaler
- BF16 is natively supported on modern NVIDIA (Ampere+) and AMD GPUs

On CPU-only environments, the `amp.autocast` context is a no-op and training proceeds in float32.

---

### 9.4 Pipeline Flow

#### Training Pipeline (Offline)

```
python main.py
    │
    ├── [1] load_settings() + set_seed(42)
    ├── [2] run_data_pipeline() — 8-stage data pipeline
    │       └── Returns: train/val/test DataLoaders
    ├── [3] AutoTokenizer.from_pretrained("roberta-base")
    ├── [4] MultiTaskTruthLensModel(config)
    ├── [5] create_multitask_trainer_fn() → Trainer
    │       ├── AdamW optimizer
    │       ├── Linear LR scheduler
    │       ├── BF16 AMP context
    │       ├── Gradient accumulation (steps=2)
    │       └── Early stopping (patience=2, metric=weighted_composite_score)
    ├── [6] trainer.fit(train_loader, val_loader)
    │       └── Per-epoch: forward → loss → backward → clip → step → eval
    ├── [7] Save checkpoint.pt + checkpoint.meta.json
    └── [8] Evaluate on test split → reports/evaluation_results.json
```

#### Inference Pipeline (Online, per API request)

```
POST /predict or /analyze
    │
    ├── Pydantic validation (text: 10–10,000 chars)
    ├── ModelRegistry.load_model()  [cached after first call]
    │       └── loads checkpoint.pt → model.eval()
    ├── tokenizer(text, max_length=512, truncation=True)
    ├── model(input_ids, attention_mask)
    │       └── Returns: {bias, ideology, propaganda, emotion, roles, frames}
    ├── /predict: assemble NewsResponse → return
    └── /analyze only:
            ├── BiasLexiconFeatures.extract(context)
            ├── EmotionLexiconAnalyzer.analyze(text)
            └── LIMEExplainer(model.predict_batch, num_samples=256)
                    └── Returns top-8 token importances
    └── Assemble AnalysisResponse → return
```

#### Linguistic Analysis Pipeline (Post-Inference, full analysis)

```
FeatureContext(text)  [spaCy doc computed once, shared]
    │
    └── AnalysisPipeline.run(context)
            │
            ├── [1]  RhetoricalDeviceDetector  → hyperbole, loaded language scores
            ├── [2]  ArgumentMiningAnalyzer    → claim-evidence structure
            ├── [3]  ContextOmissionDetector   → missing context signals
            ├── [4]  DiscourseCoherenceAnalyzer→ argument consistency
            ├── [5]  EmotionTargetAnalyzer     → emotion target entities
            ├── [6]  FramingAnalyzer           → positive/negative framing
            ├── [7]  InformationDensityAnalyzer→ lexical density
            ├── [8]  InformationOmissionDetector → omitted context
            ├── [9]  IdeologicalLanguageDetector → political markers
            ├── [10] NarrativeRoleExtractor    → hero/villain/victim signals
            ├── [11] NarrativeConflictAnalyzer → adversarial framing
            ├── [12] NarrativePropagationAnalyzer → propagation patterns
            ├── [13] NarrativeTemporalAnalyzer → timeline consistency
            └── [14] SourceAttributionAnalyzer → source credibility
                    │
                    ▼
            Aggregation Engine
            ├── FeatureMapper (raw → named groups)
            ├── WeightManager (adaptive, confidence+entropy-driven)
            │       WEIGHT_GROUPS (aggregation_config.py):
            │       "manipulation": bias, emotion, narrative, analysis_influence_manipulation
            │       "credibility":  discourse, graph, analysis_influence_credibility
            │       "final":        final_credibility, final_manipulation, final_ideology
            ├── TruthLensScoreCalculator → credibility_score ∈ [0.0, 1.0]
            ├── RiskAssessment → Low (0–0.33) | Medium (0.34–0.66) | High (0.67–1.0)
            └── ScoreExplainer → human-readable breakdown
```

---

## 10. Implementation

### 10.1 Repository Structure

```
TruthLens-AI/
│
├── api/                          # FastAPI REST service
│   └── app.py                    # All endpoints: /, /health, /predict, /analyze, /project-view
│
├── config/
│   ├── config.yaml               # Model, training, API, inference settings
│   └── data_config.yaml          # Data pipeline, cleaning, augmentation settings
│
├── data/
│   ├── raw/{bias,emotion,ideology,narrative,propaganda}/
│   ├── processed/unified_dataset.csv
│   └── splits/{train,validation,test}.csv
│
├── documentation/                # 11 technical reference documents
│
├── models/
│   ├── inference/predictor.py    # predict() and predict_batch()
│   ├── registry/model_registry.py# ModelRegistry (load + cache)
│   ├── checkpointing/            # checkpoint.pt lifecycle management
│   └── tfidf_vectorizer.joblib   # Fitted TF-IDF artifact
│
├── src/
│   ├── aggregation/              # 5-stage credibility scoring engine
│   │   ├── aggregation_config.py # WEIGHT_GROUPS (single source of truth)
│   │   ├── feature_mapper.py
│   │   ├── weight_manager.py
│   │   ├── truthlens_score_calculator.py
│   │   ├── risk_assessment.py
│   │   └── score_explainer.py
│   │
│   ├── analysis/                 # 14 linguistic analyzers
│   │   ├── analysis_registry.py  # build_default_registry() — 14 analyzers
│   │   ├── analysis_pipeline.py  # AnalysisPipeline
│   │   ├── feature_context.py    # FeatureContext (spaCy-backed)
│   │   └── [14 analyzer modules]
│   │
│   ├── data_processing/          # 8-stage data pipeline
│   │   ├── data_pipeline.py      # Orchestrator
│   │   ├── data_contracts.py     # Task schemas (canonical)
│   │   ├── data_loader.py
│   │   ├── data_validator.py
│   │   ├── data_cleaning.py
│   │   ├── data_augmentation.py
│   │   ├── leakage_checker.py
│   │   ├── data_cache.py
│   │   ├── data_profiler.py
│   │   ├── dataset_factory.py
│   │   └── dataloader_factory.py
│   │
│   ├── features/                 # Feature extractors
│   │   ├── bias/                 # Lexicon density, framing, ideological markers
│   │   ├── emotion/              # EMOTION-11 lexicon; emotion_schema.py
│   │   ├── narrative/            # Frame detection (CO/EC/HI/MO/RE), role features
│   │   ├── propaganda/           # Loaded language, fear appeals
│   │   ├── discourse/            # Claim-evidence structure
│   │   ├── graph/                # Entity co-occurrence (NetworkX)
│   │   ├── text/                 # Lexical, semantic, syntactic features
│   │   ├── fusion/               # Feature scaling and selection
│   │   ├── pipelines/            # End-to-end extraction orchestration
│   │   ├── cache/                # Feature caching layer
│   │   └── importance/           # Permutation importance, SHAP, ablation
│   │
│   ├── models/
│   │   ├── multitask/            # MultiTaskTruthLensModel
│   │   ├── architectures/        # HybridTruthLensModel (encoder + feature fusion)
│   │   ├── encoder/              # Shared RoBERTa encoder wrapper
│   │   ├── heads/                # Per-task head implementations
│   │   ├── checkpointing/        # Checkpoint save/load/resolve
│   │   └── registry/             # ModelRegistry + ModelFactory
│   │
│   ├── explainability/           # SHAP, LIME, attention rollout, explainer caches
│   ├── evaluation/               # Accuracy, F1, ROC-AUC, weighted_composite_score
│   ├── training/                 # Trainer, optimizer factory, scheduler factory
│   ├── inference/                # InferenceEngine, batch inference
│   ├── graph/                    # Entity/narrative graph builders (NetworkX)
│   ├── pipelines/                # TruthLensPipeline end-to-end orchestrator
│   └── utils/                    # Settings, config loader, seed, device, logging
│
├── tests/                        # 236+ tests across all subsystems
├── main.py                       # Training entry point (argparse: train/infer/both)
├── run_eda.py                    # EDA report generator
└── requirements.txt
```

### 10.2 Key Design Decisions

**`data_contracts.py` as single source of truth:**
All task schema definitions — column names, types, number of classes — live exclusively in `data_contracts.py`. The data validator, dataset factory, training pipeline, and configuration all import from here. Any change to a task schema requires a single file edit.

**`aggregation_config.py` as single source of truth for weight groups:**
Both `weight_manager.py` and `truthlens_score_calculator.py` previously maintained private copies of the signal-to-group mapping, causing silent drift. Consolidating `WEIGHT_GROUPS` and `TASK_TO_GROUP` into `aggregation_config.py` and having both modules import from it eliminates this class of bug.

**`FeatureContext` as the shared spaCy boundary:**
All 14 analysis module receive a single `FeatureContext` object. spaCy processing (tokenization, NER, POS, dependency parsing) is done once per article and the spaCy `Doc` is cached inside the context. This prevents 14× redundant spaCy calls per analysis request.

**`weighted_composite_score` early stopping metric:**
Switching from `eval_loss` to a per-task-balanced composite score ensures the training loop remains sensitive to under-performing heads rather than stopping as soon as the dominant (easy) heads plateau. This was the single most impactful training stability fix observed across training runs.

---

## 11. Results and Evaluation

### 11.1 Evaluation Metrics

Per-task metrics are computed after each epoch on the validation split, and over the full test split after training completes.

| Task                  | Metric Type              | Metrics Computed                        |
|-----------------------|--------------------------|-----------------------------------------|
| Bias detection        | Single-label (3-class)   | Accuracy, Precision, Recall, F1         |
| Ideology detection    | Single-label (3-class)   | Accuracy, Precision, Recall, F1         |
| Propaganda detection  | Binary                   | Accuracy, Precision, Recall, F1         |
| Emotion classification| Multi-label (11)         | Micro-F1, Macro-F1, ROC-AUC             |
| Narrative roles       | Multi-label binary (3)   | Micro-F1, Macro-F1, ROC-AUC             |
| Narrative frames      | Multi-label binary (5)   | Micro-F1, Macro-F1, ROC-AUC             |
| Combined              | Task-balanced aggregate  | `weighted_composite_score`              |

Results are written to `reports/evaluation_results.json` after training. Confusion matrices are saved to `reports/confusion_matrix.png`.

### 11.2 Inference Performance

| Endpoint   | Hardware    | Typical Latency           | Notes                                          |
|------------|-------------|---------------------------|------------------------------------------------|
| `/predict` | GPU (CUDA)  | 100–500 ms                | Single forward pass, no LIME                   |
| `/predict` | CPU         | 2–10 s                    | Warm model, standard article length            |
| `/analyze` | GPU (CUDA)  | 2–15 s                    | LIME: 256 perturbation passes                  |
| `/analyze` | CPU         | 10–60 s                   | LIME dominates; first request includes model load |
| First req  | Any         | +5–30 s                   | One-time model loading; cached thereafter      |

### 11.3 Credibility Score Components

The aggregation engine's `WEIGHT_GROUPS` structure controls signal contribution:

| Group           | Signals                                                     | Direction             |
|-----------------|-------------------------------------------------------------|-----------------------|
| `manipulation`  | bias, emotion intensity, narrative signals, analysis_influence_manipulation | ↑ lowers credibility |
| `credibility`   | discourse coherence, graph signals, analysis_influence_credibility | ↑ raises credibility |
| `final`         | final_credibility, final_manipulation, final_ideology       | Combined final score  |

Weights within groups are computed adaptively per-article based on prediction confidence and entropy — high-confidence low-entropy predictions receive higher weight for that signal.

---

## 12. Explainability

Explainability is a first-class citizen in TruthLens AI, not an afterthought. Every prediction at the `/analyze` endpoint includes three layers of explanation.

### 12.1 LIME (Local Interpretable Model-Agnostic Explanations)

LIME generates a local linear approximation of the model's decision boundary around the input article. Implementation:

1. The input text is perturbed 256 times by randomly masking tokens
2. Each perturbation is passed through `predict_batch()` to get fake probability
3. A linear model is fit on the 256 perturbation–label pairs
4. The 8 tokens with the highest absolute coefficients are returned as `important_features`

```json
"lime": {
  "important_features": [
    { "feature": "hiding",    "weight":  0.1234 },
    { "feature": "miracle",   "weight":  0.0987 },
    { "feature": "confirmed", "weight": -0.0654 }
  ]
}
```

**Positive weight** → pushes prediction toward FAKE. **Negative weight** → pushes toward REAL.

LIME errors are caught and returned as `{ "error": "lime_unavailable" }` without failing the endpoint.

### 12.2 SHAP (SHapley Additive exPlanations)

SHAP (`src/explainability/shap_explainer.py`) computes Shapley values for the engineered feature vector, providing a globally consistent feature importance ranking. Unlike LIME (which varies per perturbation sample), SHAP values satisfy the theoretical properties of efficiency, symmetry, and dummy — making cross-article comparisons valid.

### 12.3 Attention Rollout

`src/explainability/attention_rollout.py` implements the attention rollout algorithm: it recursively multiplies attention weight matrices layer-by-layer to produce an attribution map from the `[CLS]` token to every input token, accounting for skip connections. This provides a fully differentiable, model-internal explanation as an alternative to the perturbation-based LIME approach.

### 12.4 Emotion and Bias Explainers

- `EmotionExplainer` (`src/explainability/emotion_explainer.py`) traces which lexicon tokens matched which EMOTION-11 labels and their contribution intensities
- `BiasExplainer` (`src/explainability/bias_explainer.py`) renders a per-sentence heatmap of bias density, identifying which specific sentences carry the highest partisan framing load

---

## 13. Frontend and Backend Overview

### 13.1 React Frontend

The frontend is a React 19 single-page application built with Vite 8 and styled with Tailwind CSS v4.

**Pages:**

| Route        | Page            | Description                                              |
|--------------|-----------------|----------------------------------------------------------|
| `/`          | HomePage        | Hero section, live status indicator, quick analysis tool |
| `/analysis`  | AnalysisPage    | Textarea input, example prompts, inline verdict card     |
| `/results`   | ResultsPage     | Full six-panel analysis dashboard                        |
| `/features`  | FeaturesPage    | Detection capability descriptions                        |
| `/about`     | AboutPage       | Problem statement, solution overview, use cases          |

**Analysis flow:**
1. User pastes article text or clicks one of three example prompt buttons (Fake News / Real News / Biased Report)
2. `useAnalysis.js` hook fires `POST /analyze` via Axios
3. `SkeletonLoader` animates during inference
4. `VerdictCard` renders the top-level prediction badge, confidence bar, and key signal summary inline
5. "View Full Dashboard" navigates to `/results` where six panel components render the complete breakdown

**Dashboard panels:**

| Panel                  | Visualisation                                | Data source                        |
|------------------------|----------------------------------------------|------------------------------------|
| `BiasPanel`            | Bias score bar + sentence heatmap (Recharts) | `bias_score`, `sentence_heatmap`   |
| `EmotionPanel`         | Radar or bar chart of 11 emotion scores      | `emotion_scores`                   |
| `NarrativePanel`       | Role proportion bars, frame badges           | `narrative_roles`, `narrative_frame`|
| `RhetoricPanel`        | Device detection badges                      | Analysis layer signals             |
| `PropagandaPanel`      | Intensity gauge + technique breakdown        | Propaganda head + analyzers        |
| `ExplainabilityPanel`  | Horizontal bar chart of LIME weights         | `lime.important_features`          |

### 13.2 Node.js / Express Middleware

A lightweight Express server sits between the React frontend and the FastAPI Python backend:

```
React (port 5000)  →  Express /analyze proxy  →  FastAPI (port 5000)
```

The middleware layer:
- Proxies `/analyze` and `/explain` to the Python API
- Validates request format before forwarding
- Returns a structured fallback response if the ML backend is unavailable

### 13.3 API Endpoints (FastAPI)

| Method | Endpoint        | Description                                           |
|--------|-----------------|-------------------------------------------------------|
| GET    | `/`             | API online check, lists endpoints                     |
| GET    | `/health`       | Model file status (healthy / degraded / unhealthy)    |
| POST   | `/predict`      | Binary fake/real + confidence (fast path)             |
| POST   | `/analyze`      | Full six-task analysis + LIME explainability          |
| GET    | `/project-view` | Config metadata and directory structure               |
| GET    | `/docs`         | Swagger interactive documentation                     |

All endpoints use Pydantic request validation. Text must be 10–10,000 characters. Errors return `{"detail": "..."}` with appropriate HTTP status codes (400 / 422 / 500 / 503).

---

## 14. Challenges Faced

**Multi-task early stopping instability**
The first version of the training loop used `eval_loss` as the early stopping metric. Combined losses from six heads caused the combined loss to decrease even when the harder heads (ideology, narrative frames) were still improving. Solution: implemented `weighted_composite_score` as a task-balanced aggregate metric, injected into `val_metrics` by the Trainer post-evaluation.

**Post-convergence gradient explosion (ideology head)**
Training runs past epoch 4 showed gradient norms exploding (174 → 453) specifically for the ideology head. Analysis revealed the learning rate was too high for the post-saturation phase of other heads. Solution: reduced learning rate to 1.75e-6 post-convergence and added `spike_warn_threshold` in `training_step.py` to surface norm anomalies during training rather than after.

**Emotion schema reduction (20 → 11 labels)**
The GoEmotions 20-label schema produced severe class imbalance and overlapping semantic categories. Labels with poor inter-annotator agreement and low dataset frequency were retired to `_LEGACY_EMOTION_LABELS`. The positional `emotion_0…emotion_10` column naming convention was introduced to prevent silent schema drift — string-named columns (`emotion_joy`) are now rejected at the data contract validation stage.

**Leakage in cross-dataset merging**
Merging five independently collected datasets produced overlapping text samples between train and test splits. Naively evaluated models showed inflated test accuracy. The `leakage_checker.py` module was implemented to detect and report cross-split contamination by text hash before augmentation.

**AGGREGATION_CONFIG single-source-of-truth divergence**
`weight_manager.py` and `truthlens_score_calculator.py` each maintained private copies of the signal-to-group mapping that diverged silently across refactors. Per-group renormalization broke without any error. Solution: centralized `WEIGHT_GROUPS` and `TASK_TO_GROUP` into `aggregation_config.py` and made both modules import from it.

**BF16 overflow under FP16**
Early training runs with `amp_dtype: fp16` produced NaN losses on H100 hardware due to loss scaler instability. Switching to BF16 (larger exponent range, no scaler needed) resolved the issue without accuracy degradation.

**Uvicorn reload loop in Replit**
The `--reload` flag with no directory scope watched the entire project tree including `.pythonlibs/`, triggering server restarts on every package install. Solution: scope `--reload-dir` to `api`, `src`, `config`, and `models` only.

**FeatureContext spaCy redundancy**
Early versions of the analysis pipeline called `spacy.load()` and ran the full NLP pipeline (NER, POS, dep) independently in each of the 14 analyzers per article — 14× the necessary compute. Solution: `FeatureContext` computes the spaCy `Doc` once on construction and caches it; all analyzers share the same `Doc` object via the context.

---

## 15. Future Scope

**Model improvements:**
- Fine-tune on a larger unified multi-task dataset with synthetic augmentation via back-translation
- Experiment with `roberta-large` (24 layers, ~355M params) for higher task accuracy when GPU inference is available
- Add a claim-level veracity head using retrieval-augmented generation (RAG) against a fact-checking knowledge base
- Explore cross-lingual models (XLM-RoBERTa) for non-English article support

**Pipeline improvements:**
- Real-time streaming analysis for browser extension integration
- Source credibility scoring — cross-reference outlet name against known publication track records
- Temporal misinformation trending — detect coordinated inauthentic amplification patterns over a time window
- Confidence calibration with temperature scaling per task head

**Platform improvements:**
- User accounts with saved analysis history and personal credibility dashboards
- Public API with rate limiting, API key management, and usage analytics
- Mobile application (React Native) for on-the-go fact checking
- Browser extension that analyses the currently open article in a sidebar panel
- Batch CSV upload for research-scale dataset analysis

---

## 16. Conclusion

TruthLens AI demonstrates that effective misinformation detection requires a multi-layer approach that combines deep transformer-based multi-task classification with interpretable rule-based linguistic analysis. The system's core contribution is not any single component but the integration of six simultaneous detection dimensions — bias, ideology, propaganda, emotion, narrative roles, and narrative framing — into a unified, explainable credibility verdict.

The most significant engineering contributions are:

1. The **8-stage data pipeline** with strict schema contracts (`data_contracts.py`), leakage detection before augmentation, and config-keyed cache invalidation — eliminating entire classes of silent data quality bugs
2. The **`weighted_composite_score` early stopping metric** — a task-balanced aggregate that prevents dominant heads from masking under-convergence in harder tasks
3. The **5-stage aggregation engine** with a single-source-of-truth weight group definition (`aggregation_config.py`) — ensuring signal grouping is consistent across all scoring components
4. The **14-analyzer linguistic pipeline** backed by a shared `FeatureContext` — providing deep structural analysis at the cost of a single spaCy pass per article
5. **LIME + SHAP + attention rollout** at the inference layer — making every prediction traceable to specific tokens and features, not just returning a verdict

The system is production-deployed on Replit Autoscale, CPU-compatible at inference time, and returns results in under 10 seconds on `/predict`. It serves as a foundation for future work in multi-source credibility verification, cross-lingual detection, and real-time misinformation monitoring.

---

## 17. References

### Research Papers

- Liu, Y. et al. (2019). *RoBERTa: A Robustly Optimized BERT Pretraining Approach*. arXiv:1907.11692
- Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). *"Why Should I Trust You?": Explaining the Predictions of Any Classifier*. KDD 2016
- Lundberg, S. M., & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions*. NeuroPIPS 2017
- Demszky, D. et al. (2020). *GoEmotions: A Dataset of Fine-Grained Emotions*. ACL 2020
- Da San Martino, G. et al. (2019). *Fine-Grained Analysis of Propaganda in News Articles*. EMNLP 2019

### Datasets

- BABE (Bias Annotations By Experts), BASIL, MBIC — media bias benchmarks
- PTC Propaganda Corpus — propaganda technique classification
- AllSides Media Bias Ratings — ideology detection
- GoEmotions (Google) — fine-grained emotion classification

### Libraries and Frameworks

- HuggingFace Transformers — `transformers` library, RoBERTa weights
- PyTorch — deep learning framework, AMP, `torch.compile`
- FastAPI — modern Python REST framework with Pydantic validation
- spaCy — industrial NLP: NER, POS, dependency parsing
- LIME (`lime`) — local interpretable explanations
- SHAP (`shap`) — Shapley value attribution
- NetworkX — graph construction and topological analysis
- Optuna — hyperparameter optimization framework
- React 19, Vite 8, Tailwind CSS v4 — frontend framework and tooling
- Recharts — data visualisation for React
- Framer Motion — animation library for React
