# Project Structure

This document explains the **directory structure and architectural organization** of the TruthLens AI repository.

TruthLens AI is organized as a **modular machine learning platform** designed for:

- Misinformation detection and fake news classification
- Credibility analysis and bias profiling
- Linguistic signal extraction (emotion, propaganda, narrative)
- Explainable AI (SHAP, LIME, attention rollout)
- Scalable inference via REST API

The repository follows a **layered architecture** separating data processing, feature extraction, modeling, inference, and evaluation.

---

## Root Directory

```
TruthLens-AI/
├── api/                     # FastAPI REST service
├── config/                  # YAML configuration files
├── data/                    # Raw, processed, and split datasets
├── documentation/           # Architecture and system documentation
├── logs/                    # Training and inference logs
├── models/                  # Trained model artifacts and inference helpers
├── reports/                 # Evaluation reports and EDA outputs
├── src/                     # Core application source code
├── tests/                   # Unit and integration tests (236+ tests)
├── main.py                  # Training entry point
├── run_eda.py               # EDA report generator
├── requirements.txt         # Python dependencies
└── replit.md                # Replit-specific project notes
```

---

## API Layer — `api/`

```
api/
├── __init__.py
└── app.py                   # FastAPI application entry point
```

Exposes a **FastAPI-based REST service** for article analysis and model inference.

**Endpoints:**

| Method | Path             | Description                                   |
|--------|------------------|-----------------------------------------------|
| GET    | `/`              | Health check, lists all available endpoints   |
| GET    | `/health`        | Detailed health check (model file status)     |
| POST   | `/predict`       | Binary fake/real classification               |
| POST   | `/analyze`       | Full analysis: bias, emotion, explainability  |
| GET    | `/project-view`  | API metadata and directory structure          |
| GET    | `/docs`          | Interactive Swagger API documentation         |

---

## Configuration — `config/`

```
config/
├── config.yaml              # Model, training, API, and inference settings
└── data_config.yaml         # Dataset pipeline and preprocessing settings
```

Stores all system configuration parameters. See [CONFIGURATION.md](CONFIGURATION.md) for details.

---

## Data Layer — `data/`

```
data/
├── raw/                     # Original source datasets
│   ├── bias/
│   ├── emotion/             # Emotion CSVs use columns emotion_0…emotion_10
│   ├── ideology/
│   ├── narrative/
│   └── propaganda/
├── interim/                 # Intermediate processing outputs
├── processed/               # Cleaned and merged datasets
│   └── unified_dataset.csv
└── splits/                  # Train / validation / test CSVs
    ├── train.csv
    ├── validation.csv
    └── test.csv
```

Datasets cover: fake news, bias, emotion (EMOTION-11 schema), narrative framing, propaganda, and ideology. All are unified using a **shared label schema** defined in `src/data_processing/data_contracts.py`.

---

## Documentation — `documentation/`

```
documentation/
├── API_REFERENCE.md         # Complete REST API reference
├── ARCHITECTURE.md          # System architecture overview
├── CONFIGURATION.md         # Configuration file reference
├── CONTRIBUTING.md          # Contributor guidelines
├── DEPLOYMENT.md            # Deployment instructions
├── FEATURE_ENGINEERING.md   # Feature engineering system
├── MODEL_CARD.md            # Model details, datasets, limitations
├── PROJECT_STRUCTURE.md     # This file
├── SYSTEM_DESIGN.md         # End-to-end system design
├── TRAINING_GUIDE.md        # Model training walkthrough
└── TROUBLESHOOTING.md       # Common issues and fixes
```

---

## Logs — `logs/`

```
logs/
├── training.log             # Training run logs
└── inference.log            # Inference and API logs
```

---

## Models — `models/`

```
models/
├── inference/
│   ├── __init__.py
│   └── predictor.py         # predict() and predict_batch() functions
├── registry/
│   └── model_registry.py    # ModelRegistry — loads and caches model assets
├── checkpointing/
│   ├── artifact_manager.py  # Manages checkpoint artifact lifecycle
│   ├── checkpoint_manager.py# Saves/loads checkpoint.pt
│   ├── model_loader.py      # Loads model from checkpoint
│   └── resolver.py          # Resolves checkpoint paths
├── checkpoints/             # Checkpoint directory (created after training)
│   ├── checkpoint.pt        # Trained model weights (PyTorch state dict)
│   └── checkpoint.meta.json # Training metadata
├── cache/                   # HuggingFace model download cache
└── tfidf_vectorizer.joblib  # TF-IDF vectorizer artifact
```

The `models/inference/predictor.py` module provides:
- `predict(text)` — single article inference, returns label + fake probability + confidence
- `predict_batch(texts)` — batch inference for LIME explanations

---

## Reports — `reports/`

```
reports/
├── evaluation_results.json
├── confusion_matrix.png
├── data_cleaning_report.json
└── figures/                 # EDA plots and charts
```

Generated by training runs and `python run_eda.py`.

---

## Source Code — `src/`

The `src/` directory contains the **core implementation of TruthLens AI**, organized into subsystems:

### Aggregation — `src/aggregation/`

Computes the **final TruthLens Credibility Score** through a five-stage pipeline:
`FeatureMapper → WeightManager → TruthLensScoreCalculator → RiskAssessment → ScoreExplainer`

```
src/aggregation/
├── aggregation_config.py        # WEIGHT_GROUPS (single source of truth for signal grouping)
├── aggregation_pipeline.py      # Orchestrates the five-stage scoring pipeline
├── aggregation_metrics.py       # Aggregation-level metrics
├── aggregation_validator.py     # Input signal validation
├── calibration.py               # Score calibration utilities
├── feature_mapper.py            # Maps raw model/analysis outputs to named groups
├── risk_assessment.py           # Low / Medium / High risk classification
├── score_explainer.py           # Human-readable score explanations
├── score_normalizer.py          # Signal normalization
├── score_schema.py              # Typed output schemas
├── truthlens_score_calculator.py# Main scoring engine
└── weight_manager.py            # Adaptive signal weighting (imports WEIGHT_GROUPS)
```

`WEIGHT_GROUPS` in `aggregation_config.py` is the single source of truth for group membership:
- `"manipulation"`: bias, emotion, narrative, analysis_influence_manipulation
- `"credibility"`: discourse, graph, analysis_influence_credibility
- `"final"`: final_credibility, final_manipulation, final_ideology

### Analysis — `src/analysis/`

Performs deep linguistic analysis via **14 analyzers** registered in `AnalyzerRegistry` through `build_default_registry()`. `AnalysisPipeline` runs them sequentially against a shared `FeatureContext`.

```
src/analysis/
├── analysis_registry.py          # AnalyzerRegistry + build_default_registry() (14 analyzers)
├── analysis_pipeline.py          # AnalysisPipeline orchestrator
├── analysis_config.py            # Analysis configuration
├── base_analyzer.py              # BaseAnalyzer interface
├── feature_context.py            # FeatureContext (spaCy-backed, shared across all analyzers)
├── feature_keys.py               # Canonical feature key constants
├── feature_merger.py             # Merges analyzer outputs
├── feature_schema.py             # Typed analysis output schemas
├── output_models.py              # AnalysisResult and related models
├── argument_mining.py            # ArgumentMiningAnalyzer (order 2)
├── bias_profile_builder.py       # Bias profiling utilities
├── context_omission_detector.py  # ContextOmissionDetector (order 3)
├── discourse_coherence_analyzer.py # DiscourseCoherenceAnalyzer (order 4)
├── emotion_lexicon.py            # Emotion lexicon utilities
├── emotion_target_analysis.py    # EmotionTargetAnalyzer (order 5)
├── framing_analysis.py           # FramingAnalyzer (order 6)
├── ideological_language_detector.py # IdeologicalLanguageDetector (order 9)
├── information_density_analyzer.py  # InformationDensityAnalyzer (order 7)
├── information_omission_detector.py # InformationOmissionDetector (order 8)
├── label_analysis.py             # Per-task label distribution analysis
├── multitask_validator.py        # Multi-task column validation
├── narrative_conflict.py         # NarrativeConflictAnalyzer (order 11)
├── narrative_propagation.py      # NarrativePropagationAnalyzer (order 12)
├── narrative_role_extractor.py   # NarrativeRoleExtractor (order 10)
├── narrative_temporal_analyzer.py # NarrativeTemporalAnalyzer (order 13)
├── preprocessing.py              # Text preprocessing utilities
├── propaganda_pattern_detector.py # Propaganda detection utilities
├── rhetorical_device_detector.py # RhetoricalDeviceDetector (order 1)
├── source_attribution_analyzer.py # SourceAttributionAnalyzer (order 14)
└── spacy_loader.py               # spaCy model loading and caching
```

### Data Processing — `src/data_processing/`

Handles the **8-stage data pipeline**: path resolution → load/validate/clean → multi-task validation → leakage check → augmentation → cache → profiling → dataset/dataloader build.

```
src/data_processing/
├── data_pipeline.py          # 8-stage pipeline orchestrator (main entry point)
├── data_contracts.py         # Task schemas — single source of truth for column names
├── data_resolver.py          # Resolves raw data paths from config
├── data_loader.py            # CSV reading and initial column checks
├── data_validator.py         # Null ratio, duplicate, class balance validation
├── data_cleaning.py          # Unicode, URL, HTML, contraction, lowercasing
├── data_augmentation.py      # Synonym replacement, random swap, random deletion
├── data_cache.py             # Cache read/write with config-keyed invalidation
├── data_profiler.py          # Distribution statistics for processed splits
├── leakage_checker.py        # Cross-split contamination detection
├── dataset.py                # PyTorch Dataset implementation
├── dataset_factory.py        # Builds per-task datasets from processed splits
├── dataloader_factory.py     # DataLoader construction with configurable workers
├── multitask_loader.py       # Multi-task batch collation
├── collate.py                # Custom collate functions
├── class_balance.py          # Oversampling / undersampling / SMOTE
├── samplers.py               # Weighted and stratified samplers
├── data_cache.py             # Cache management
└── test_loader.py            # Test set loading utilities
```

### Evaluation — `src/evaluation/`

Measures model performance with comprehensive metrics.

```
src/evaluation/
├── metrics.py
├── calibration.py
├── uncertainty_estimator.py
└── evaluation_dashboard.py
```

Metrics: Accuracy, Precision, Recall, F1, ROC-AUC, `weighted_composite_score` (task-balanced early stopping metric)

### Explainability — `src/explainability/`

Provides interpretable explanations for model predictions.

```
src/explainability/
├── shap_explainer.py
├── lime_explainer.py
├── attention_rollout.py
├── attention_visualizer.py
├── emotion_explainer.py
├── bias_explainer.py
├── propaganda_explainer.py
├── explanation_aggregator.py
├── explanation_cache.py
├── explanation_metrics.py
└── explanation_report_generator.py
```

### Feature Engineering — `src/features/`

Generates structured features for the models. See [FEATURE_ENGINEERING.md](FEATURE_ENGINEERING.md) for full details.

```
src/features/
├── base/           # BaseFeature base class
├── bias/           # Bias and ideology lexicon features
├── discourse/      # Argument structure and coherence
├── emotion/        # EMOTION-11 lexicon and trajectory features
│                   # (emotion_schema.py defines EMOTION_LABELS — 11 labels)
├── fusion/         # Feature combination and scaling
├── graph/          # Entity/narrative graph features
├── importance/     # Feature importance analysis tools
├── narrative/      # Frame detection (CO/EC/HI/MO/RE) and role features
├── pipelines/      # End-to-end feature pipeline orchestration
├── propaganda/     # Manipulative rhetoric patterns
├── text/           # Lexical, semantic, syntactic features
└── cache/          # Feature caching system
```

### Graph Analysis — `src/graph/`

Builds entity and narrative graphs for relational reasoning.

```
src/graph/
├── entity_graph.py
├── narrative_graph.py
├── graph_embeddings.py
└── graph_pipeline.py
```

### Models — `src/models/`

Contains model implementations and task-specific heads.

```
src/models/
├── encoder/         # Shared RoBERTa transformer encoder
├── multitask/       # MultiTaskTruthLensModel (shared encoder + 6 task heads)
├── narrative/       # Narrative role classification head
├── propaganda/      # Propaganda detection head
├── ideology/        # Ideology classification head
├── emotion/         # Multi-label emotion classification head (EMOTION-11)
├── checkpointing/   # Checkpoint save/load/resolve utilities
├── ensemble/        # Ensemble methods
├── calibration/     # Model confidence calibration
├── training/        # Training utilities (optimizer, scheduler)
└── registry/        # ModelRegistry — model loading and caching
```

### Inference — `src/inference/`

Production inference pipeline.

```
src/inference/
├── inference_engine.py
├── prediction_pipeline.py
├── batch_inference.py
├── model_loader.py
└── report_generator.py
```

### Pipelines — `src/pipelines/`

End-to-end ML workflow orchestration.

```
src/pipelines/
├── preprocessing_pipeline.py
├── feature_pipeline.py
├── prediction_pipeline.py
└── truthlens_analysis_pipeline.py
```

### Training — `src/training/`

Model training and optimization utilities.

```
src/training/
├── cross_validation.py
├── hyperparameter_tuning.py
├── optimizer_factory.py
└── scheduler_factory.py
```

### Utilities — `src/utils/`

Shared utilities used across the project.

```
src/utils/
├── config_loader.py        # YAML configuration loading and dataclass conversion
├── settings.py             # Centralized settings system (primary config interface)
├── logging_utils.py        # Structured logging setup
├── device_utils.py         # CUDA / MPS / CPU detection and tensor routing
├── input_validation.py     # Text and DataFrame validation
├── json_utils.py           # JSON artifact save/load helpers
├── seed_utils.py           # Reproducibility (random, numpy, torch seeds)
├── time_utils.py           # Benchmarking timer and decorator
└── helper_functions.py     # General-purpose utilities
```

---

## Tests — `tests/`

```
tests/
├── test_data/
├── test_features/
├── test_models/
├── test_inference/
├── test_explainability/
├── test_api/
└── test_utils/
```

236+ tests covering: data processing, feature pipelines, model training, inference, explainability, API endpoints, and configuration validation. Run with:

```bash
pytest
```

---

## End-to-End System Pipeline

```
News Article Input
       ↓
Preprocessing & Text Cleaning
       ↓
Feature Engineering (Lexical · Bias · Emotion (11-label) · Narrative · Propaganda)
       ↓
MultiTask Transformer (RoBERTa + 6 Task Heads)
       ↓
Linguistic Analysis (14 Analyzers via AnalyzerRegistry)
       ↓
Graph Analysis (Entity & Narrative Graphs)
       ↓
Explainability (SHAP · LIME · Attention Rollout)
       ↓
Aggregation Engine (FeatureMapper → WeightManager → TruthLensScoreCalculator)
       ↓
TruthLens Credibility Score + Risk Level + API Response
```
