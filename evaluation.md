# TruthLens AI — Evaluation System Documentation

**Module:** `src/evaluation/`  
**File count:** 23 (20 top-level + 3 in `importance/`)  
**Python:** 3.12 · **Framework:** FastAPI / Uvicorn · **Encoder:** `roberta-base`  
**Tasks:** `emotion`, `narrative`, `propaganda`, `bias`, `ideology`, `narrative_frame`

---

## Table of Contents

1. [Overview](#1-overview)
2. [Module File Index](#2-module-file-index)
3. [Architecture and Design Principles](#3-architecture-and-design-principles)
4. [End-to-End Data Flow](#4-end-to-end-data-flow)
5. [Prediction Collection](#5-prediction-collection)
6. [Metrics Engine](#6-metrics-engine)
7. [Calibration System](#7-calibration-system)
8. [Uncertainty Quantification](#8-uncertainty-quantification)
9. [Error Analysis](#9-error-analysis)
10. [Fairness and Group Metrics](#10-fairness-and-group-metrics)
11. [Feature Importance (Offline)](#11-feature-importance-offline)
12. [Task Correlation](#12-task-correlation)
13. [Threshold Optimization](#13-threshold-optimization)
14. [Report Generation](#14-report-generation)
15. [MLflow Tracking](#15-mlflow-tracking)
16. [Developer Reference](#16-developer-reference)

---

## 1. Overview

The evaluation system is the post-training and production-monitoring backbone of TruthLens AI. It provides a unified, production-grade pipeline for measuring, diagnosing, and reporting model performance across all six multi-task classification heads.

### What it does

- **Collects predictions** from a live model (text → tokenization → logits → probabilities → class predictions) or from a `PredictionService` abstraction that wraps any backend.
- **Computes metrics** for binary, multiclass, and multilabel tasks in a single consistent API (`MetricsEngine`), producing accuracy, balanced accuracy, F1 (macro/micro/weighted/per-class), MCC, ROC-AUC, log-loss, confusion matrix, and multilabel-specific metrics.
- **Calibrates probability outputs** using temperature scaling (scalar and per-label vector), Platt regression, and isotonic regression, with a strict fit-on-validation / apply-on-test split to prevent ECE/Brier leakage.
- **Quantifies uncertainty** via Shannon entropy, confidence margin, energy score, MC-dropout mutual information, and per-task drift signals.
- **Analyzes errors** with type-specific analysis (false positives/negatives for binary, confusion pairs for multiclass, per-label error counts for multilabel) and ranked hard-example surfacing.
- **Audits fairness** across sensitive demographic attributes using demographic parity, equal opportunity, equalized odds, and per-group metrics with bootstrap confidence intervals.
- **Identifies feature importance** (offline only) through SHAP, permutation importance, and ablation importance.
- **Correlates task outputs** with Spearman correlation after K-1 dummy encoding to remove softmax collinearity.
- **Optimizes decision thresholds** by sweeping the precision-recall curve (binary, per-label multilabel, or vectorized global multilabel), with constrained optimization supporting precision-floor / recall-floor constraints.
- **Generates reports** as JSON with companion PNG plots (bar, list, histogram, reliability diagrams) and as PDF documents via ReportLab.
- **Tracks experiments** in MLflow with per-task metrics, flattened parameter dicts, artifact logging, and model/tokenizer/config persistence.
- **Serves an interactive dashboard** via Streamlit for live exploration of any saved JSON report.

### Design invariants

| Invariant | Where enforced |
|-----------|---------------|
| Temperature fitted on validation, applied on test (CRIT E2) | `calibration.py`, `evaluator.py`, `evaluate_saved_model.py`, `evaluation_pipeline.py` |
| Thresholds fitted on validation, applied on test (CRIT E1) | `evaluator.py`, `evaluation_pipeline.py` |
| Task type passed explicitly, never re-inferred from label range (CRIT E5) | `metrics_engine.py`, `evaluate_model.py`, `evaluator.py`, `evaluation_pipeline.py` |
| `log_loss` aggregated separately by sample count, not averaged with bounded metrics (CRIT E3) | `metrics_engine.py` |
| Non-finite metrics coerced to 0.0 before JSON serialization (Section 5) | `evaluation_pipeline.py`, `metrics_engine.py` |
| Softmax K-1 dummy encoding before correlation to remove collinearity (HIGH E10) | `task_correlation.py` |
| Ablation/permutation/SHAP never imported from inference paths (Audit §8) | `importance/*.py` module docstrings |

---

## 2. Module File Index

```
src/evaluation/
├── __init__.py                          # Empty — namespace package
├── advanced_analysis.py                 # Graph metrics, batched predict, service-aware importance dispatch
├── calibration.py                       # Temperature / Platt / Isotonic calibration + ECE / Brier / reliability
├── error_analysis.py                    # Binary / multiclass / multilabel error analysis + ErrorAnalyzer class
├── evaluate_model.py                    # Top-level numpy-friendly evaluate() entry point
├── evaluate_saved_model.py              # Offline pipeline: load JSON preds → metrics → calibration → report
├── evaluation_dashboard.py              # Streamlit interactive dashboard
├── evaluation_engine.py                 # In-loop trainer evaluation: DataLoader → MetricsEngine → val_loss
├── evaluation_pipeline.py              # Main run_evaluation_pipeline() orchestrator
├── evaluator.py                         # Evaluator class: inference + calibration + error + threshold
├── fairness.py                          # Group fairness metrics + bootstrap CI
├── metrics_engine.py                    # Single source of truth for all classification metrics
├── mlflow_tracker.py                    # MLflow run context, metric/param/artifact logging
├── pdf_report.py                        # ReportLab PDF report generation
├── prediction_collector.py             # Batched text → logits → predictions, DataLoader path, streaming
├── reliability_diagram.py              # Binned reliability stats + matplotlib plots + ReliabilityDiagram class
├── report_writer.py                     # JSON save + PNG plot generation (threaded) + _make_serializable
├── task_correlation.py                  # Spearman/Pearson task correlation with K-1 encoding
├── threshold_optimizer.py              # Binary / multilabel threshold optimization + constrained optimization
├── uncertainty.py                       # Entropy, confidence, margin, energy, mutual information, drift
└── importance/
    ├── __init__.py                      # Empty
    ├── feature_ablation.py              # FeatureAblation: single, group, bootstrap, rank — OFFLINE ONLY
    ├── permutation_importance.py        # PermutationImportance: compute, variance, group — OFFLINE ONLY
    └── shap_importance.py               # ShapImportance: TreeExplainer / LinearExplainer / KernelExplainer — OFFLINE ONLY
```

### Dependency graph (simplified)

```
evaluation_pipeline.py
    ├── prediction_collector.py    (collect_all_tasks)
    ├── evaluate_model.py          (evaluate, _postprocess_logits)
    ├── calibration.py             (fit_calibration, apply_temperature, compute_calibration)
    ├── uncertainty.py             (uncertainty_statistics)
    ├── error_analysis.py          (error_analysis)
    ├── threshold_optimizer.py     (optimize_thresholds)
    ├── task_correlation.py        (compute_task_correlation)
    ├── fairness.py                (fairness_report_multi)
    ├── report_writer.py           (save_report)
    ├── pdf_report.py              (generate_pdf_report)
    └── mlflow_tracker.py          (log_task_metrics, log_evaluation_report)

evaluation_engine.py
    ├── metrics_engine.py          (MetricsEngine.compute_multitask)
    └── prediction_collector.py    (collect_all_tasks_from_loader)

evaluator.py
    ├── calibration.py
    ├── error_analysis.py          (ErrorAnalyzer)
    ├── metrics_engine.py          (compute_metrics_from_preds)
    ├── prediction_collector.py    (PredictionCollector)
    └── threshold_optimizer.py     (ThresholdOptimizer)

advanced_analysis.py
    ├── importance/feature_ablation.py
    ├── importance/permutation_importance.py
    └── importance/shap_importance.py
```

---

## 3. Architecture and Design Principles

### 3.1 Two evaluation surfaces

The system provides two surfaces that serve different callers:

**Surface A — Trainer in-loop evaluation (`EvaluationEngine`)**  
Used during training to compute validation metrics and derive `val_loss` for early stopping. Accepts a `DataLoader`, runs the model forward pass, collects per-task predictions, and reduces to a multi-task metric dict. The `val_loss` derivation prefers `log_loss` over `1 - balanced_accuracy` over `1 - accuracy`, in that order, to guard against degenerate models on imbalanced datasets.

**Surface B — Offline / post-hoc evaluation (`run_evaluation_pipeline`, `evaluate_and_save`)**  
Used after training or deployment to produce a full diagnostic report. Accepts either a model + tokenizer or a `PredictionService` instance, and can also process pre-computed prediction arrays stored as JSON.

### 3.2 PredictionService abstraction

The pipeline is not coupled to the internal `MultiTaskTruthLensModel`. Any object exposing `predict(text) → dict` or `predict_batch(texts) → List[dict]` can be substituted via the `prediction_service` argument. Pre-allocated typed numpy arrays (`np.zeros` per task per slot) are built from `TASK_CONFIG[task]["num_labels"]` before iteration to avoid ragged lists and silent object-dtype arrays (HIGH E6).

### 3.3 Shared-encoder multi-task collection

`collect_all_tasks` detects whether the model exposes `encode()` and `forward_heads()`. When both are present, the RoBERTa encoder runs **once per batch** and all six task heads fan out from the shared hidden state. When absent (older checkpoints), it falls back to calling `collect_predictions` once per task, which re-runs the full encoder per task.

### 3.4 Calibration split contract

All calibration code enforces:
- `fit_calibration(val_logits, val_y_true, task_type)` → fitted temperature `T`
- `compute_calibration(logits, y_true, task_type, temperature=T)` → ECE / Brier / reliability

`apply_temp_scaling=True` (fit on the same data metrics are measured on) is still accepted for backwards compatibility but always logs a `WARNING` about statistical bias. When a validation-fitted `T` is present, the pipeline recomputes `metrics_calibrated` from scaled logits so the lift from temperature scaling is visible in the report.

### 3.5 Early stopping signal hierarchy

`EvaluationEngine._extract_val_loss` selects the scalar used for early stopping in this order:

1. `__aggregate__.log_loss` — sample-weighted across tasks (preferred; same scale, meaningful for imbalanced sets)
2. Mean per-task `log_loss` — when aggregate key is absent
3. `1 - __aggregate__.balanced_accuracy` — guards against class-prior baselines on imbalanced tasks
4. `1 - mean balanced_accuracy` — per-task fallback
5. `1 - accuracy` — last resort for multilabel-only runs
6. `0.0` — when no metrics are available

---

## 4. End-to-End Data Flow

### 4.1 Full pipeline (`run_evaluation_pipeline`)

```
Input
  texts: List[str]
  labels: Dict[task, array]
  model / tokenizer  OR  prediction_service
  val_logits / val_labels (optional, for calibration split)
  sensitive_attributes (optional, for fairness)
        │
        ▼
[1] PREDICTION COLLECTION
    PredictionService path:
      _collect_via_prediction_service()
        → pre-allocate typed numpy slots per task (HIGH E6)
        → predict_batch() or predict() in batch_size chunks (HIGH E2)
    Model path:
        collect_all_tasks() → shared encoder or per-task fallback
        │
        ▼
[2] CALIBRATION FIT (if val_logits provided)
    fit_calibration(val_logits[task], val_labels[task], task_type)
    → fitted_temperatures: Dict[task, float]
        │
        ▼
[3] PER-TASK LOOP (for each task in tasks ∩ labels):
    a. Shape validation (_validate_pred_shape)
       - multilabel: threshold float probs at 0.5
       - binary/multiclass: argmax 2D arrays to 1D
    b. evaluate(y_true, y_pred, y_proba, task)
       → metrics dict (accuracy, f1, MCC, ROC-AUC, log_loss, confusion_matrix, ...)
    c. optimize_thresholds(y_true, probs, task)
       → optimal_thresholds[task]
    d. compute_calibration(logits, y_true, task_type, temperature=T)
       → calibration[task] (ECE, Brier, reliability_diagram, confidence)
       If fitted T: recompute metrics_calibrated from scaled logits
    e. uncertainty_statistics(probs, task, logits)
       → uncertainty[task]
    f. error_analysis(y_true, y_pred, probs, texts, task)
       → error_analysis[task]
    g. log_task_metrics(task, metrics) [MLflow, optional]
        │
        ▼
[4] CROSS-TASK
    compute_task_correlation(all_probs)   → task_correlation
    fairness_report_multi(...)            → fairness (if sensitive_attributes)
        │
        ▼
[5] SUMMARY
    Per-task f1 → worst_task_f1, f1_imbalance_index
    Non-finite metrics coerced to 0.0
        │
        ▼
[6] OUTPUT
    save_report(report, output_path)      → JSON + PNG plots
    generate_pdf_report(report, ...)      → PDF (ReportLab)
    log_evaluation_report(report)         → MLflow artifact
```

### 4.2 Saved-model / offline pipeline (`evaluate_and_save`)

```
pred_path (JSON)  label_path (JSON)  [pred_probs] [logits] [dataset_path CSV]
        │
        ▼
load_json → validate_inputs() (common tasks, equal lengths)
        │
        ▼
evaluate_tasks()     → per-task metrics
compute_all_calibration()  → ECE / Brier (fit-on-val when val_logits provided)
uncertainty_statistics()   → per-task uncertainty stats
compute_task_correlation() → correlation matrix
actor_graph_metrics()      → networkx graph stats (if df provided)
        │
        ▼
save_report() → JSON
generate_pdf_report() → PDF
```

### 4.3 Trainer in-loop (`EvaluationEngine.evaluate`)

```
model + DataLoader
        │
collect_all_tasks_from_loader()
  → per-batch: move_batch → model(task=t) → logits → _postprocess
  → per-task: vstack logits, vstack labels
        │
MetricsEngine.compute_multitask()
  → compute_task() per task → compute_metrics_from_preds()
  → _aggregate() → sample-weighted bounded metrics + log_loss
        │
_extract_val_loss(metrics) → scalar
        │
{"metrics": metrics, "val_loss": float}
```

---

## 5. Prediction Collection

**File:** `prediction_collector.py`

### 5.1 Core functions

| Function | Description |
|----------|-------------|
| `collect_predictions(model, texts, task, tokenizer, ...)` | Single-task collection. Tokenizes, runs forward pass in batches, handles four logit output shapes (A/B/C/D), returns `{task, task_type, logits, probabilities, predictions}`. |
| `collect_all_tasks(model, texts, tokenizer, ...)` | Multi-task collection. Detects `encode()` + `forward_heads()` for shared-encoder mode; falls back to per-task `collect_predictions`. |
| `collect_all_tasks_from_loader(model, dataloader, ...)` | DataLoader path for `EvaluationEngine`. Normalizes label dict (single tensor or task-keyed dict), buffers logits/labels across batches. |
| `stream_logits(model, texts, task, tokenizer, ...)` | Generator that yields per-batch logit arrays. For large datasets that cannot fit all logits in memory. |

**Class:** `PredictionCollector` — stateless wrapper that packs `y_true / y_pred / y_proba / logits / task / task_type` into a uniform dict for downstream consumers (ErrorAnalyzer, ThresholdOptimizer).

### 5.2 Logit output shape negotiation

The collector handles four output formats that arise from different model paths:

| Shape | Key pattern | Notes |
|-------|-------------|-------|
| A | `out[f"{task}_logits"]` | Predictor._format_outputs flattened output |
| B | `out[task]["logits"]` | Raw multi-task forward, per-task sub-dict |
| C | `out["task_logits"][task]` | Raw multi-task forward, parallel view |
| D | `out["logits"]` | Legacy single-task or generic |

### 5.3 Postprocess (`_postprocess`)

Uses `scipy.special.softmax` and `expit` directly on numpy arrays, skipping the numpy → torch → numpy round-trip (HIGH E3). All three task types handled:

- `multiclass`: softmax over last axis → argmax
- `binary`: softmax (2-logit head) or expit (1-logit head), threshold at 0.5
- `multilabel`: expit element-wise, threshold at 0.5 (configurable)

---

## 6. Metrics Engine

**File:** `metrics_engine.py`

### 6.1 Public API

| Function / Class | Signature | Returns |
|------------------|-----------|---------|
| `compute_classification_metrics` | `(y_true, y_pred, *, y_proba, average, threshold, confidence, labels, task_type)` | `Dict[str, Any]` |
| `compute_multilabel_metrics` | `(y_true, y_pred, *, y_proba, threshold)` | `Dict[str, Any]` |
| `compute_metrics_from_preds` | `(y_true, y_pred, *, task_type, y_proba, threshold, average)` | `Dict[str, Any]` |
| `MetricsEngine` | class | multi-task orchestrator |
| `MetricsEngineConfig` | dataclass | `default_threshold`, `enable_confidence_weighting`, `return_per_task`, `aggregate` |

### 6.2 Classification metrics (binary + multiclass)

Minimum guaranteed keys: `accuracy`, `balanced_accuracy`, `precision`, `recall`, `f1`, `f1_macro`, `f1_micro`, `f1_weighted`, `mcc`, `confusion_matrix`, `per_class_f1`, `metric_average`.

Optional (require `y_proba`): `roc_auc` (skipped when y_true has a single class — logged at WARNING), `log_loss`.

**Task type routing (CRIT E5):** The authoritative `task_type` is forwarded explicitly. The legacy `{0, 1}`-based heuristic that silently misrouted pruned multiclass slices is only the last fallback when no `task_type` is provided.

**Binary ROC-AUC:** `_binary_proba_for_auc` extracts `p[:, 1]` when proba is (N, 2) or returns the 1D array as-is. `roc_auc_score` is only called when `y_true` has more than one class.

### 6.3 Multilabel metrics

Minimum guaranteed keys: `subset_accuracy`, `element_accuracy`, `hamming_loss`, `f1_micro`, `f1_macro`, `f1_samples`, `f1_weighted`, `per_label_f1`, `threshold`.

Optional (require `y_proba`): `log_loss` (averaged over valid labels using sklearn `log_loss` per column — CRIT E6 — matching the multiclass scale), `roc_auc_macro` (columns with all-zero or all-one labels excluded).

### 6.4 MetricsEngine: multi-task orchestration

`compute_multitask(predictions, task_types, thresholds)` iterates tasks, calls `compute_task`, tracks per-task sample counts, then calls `_aggregate`.

**Aggregation (`_aggregate`):**

Bounded metrics (accuracy, balanced_accuracy, f1_*, mcc, roc_auc, hamming_loss, etc.) are **sample-count weighted** — a 50-sample edge task cannot swing the aggregate the same as a 50,000-sample core task.

`log_loss` is **excluded from bounded-metric averaging** (different scale `[0, ∞)`) and aggregated separately with its own sample-weighted average under the `log_loss` key (CRIT E3).

Additional aggregate keys: `worst_task_f1`, `worst_task_f1_name`, `f1_imbalance_index` = `clip((best - worst) / best, 0, 1)`, `num_tasks`.

### 6.5 Input validation helpers

| Helper | Contract |
|--------|----------|
| `_as_1d_int_array(values, name)` | Raises `ValueError` if empty or not 1D |
| `_as_2d_int_array(values, name)` | Raises `ValueError` if empty or not 2D |
| `_check_shape_match(a, b)` | Raises `ValueError` on shape mismatch |
| `_binary_proba_for_auc(y_proba)` | Returns 1D P(positive) or None |

---

## 7. Calibration System

**File:** `calibration.py`  
**Re-export:** `TemperatureScaler` from `src.models.calibration.temperature_scaling` (stable public symbol, no layering violation)

### 7.1 Calibration classes

| Class | Algorithm | Notes |
|-------|-----------|-------|
| `TemperatureScaler` | Single scalar T, LBFGS | Defined in `src.models.calibration`, re-exported here |
| `VectorTemperatureScaler` | Per-label T array for multilabel | Fits one binary-path T per label column; prevents a single T from squeezing diverse per-label regimes |
| `PlattCalibrator` | Logistic regression on scores | Rescales + shifts; outperforms scalar T when prior has shifted |
| `IsotonicCalibrator` | Isotonic regression | Monotone non-parametric; preferred when validation set > ~1k samples |

### 7.2 Core functions

| Function | Description |
|----------|-------------|
| `fit_temperature(logits, labels, task_type, max_iter)` | LBFGS over CrossEntropy (multiclass), BCEWithLogitsLoss (binary/multilabel). Clamps T ≥ 1e-3 inside the closure to prevent NaN. Returns 1.0 on invalid result. |
| `apply_temperature(logits, T)` | `logits / max(T, EPS)` — pure numpy, no torch. |
| `fit_calibration(val_logits, val_y_true, task_type)` | Thin wrapper: validates `task_type`, calls `fit_temperature`. |
| `compute_calibration(logits, y_true, task_type, *, apply_temp_scaling, temperature, n_bins, return_confidence_array)` | Full calibration pipeline. Prefers pre-fitted `temperature`; `apply_temp_scaling=True` fits on test (deprecated, logs WARNING). Returns ECE, Brier, reliability_diagram, mean/std confidence, and optionally the confidence array. |

### 7.3 ECE computation (`expected_calibration_error`)

Uses equal-width bins `[0, 1]` with `n_bins=10` (default). Raises `ValueError` on empty input rather than returning `0.0` for a degenerate look of perfect calibration. Per-bin accuracy and confidence are weighted by bin occupancy for the ECE sum.

Variants:
- `classwise_ece(y_true, probs, n_bins)` → `Dict[str, float]` — one ECE per class (binary OvR slices)
- `multilabel_ece(y_true, probs, n_bins)` → `{macro_ece, per_label_ece}` — per-label binary ECE, then macro average

### 7.4 Brier score

- `multiclass`: `mean(sum((p - onehot)^2, axis=1))`
- `binary`: `mean((p_positive - y)^2)`
- `multilabel`: `mean((p - y)^2)` (element-wise)

### 7.5 Activations

`softmax` delegates to `scipy.special.softmax` (numerically stable, matches `prediction_collector`). `sigmoid` delegates to `scipy.special.expit` (prevents overflow on large negative inputs that hand-rolled `1/(1+exp(-x))` cannot handle).

### 7.6 Confidence extraction (`extract_confidence`)

- `multilabel`: `mean(max(p, 1-p), axis=1)` — mean best-label confidence
- `binary` 1D: `max(p, 1-p)` — distance from 0.5
- `multiclass` / default: `max(probs, axis=1)`

---

## 8. Uncertainty Quantification

**File:** `uncertainty.py`

### 8.1 Functions

| Function | Inputs | Output | Description |
|----------|--------|--------|-------------|
| `predictive_entropy(probs)` | (N, K) | (N,) | Shannon entropy: `-sum(p * log(p))`. Requires 2D input. |
| `normalized_entropy(probs)` | (N, K) | (N,) | `H / log(K)` — scale-free across different class counts. |
| `confidence_scores(probs)` | (N, K) or (N,) | (N,) | `max(probs, axis=-1)`. Accepts 1D (binary P(pos)). |
| `margin_confidence(probs)` | (N, K) | (N,) | Top-1 minus top-2 probability. Returns zeros if K < 2. |
| `confidence_weighted_entropy(probs)` | (N, K) | (N,) | `H(p) * (1 - max(p))` — high when uncertain AND unconfident. |
| `multilabel_uncertainty(probs)` | (N, L) | dict | Per-label binary entropy + mean_entropy + confidence, all clipped to `[EPS, 1-EPS]`. |
| `predictive_variance(prob_samples)` | (T, N, C) | (N,) | Variance of MC-dropout samples along T axis, averaged over classes. |
| `mutual_information(prob_samples)` | (T, N, C) | (N,) | `H(E[p]) - E[H(p)]`, normalized by `log(K)` (HIGH E11: prevents per-batch re-scaling that made cross-batch thresholds meaningless). |
| `energy_score(logits)` | (N, K) | (N,) | Negative log-sum-exp of logits (OOD signal), z-score normalized. |
| `uncertainty_drift(entropy)` | (N,) | dict | `entropy_shift` (mean), `entropy_spread` (std), `high_uncertainty_ratio` (>0.8). |

### 8.2 `uncertainty_statistics` (aggregate scalar stats)

Main public API for the pipeline. Returns a flat dict of scalars:

```
mean_entropy, std_entropy, min_entropy, max_entropy,
p95_entropy, p99_entropy,
mean_confidence, std_confidence, min_confidence, max_confidence,
mean_weighted_entropy,
mean_margin  (multiclass/binary only),
mean_energy, std_energy  (if logits provided),
mean_mutual_information  (if prob_samples provided),
entropy_shift, entropy_spread, high_uncertainty_ratio  (drift signals),
uncertainty_explanation_corr  (if explanation_scores provided)
```

### 8.3 `uncertainty_per_sample`

Returns per-sample arrays (not scalars): `entropy`, `confidence`, `weighted_entropy`, `margin` (multiclass), `energy` (if logits), `mutual_information` (if MC samples). Used for hard-example ranking and visualization.

### 8.4 Validation

`_validate_probs` always returns a 2D `(N, K)` array. 1D inputs are accepted only when `allow_1d=True` (then reshaped to `(N, 1)`). Empty arrays raise `ValueError`.

---

## 9. Error Analysis

**File:** `error_analysis.py`

### 9.1 Task-specific analyzers

**Binary (`analyze_binary_errors`):**
- Counts false positives (y=0, pred=1) and false negatives (y=1, pred=0)
- Surfaces `top_false_positives`: hardest FPs ranked by highest P(class=1) — these are the most confident wrong positives
- Surfaces `top_false_negatives`: hardest FNs ranked by **lowest** P(class=1) (HIGH E12: direct sort on raw confidence, avoiding float32 rounding in the complementary `1-p` that caused tie reordering)
- `_binary_positive_proba` coerces (N,), (N,1), (N,2) shapes to a 1D P(positive) vector

**Multiclass (`analyze_multiclass_errors`):**
- `total_errors`: count of misclassified samples
- `confusion_pairs`: `{true→pred: count}` dict of most common error transitions
- `error_rate_per_class`: per-class fraction of samples where y_true[mask] ≠ y_pred[mask]
- `hard_examples`: top-k most confident wrong predictions (high max prob, still wrong)

**Multilabel (`analyze_multilabel_errors`):**
- `per_label_error_count`: number of wrong predictions per label column
- `total_error_labels`: total cell-level errors across all labels
- `hard_samples`: top-k samples with the most simultaneously wrong labels

### 9.2 Main entry point (`error_analysis`)

```python
error_analysis(
    y_true, y_pred, *,
    probs=None, texts=None,
    task=None, task_type=None,
    top_k=10
) -> Dict[str, Any]
```

Routes to the correct analyzer by `task_type`. If neither is provided, infers from `y_true.ndim` (2D → multilabel) and number of unique values (≤2 → binary).

### 9.3 `ErrorAnalyzer` class

Stateless OO wrapper used by `Evaluator`. `analyze(collected_dict)` extracts `y_true / y_pred / y_proba / task / task_type` from a `PredictionCollector` record and calls `error_analysis`, returning `{}` on any shape/type error.

---

## 10. Fairness and Group Metrics

**File:** `fairness.py`

### 10.1 Functions

| Function | Description |
|----------|-------------|
| `per_group_metrics(y_true, y_pred, groups, *, positive_label, task_type)` | Accuracy, precision, recall, F1, positive rate per group value. Routes binary vs macro averaging by explicit `task_type` (Section 10: prevents 3-class slices with labels {0,1} from misrouting). |
| `demographic_parity(y_pred, groups, *, positive_label)` | Positive prediction rate per group + `max_diff` + `ratio` (disparate impact). |
| `equal_opportunity(y_true, y_pred, groups, *, positive_label)` | Per-group TPR + `max_diff`. NaN for groups with no positives. |
| `equalized_odds(y_true, y_pred, groups, *, positive_label)` | Per-group TPR + FPR + `tpr_max_diff` + `fpr_max_diff`. |
| `fairness_report(y_true, y_pred, groups, *, positive_label, group_name, task_type)` | Full single-attribute report: per_group_metrics + demographic_parity + equal_opportunity + equalized_odds. |
| `fairness_report_multi(y_true, y_pred, sensitive_attributes, *, positive_label, task_type)` | Iterates `{attr_name: values}` dict, calling `fairness_report` per attribute. |
| `bootstrap_metric_ci(y_true, y_pred, metric_fn, *, n_bootstrap, alpha, seed)` | Percentile bootstrap CI for any scalar metric. Returns `{point, low, high, n}`. Filters non-finite bootstrap samples. |

### 10.2 Input to the pipeline

`run_evaluation_pipeline` accepts `sensitive_attributes: Dict[task, Dict[attr_name, values]]`. A fairness block is only emitted for tasks explicitly listed — downstream reports are not polluted with empty slices for tasks without attributes.

### 10.3 Bootstrap CI usage

```python
from src.evaluation.fairness import bootstrap_metric_ci, equal_opportunity

def tpr_diff(y_true, y_pred):
    result = equal_opportunity(y_true, y_pred, groups)
    return result["max_diff"]

ci = bootstrap_metric_ci(y_true, y_pred, tpr_diff, n_bootstrap=500, alpha=0.05)
# → {"point": 0.04, "low": -0.01, "high": 0.09, "n": 500}
```

---

## 11. Feature Importance (Offline)

**Location:** `src/evaluation/importance/`  
**Critical constraint:** These modules are **OFFLINE-ONLY**. They must not be imported by `src/inference/`, `api/app.py`, or any model forward path. Ablation sweeps multiply latency by the number of features; SHAP is exponential in feature count or several seconds per sample under sampling approximation.

### 11.1 `FeatureAblation` (`feature_ablation.py`)

```python
@dataclass
class FeatureAblation:
    model: object          # must expose predict()
    metric: MetricFn       # default: accuracy
    normalize: bool        # divide impact by baseline score
    bootstrap_runs: int    # 0 = disabled
```

| Method | Description |
|--------|-------------|
| `single_feature_ablation(X, y, feature_names)` | Zero out column i → score drop → importance. Baseline score is cached. |
| `group_ablation(X, y, feature_names, groups)` | Zero out all columns in a group simultaneously. |
| `bootstrap_ablation(X, y, feature_names)` | Bootstrap resampling over `bootstrap_runs` iterations → `{name: (mean, std)}`. |
| `rank_features(X, y, feature_names)` | Returns list of `(name, importance)` tuples sorted descending. |
| `top_k(X, y, feature_names, k)` | First k entries from `rank_features`. |

**Integration with `advanced_analysis.py`:** `ablation_importance(model, texts, y, feature_names, task, tokenizer, metric, ...)` wraps a `predict_texts` call (service-aware) as the `predict_fn` passed to `FeatureAblation`.

### 11.2 `PermutationImportance` (`permutation_importance.py`)

```python
@dataclass
class PermutationImportance:
    model: object
    metric: MetricFn     # default: accuracy
    n_repeats: int = 5   # shuffle repeats per feature for variance reduction
    random_seed: int = 42
    normalize: bool = True
    use_proba: bool = False  # use predict_proba if available
```

| Method | Description |
|--------|-------------|
| `compute(X, y, feature_names)` | Shuffle column j × n_repeats → mean score drop. |
| `compute_with_variance(X, y, feature_names)` | Returns `{name: (mean, std)}`. |
| `group_permutation(X, y, feature_names, groups)` | Shuffle all group columns jointly per repeat. |
| `rank_features / top_k` | Same interface as `FeatureAblation`. |

### 11.3 `ShapImportance` (`shap_importance.py`)

```python
@dataclass
class ShapImportance:
    model: object
    max_samples: int = 1000    # subsample for efficiency
    batch_size: int = 128
    random_seed: int = 42
    use_interactions: bool = False
```

Explainer selection strategy:
1. `shap.TreeExplainer` (if model has `predict_proba`)
2. `shap.LinearExplainer`
3. `shap.KernelExplainer` on 100-sample background (fallback)

SHAP values are L1-normalized (`mean |shap| / sum(mean |shap|)`) for cross-run comparability. Multi-class (3D) values collapse to the last class column. Falls back to `{name: 0.0}` when SHAP unavailable.

| Method | Description |
|--------|-------------|
| `compute(X, feature_names)` | Mean absolute SHAP, L1-normalized. |
| `compute_with_variance(X, feature_names)` | `{name: (mean, std)}` of normalized SHAP. |
| `group_importance(X, feature_names, groups)` | Sum base scores for features in each group. |
| `rank_features / top_k` | Same interface. |

---

## 12. Task Correlation

**File:** `task_correlation.py`

### 12.1 `compute_task_correlation`

```python
compute_task_correlation(
    predictions: Dict[str, Any] | pd.DataFrame,
    *,
    normalize: bool = True,
    method: Literal["pearson", "spearman"] = "spearman",
    robust: bool = True,
    confidence: Optional[np.ndarray] = None,
    uncertainty: Optional[np.ndarray] = None,
    graph_signal: Optional[np.ndarray] = None,
) -> pd.DataFrame
```

**Feature extraction (`_extract_task_features`):**

Accepts `PredictionCollector`-style dicts (extracts `y_proba / probabilities / y_pred` in priority order).

| Task type | Encoding |
|-----------|----------|
| `binary` | 1D P(positive) |
| `multiclass` | K-1 columns (last dropped to remove softmax collinearity — HIGH E10) |
| `multilabel` | All K label columns (independent sigmoids, no collinearity) |

The `_DELIM = "::"` separator ensures that task names containing underscores (e.g. `narrative_frame`) do not collide with column name parsing.

`_resolve_task_type` is `@lru_cache(maxsize=8)` — the six-task project never evicts.

**Pre-processing:**
1. `_winsorize(df, lower=0.01, upper=0.99)` — clips outliers (when `robust=True`)
2. `_normalize(df)` — z-score (when `normalize=True`)
3. Optionally append `global_confidence`, `global_uncertainty`, `graph_signal` columns

Replaces ±inf and NaN in the final correlation matrix with 0.0.

### 12.2 `aggregate_task_correlation`

Collapses the per-column correlation matrix (which may have K-1 or L sub-columns per task) into a task×task matrix by averaging the block of coefficients between each task pair. Self-correlation forced to 1.0.

### 12.3 `correlation_statistics`

Returns `{mean_correlation, std_correlation, max_correlation, min_correlation, high_correlation_ratio}` over all finite values in the matrix. `high_correlation_ratio` = fraction of |r| > 0.8 (co-linearity alert threshold).

### 12.4 `save_correlation_matrix`

Writes the correlation `pd.DataFrame` to CSV at the specified path, creating parent directories.

---

## 13. Threshold Optimization

**File:** `threshold_optimizer.py`

### 13.1 Functions

| Function | Task type | Strategy | Returns |
|----------|-----------|----------|---------|
| `default_threshold(task, fallback)` | any | lookup `TASK_CONFIG[task]["threshold"]` | `float` |
| `optimize_binary_threshold(y_true, probs, *, metric)` | binary | PR-curve sweep, argmax of F1/precision/recall | `{threshold, score, metric, valid, reason?}` |
| `optimize_constrained(y_true, probs, *, min_precision, min_recall, objective)` | binary | Constrained PR-curve sweep | `{threshold, score, precision, recall, objective, valid, reason?}` |
| `optimize_multilabel_thresholds(y_true, probs, *, metric, strategy)` | multilabel | `per_label` or `global` | See below |
| `optimize_thresholds(y_true, probs, *, task, metric, strategy)` | any | routes by `task_type` | unified result dict |

### 13.2 Binary threshold

Uses `sklearn.metrics.precision_recall_curve` and drops the trailing boundary point. Returns `valid=False, reason="single_class"` when y_true has only one class (Section 5: prevents spurious 0.5 being treated as a tuned threshold downstream).

### 13.3 Multilabel: `per_label` strategy

Calls `_score_per_label` independently for each label column. Returns `{strategy, thresholds: List[float], scores: List[float], mean_score}`. Single-class columns tagged `valid=False, reason="single_class"` without crashing the loop.

### 13.4 Multilabel: `global` strategy (vectorized — HIGH E9)

Builds the full `(N, L, T)` prediction tensor once, reduces TP/FP/FN along the sample axis, then picks the candidate threshold that maximizes macro-F1/precision/recall. Avoids calling `f1_score` inside the candidate loop (which was O(T × N × L) per threshold).

### 13.5 `ThresholdOptimizer` class

Stateless wrapper used by `Evaluator`. `optimize(collected)` extracts `y_true / y_proba / task_type / task` from a `PredictionCollector` record. `optimize_from_arrays(y_true, probs, task_type)` is the direct-array path used by `Evaluator` when validation arrays are available separately.

---

## 14. Report Generation

### 14.1 JSON + plots (`report_writer.py`)

**`save_report(report, path, generate_plots)`** is the main entry point:

1. Injects `metadata` block: UTC timestamp, `evaluation_version="v4"`, task list.
2. Serializes via `_make_serializable` (see below) and writes JSON.
3. If `generate_plots=True`, creates a directory tree:
   ```
   <report_dir>/plots/
   ├── summary/      summary.png
   ├── tasks/        {task}_metrics.png, {task}_per_class_f1.png, {task}_per_label_f1.png
   ├── calibration/  {task}_calibration.png, {task}_classwise_ece.png,
   │                 {task}_per_label_ece.png, {task}_reliability.png
   ├── error_analysis/  {task}_error_rate.png
   ├── confidence/   {task}_confidence.png  (histogram)
   ├── thresholds/   {task}_threshold.png  (bar) / {task}_thresholds.png  (list)
   └── monitoring/   {key}.png
   ```
4. All plot jobs collected first, then dispatched to a `ThreadPoolExecutor(max_workers=4)` (Section 7: I/O overlap cuts wall time ~50% on 6-task reports).

**`_make_serializable(obj)` type handling:**

| Type | Behaviour |
|------|-----------|
| `None / str / int / float / bool` | Pass-through |
| `dict` | Recurse; convert all keys to `str` |
| `list / tuple / set` | Recurse; truncate to `_MAX_LIST_LEN = 5000` |
| `np.ndarray` | `.tolist()` for small arrays; for large (> 5000 elements): `{shape, dtype, truncated:True, data}` preserving shape metadata (HIGH E14) |
| `np.integer / np.floating / np.bool_` | Cast to Python int/float/bool |
| `torch.Tensor` | `.detach().cpu().numpy()` then handle as ndarray |
| `pd.DataFrame / pd.Series` | `.to_dict()` then recurse |
| `Path` | `str(path)` |
| `datetime.datetime` | `.isoformat()` |
| Anything else with `.tolist()` | Call it |
| Fallback | `str(obj)` |

### 14.2 PDF report (`pdf_report.py`)

Uses `reportlab.platypus` (SimpleDocTemplate, Paragraph, Table, PageBreak, Spacer). Layout:

1. Title: "TruthLens AI Evaluation Report"
2. Per-task section (`render_tasks`): Task heading → Metrics table → Dataset Statistics table → PageBreak
3. Generic sections via `render_section(elements, title, data)` (skipped if data is empty):
   - Overall Summary
   - Calibration
   - Error Analysis
   - Optimal Thresholds
   - Graph Features / Graph Explanation
   - Drift Detection
   - Monitoring
   - Uncertainty
   - Task Correlation
   - Advanced Analysis

`flatten_nested` converts nested dicts/lists to `(key.subkey, value)` rows. Lists are truncated to 10 items with `"..."`.

Table style: light grey header row, 0.25pt grey grid, Helvetica-Bold header font.

### 14.3 Streamlit dashboard (`evaluation_dashboard.py`)

**`launch_dashboard(report_path)`** loads a JSON report and renders:

Left column: `render_metrics` → `render_dataset_stats` → `render_calibration` (ECE, Brier, confidence histogram, reliability diagram) → `render_thresholds`

Right column: `render_uncertainty` → `render_confusion` (matplotlib imshow with cell labels; reads `metrics["confusion_matrix"]` — HIGH E13: fixed wrong key path) → `render_error_analysis`

Full width: `render_correlation` (matshow heatmap) → `render_advanced`

All renders gracefully skip when data is absent.

---

## 15. MLflow Tracking

**File:** `mlflow_tracker.py`

MLflow is an optional dependency. All functions start with `_ensure_mlflow()` which raises `RuntimeError("MLflow not installed")` if the package is absent. All logging is guarded by `is_primary_process()` (from `src.utils.device_utils`) to prevent duplicate metric writes in DDP training.

### 15.1 `MLflowRun` context manager

```python
with MLflowRun(experiment_name="truthlens", run_name="eval_v2", tags={"env": "prod"}) as run:
    log_task_metrics("bias", metrics_dict)
    log_params({"model": "roberta-base", "encoder_lr": 2e-5})
    log_model(model, tokenizer=tokenizer, config=config_dict)
```

Sets experiment, starts run, tags, ends run with `"FINISHED"` or `"FAILED"` based on exception presence.

### 15.2 Logging functions

| Function | Description |
|----------|-------------|
| `log_task_metrics(task, metrics, step)` | Flattens nested dict → `task.subkey` names, logs only `int/float` values. |
| `log_metrics(metrics, step)` | Global (non-task-prefixed) metric logging. |
| `log_params(params, prefix)` | Flattens and logs all values as params. |
| `log_dataset_info(name, size, version, hash)` | Logs `dataset.*` params. |
| `log_artifact(path, artifact_path)` | Validates file exists before logging. |
| `log_evaluation_report(report)` | Serializes report via `_make_serializable` → temp JSON → `mlflow.log_artifact` under `"evaluation/"` → deletes temp file. |
| `log_model(model, tokenizer, config, name)` | `mlflow.pytorch.log_model` + tokenizer artifacts + config JSON. |
| `log_system_info()` | Python version + platform as params. |

`flatten_dict(d, parent_key, sep)` is a recursive dict flattener used by `log_task_metrics` and `log_params`. MLflow metric names use `.` as separator (e.g. `bias.f1_macro`).

---

## 16. Developer Reference

### 16.1 Adding a new task

1. Add the task to `config/task_config.py` (`TASK_CONFIG`) with `type`, `num_labels`, and optionally `threshold`.
2. Add the task head to the model (`MultiTaskTruthLensModel`).
3. The evaluation system picks it up automatically — all loops use `TASK_CONFIG.keys()` as the default task list.

### 16.2 Plugging in a custom prediction backend

Implement either:
```python
class MyService:
    def predict(self, text: str) -> dict:
        # must return {"tasks": {task: {"probabilities": ..., "predictions": ..., "logits": ...}}}
        ...

    def predict_batch(self, texts: List[str]) -> List[dict]:
        # optional; used when available for batched efficiency
        ...
```

Pass as `prediction_service=MyService()` to `run_evaluation_pipeline`.

### 16.3 Running the full offline pipeline

```python
from src.evaluation.evaluate_saved_model import run_evaluation

report = run_evaluation(
    pred_path="outputs/preds.json",
    label_path="data/test_labels.json",
    output_report="outputs/report.json",
    pred_probs="outputs/probs.json",   # optional
    logits="outputs/logits.json",      # optional, enables calibration
    dataset_path="data/test.csv",      # optional, enables graph metrics
)
```

### 16.4 Trainer integration

```python
from src.evaluation.evaluation_engine import EvaluationEngine

engine = EvaluationEngine(task_types={"bias": "binary", "emotion": "multiclass"})
result = engine.evaluate(model, val_dataloader, device=device)
val_loss = result["val_loss"]   # use for early stopping
metrics  = result["metrics"]    # log to MLflow / TensorBoard
```

### 16.5 Calibration best practice

```python
from src.evaluation.calibration import fit_calibration, compute_calibration

# Fit on validation
T = fit_calibration(val_logits["bias"], val_labels["bias"], task_type="binary")

# Measure on test (no bias)
cal = compute_calibration(
    logits=test_logits["bias"],
    y_true=test_labels["bias"],
    task_type="binary",
    temperature=T,           # pre-fitted, no leakage
)
# cal["ece"], cal["brier"], cal["reliability_diagram"]
```

### 16.6 Adding a fairness attribute

```python
report = run_evaluation_pipeline(
    ...,
    sensitive_attributes={
        "bias": {
            "gender": gender_array,       # shape (N,) with group labels
            "source_region": region_array,
        },
    },
)
# report["fairness"]["bias"]["gender"]["demographic_parity"]["max_diff"]
```

### 16.7 Common failure modes

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ValueError: y_pred must be 1D after argmax` | Probability matrix passed as y_pred for a non-multilabel task | `_validate_pred_shape` does this automatically in the pipeline; check custom callers |
| `RuntimeError: MLflow not installed` | `mlflow` not in environment | Install mlflow or catch the import error; pipeline soft-fails gracefully |
| ECE / Brier suspiciously low | Temperature fitted on same data it is measured on | Pass `temperature` (fitted on validation) to `compute_calibration`; check WARNING logs |
| `worst_task_f1` = 0.0 in summary | One task has all samples in one class | Check dataset balance; consider oversampling or weighted loss |
| Graph metrics empty | `df` has no `hero_entities` / `villain_entities` columns, or all NaN | Expected — `actor_graph_metrics` returns `{}` for empty graphs |
| Streamlit confusion matrix not rendering | `metrics["confusion_matrix"]` absent | Ensure `compute_classification_metrics` ran with non-empty `y_true`/`y_pred` |
| `collect_all_tasks_from_loader` TypeError on labels | DataLoader returns a single tensor instead of a dict | Supported when exactly one task is selected — declare single task or update DataLoader |

### 16.8 Key constants

| Constant | Location | Value | Purpose |
|----------|----------|-------|---------|
| `EPS` | `metrics_engine.py`, `calibration.py`, `uncertainty.py`, `task_correlation.py`, `threshold_optimizer.py` | `1e-12` | Numerical stability floor |
| `_MAX_LIST_LEN` | `report_writer.py` | `5000` | Max list/array entries per JSON field |
| `_PLOT_POOL_SIZE` | `report_writer.py` | `4` | Max parallel plot threads |
| `_DELIM` | `task_correlation.py` | `"::"` | Sub-column separator (safe for task names with underscores) |
| `evaluation_version` | `report_writer.py` | `"v4"` | Report format version embedded in metadata |

### 16.9 Imports overview

| External library | Used in |
|-----------------|---------|
| `numpy` | All files |
| `scipy.special` (softmax, expit) | `calibration.py`, `evaluate_model.py`, `evaluator.py`, `prediction_collector.py` |
| `sklearn.metrics` | `metrics_engine.py`, `fairness.py`, `threshold_optimizer.py` |
| `sklearn.isotonic` | `calibration.py` |
| `sklearn.linear_model` | `calibration.py` |
| `torch` / `torch.nn` / `torch.optim` | `calibration.py`, `evaluate_model.py`, `evaluator.py`, `prediction_collector.py`, `advanced_analysis.py` |
| `transformers` (AutoTokenizer) | `evaluate_model.py`, `evaluator.py`, `prediction_collector.py`, `advanced_analysis.py` |
| `pandas` | `error_analysis.py`, `task_correlation.py`, `evaluate_saved_model.py`, `evaluation_dashboard.py`, `report_writer.py` |
| `matplotlib` | `reliability_diagram.py`, `report_writer.py`, `evaluation_dashboard.py` |
| `networkx` | `advanced_analysis.py` |
| `reportlab` | `pdf_report.py` |
| `mlflow` | `mlflow_tracker.py` (optional) |
| `streamlit` | `evaluation_dashboard.py` (optional) |
| `shap` | `importance/shap_importance.py` (optional) |
