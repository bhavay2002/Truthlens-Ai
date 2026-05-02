# TruthLens AI — Explainability System Documentation

**Module:** `src/explainability/`  
**File count:** 19 Python files  
**Python:** 3.12 · **Encoder:** `roberta-base` · **Tasks:** `emotion`, `narrative`, `propaganda`, `bias`, `ideology`, `narrative_frame`

---

## Table of Contents

1. [Overview](#1-overview)
2. [Folder Architecture](#2-folder-architecture)
3. [End-to-End Explainability Flow](#3-end-to-end-explainability-flow)
4. [File-by-File Deep Dive](#4-file-by-file-deep-dive)
5. [Explanation Types](#5-explanation-types)
6. [Feature Importance Interpretation](#6-feature-importance-interpretation)
7. [Output Artifacts](#7-output-artifacts)
8. [Model Compatibility](#8-model-compatibility)
9. [Config Integration](#9-config-integration)
10. [Performance and Efficiency](#10-performance-and-efficiency)
11. [Validation of Explanations](#11-validation-of-explanations)
12. [Bias and Fairness Insights](#12-bias-and-fairness-insights)
13. [Extensibility Guide](#13-extensibility-guide)
14. [Common Pitfalls and Risks](#14-common-pitfalls-and-risks)
15. [Example Usage](#15-example-usage)
16. [Simple Explanation for Non-Technical Reviewers](#16-simple-explanation-for-non-technical-reviewers)

---

## 1. Overview

The explainability system is TruthLens AI's transparency and interpretability layer. It sits between the model inference layer and the end user — after the model decides whether an article is fake or real, this system explains **why** it made that decision.

### Purpose

Every prediction from TruthLens AI is accompanied by a full explanation that:
- Highlights which words most influenced the decision (token-level attribution)
- Shows agreement across multiple independent explanation methods (cross-method consistency)
- Detects emotion, bias, and propaganda signals in the text
- Measures how trustworthy and faithful each explanation is
- Provides a single aggregated importance score per token across all methods

### Role in the ML Pipeline

```
src/models/           → Trained MultiTaskTruthLensModel (encoder + task heads)
    ↓
src/inference/        → Produces predict_fn(text) → {fake_probability, ...}
    ↓
src/explainability/   ← YOU ARE HERE
    ↓
API /explain endpoint → ExplainabilityResult (JSON)
```

The explainability system does **not** re-train or modify the model. It is a post-inference, read-only interpretability layer. It calls the model many times (for SHAP/LIME perturbations) and reads attention weights (for rollout), but never updates weights.

### Relationship with Other Modules

| Upstream module | How it is used |
|----------------|----------------|
| `src/models` | `MultiTaskTruthLensModel` is passed to `bias_explainer.py` for IG and SHAP; tokenizer is used for subword alignment |
| `src/inference` | `predict_fn(text) → dict` is the universal prediction callback used by SHAP, LIME, and ExplanationMetrics |
| `src/evaluation` | `ExplanationMetrics` re-uses ablation patterns; `ExplanationConsistency` cross-checks with evaluation correlation logic |
| `src/graph` | `GraphExplainer` is integrated into the aggregator via `graph_explanation` for entity-level attribution |

### Design invariants

| Invariant | Location |
|-----------|----------|
| `ExplainabilityResult` has a single definition — `common_schema.py` (CRIT-6/7) | `common_schema.py`, `explainability_pipeline.py` |
| Faithful flag: heuristic explainers (propaganda, emotion-lexicon) set `faithful=False` (CRIT-9) | `propaganda_explainer.py`, `emotion_explainer.py`, `common_schema.py` |
| Aggregator never mixes subword (IG) and word-level (lexicon) vectors (CRIT-2) | `emotion_explainer.py`, `explanation_aggregator.py` |
| Token order preserved — `sorted(set(...))` vocab banned (CRIT-3) | `explanation_aggregator.py` |
| Repeated tokens handled by-position, not by-name (CRIT-4) | `explanation_aggregator.py` |
| Text-level ablation via character offsets, not `" ".join(tokens)` (CRIT-11) | `explanation_metrics.py`, `orchestrator.py` |
| Base prediction computed once per article, forwarded to all 5 metrics (REC-3) | `orchestrator.py`, `explanation_metrics.py` |
| Spearman correlation uses ranks, not sort indices (CRIT-10) | `explanation_consistency.py` |
| SHAP explainer instances cached per (tokenizer, task) — never re-built per article (PERF-3) | `bias_explainer.py`, `shap_explainer.py` |
| Orchestrator singleton keyed by config hash — never re-instantiated per article (PERF-6) | `orchestrator.py` |

---

## 2. Folder Architecture

```
src/explainability/
├── __init__.py                    → Namespace package (empty)
│
├── common_schema.py               → Pydantic schemas: TokenImportance, ExplanationOutput,
│                                    AggregatedExplanation, ExplainabilityResult (CRIT-6/7)
│
├── explainability_pipeline.py     → Public API: run_explainability_pipeline(),
│                                    explain_prediction_full(), explain_fast()
│
├── orchestrator.py                → ExplainabilityOrchestrator — coordinates all sub-explainers,
│                                    applies faithfulness gate (FAITH-1), surfaces failures (FAITH-6)
│
├── model_explainer.py             → Legacy backward-compat wrapper around the orchestrator
│
│── shap_explainer.py              → SHAP text explainer with LRU explainer + value caches
├── lime_explainer.py              → LIME text explainer with bounded explanation cache
├── attention_rollout.py           → Transformer attention rollout (multi-layer, vectorised)
├── attention_visualizer.py        → Extracts attention tensors + plots heatmaps / bar charts
├── bias_explainer.py              → SHAP + Integrated Gradients + Attention fusion for bias task
├── emotion_explainer.py           → Lexicon-based emotion signal + gradient-based model attribution
├── propaganda_explainer.py        → Pattern-matching propaganda technique detection
│
├── token_alignment.py             → Sub-word → word merging (WordPiece / SentencePiece / BPE)
│
├── explanation_aggregator.py      → Multi-method weighted fusion into AggregatedExplanation
├── explanation_calibrator.py      → L1 normalization + entropy-based confidence scoring
├── explanation_consistency.py     → Pairwise Pearson / Spearman / cosine cross-method agreement
├── explanation_metrics.py         → Faithfulness, comprehensiveness, sufficiency, deletion, insertion
├── explanation_monitor.py         → Running stats + drift detection on importance score history
├── explanation_cache.py           → Memory + disk LRU cache with TTL, compression, versioning
├── explanation_report_generator.py→ JSON + HTML report generation with token highlighting
├── explanation_visualizer.py      → Matplotlib/Plotly token heatmap, bar chart, multi-method overlay
│
└── utils_validation.py            → validate_tokens_scores() — shared input guard for all explainers
```

---

## 3. End-to-End Explainability Flow

### 3.1 Standard pipeline (full mode)

```
User calls run_explainability_pipeline(text, predict_fn, model, tokenizer, config)
         │
         ▼
[0] CACHE CHECK
    ExplanationCache.get(text) → hit? return cached result : continue
         │
         ▼
[1] SUB-EXPLAINERS (each wrapped in _run() for safe isolation)
    │
    ├─ SHAP (if config.use_shap)
    │     shap_explainer.explain_text(predict_fn, text)
    │     → shap.Explainer(text masker) → SHAP values → calibrate → ExplanationOutput
    │
    ├─ LIME (if config.use_lime)
    │     lime_explainer.explain_prediction(predict_fn, text)
    │     → LimeTextExplainer.explain_instance() → calibrate → ExplanationOutput
    │
    ├─ BIAS + EMOTION (if config.use_bias_emotion AND model AND tokenizer)
    │     bias_explainer.explain_bias(model, tokenizer, text)
    │       → compute_ig() [Riemann-sum IG, 8 steps]
    │       → compute_attention_rollout() [AttentionRollout]
    │       → fuse_methods(shap=None, ig, attn) → BiasExplanation
    │     emotion_explainer.explain_emotion(text, model, tokenizer)
    │       → compute_lexicon() [word-level] + compute_gradients() [subword]
    │       → EmotionExplanation (faithful=False for lexicon signal)
    │     _wrap_bias_ig(bias) → ExplanationOutput[method="integrated_gradients"]
    │
    ├─ PROPAGANDA (if config.use_propaganda_explainer)
    │     propaganda_explainer.explain_propaganda(text)
    │     → _score_tokens + _apply_phrase_scores → calibrate → ExplanationOutput(faithful=False)
    │
    ├─ ATTENTION ROLLOUT (if config.use_attention_rollout AND tokens AND attentions)
    │     AttentionRollout.compute_rollout(attentions, tokens)
    │     → head-aggregate → residual-add → matrix product across layers → calibrate
    │     → ExplanationOutput[method="attention"]
    │     [FAITH-1 gate]: if |Spearman(attention, IG)| < threshold → drop attention
    │
    └─ GRAPH (if config.use_graph_explainer)
          GraphExplainer.explain_from_text(text)
          → node_importance dict from NER + graph centrality
         │
         ▼
[2] AGGREGATION (if config.use_aggregation)
    ExplanationAggregator.aggregate(shap, ig, attention, lime, graph_explanation)
    │
    ├─ Filter faithful sources (CRIT-9: propaganda excluded by default)
    ├─ Pick canonical token sequence (shap → ig → attn → lime priority)
    ├─ Align all sources to canonical by position or first-occurrence fallback (CRIT-3/4)
    ├─ Build [methods × tokens] matrix → vectorised weighted fusion (PERF-5)
    ├─ Compute per-token confidence = 1 - std across active methods
    ├─ Compute agreement_score via ExplanationConsistency.compute()
    └─ Return AggregatedExplanation (tokens, final_token_importance, method_weights, confidence)
         │
         ▼
[3] CONSISTENCY (if config.use_consistency)
    ExplanationConsistency.compute(shap, ig, attention, lime)
    → pairwise Pearson / Spearman / cosine (CRIT-10: real ranks, not sort indices)
    → overall_agreement, token_agreement_mean
         │
         ▼
[4] FAITHFULNESS METRICS (if config.use_explanation_metrics AND agg available)
    base_proba = predict_fn(text)["fake_probability"]  [once — REC-3]
    offsets = _compute_offsets(tokenizer, canonical_tokens, text)  [CRIT-11]
    ExplanationMetrics.evaluate(tokens, scores, batch_predict_fn,
                                text, offsets, base_proba)
    → faithfulness, comprehensiveness, sufficiency, deletion_score,
      insertion_score, overall_score
         │
         ▼
[5] MONITORING (if aggregation succeeded)
    ExplanationMonitor.update(final_token_importance)
    → running mean/std/min/max/drift
         │
         ▼
[6] ASSEMBLE ExplainabilityResult
    { prediction, shap_explanation, lime_explanation, attention_explanation,
      propaganda_explanation, bias_explanation, emotion_explanation,
      aggregated_explanation, consistency_metrics, explanation_metrics,
      monitoring, explanation_quality_score, module_failures, metadata }
         │
         ▼
[7] CACHE WRITE + RETURN
```

### 3.2 Fast pipeline

`explain_fast()` uses LIME only, skips SHAP/bias/emotion/attention/aggregation/consistency/metrics. Latency target: < 200 ms on CPU.

### 3.3 Module interaction summary

| Caller | Calls | Returns |
|--------|-------|---------|
| `explainability_pipeline` | `orchestrator.get_default_orchestrator()` | singleton `ExplainabilityOrchestrator` |
| `orchestrator` | `shap_explainer`, `lime_explainer`, `bias_explainer`, `emotion_explainer`, `propaganda_explainer`, `AttentionRollout`, `GraphExplainer` | `ExplanationOutput` per method |
| `orchestrator` | `ExplanationAggregator.aggregate()` | `AggregatedExplanation` |
| `orchestrator` | `ExplanationConsistency.compute()` | `Dict[str, float]` |
| `orchestrator` | `ExplanationMetrics.evaluate()` | `Dict[str, float]` |
| `orchestrator` | `ExplanationMonitor.update()` | nothing |
| `shap_explainer`, `lime_explainer`, `attention_rollout` | `explanation_calibrator.calibrate_explanation()` | `{scores, confidence, entropy}` |
| `bias_explainer` | `attention_rollout.AttentionRollout`, `token_alignment.align_tokens` | importance arrays |
| `explanation_aggregator` | `explanation_consistency.ExplanationConsistency` | agreement score |

---

## 4. File-by-File Deep Dive

---

### File: `common_schema.py`

**Purpose**

Single source of truth for all Pydantic data contracts used across the explainability system. Prevents schema drift between modules (CRIT-6/7).

**Key Classes**

| Class | Description |
|-------|-------------|
| `TokenImportance` | Immutable `(token: str, importance: float)` pair. Token must be non-empty; importance must be finite. |
| `ExplanationOutput` | Output of a single explainer method. Contains `method`, `tokens`, `importance`, `structured` (list of `TokenImportance`), optional `confidence`, `entropy`, `raw`, and `faithful` flag. Validators enforce alignment between flat `importance` list and `structured` list (any drift > 1e-6 raises ValueError). |
| `AggregatedExplanation` | Output of the aggregator. Contains `tokens`, `final_token_importance`, `structured`, `method_weights`, `confidence_score`, `agreement_score`, and optional `text` + `offsets` for CRIT-11 ablation. |
| `ConsistencyMetrics` | Pairwise consistency scores between explainer pairs (shap_vs_lime, shap_vs_attention, etc.). |
| `ExplanationMetricsOutput` | Five faithfulness metrics: `faithfulness`, `comprehensiveness`, `sufficiency`, `deletion_score`, `insertion_score`, `overall_score`. |
| `ExplainabilityResult` | Top-level canonical result. All optional except `prediction`. Carries all sub-explanations, `consistency_metrics`, `explanation_metrics`, `monitoring`, `explanation_quality_score`, `module_failures`, and `metadata`. Uses `extra="ignore"` so orchestrator can pass extra fields without crashing. |

**Key invariants**

- `ExplanationOutput.importance[i]` and `ExplanationOutput.structured[i].importance` are enforced equal (tolerance 1e-6)
- `faithful=True` is the default — heuristic explainers must explicitly override to `False`
- Importance values are not bounded to [0, 1] — signed SHAP values are allowed (CRIT-5)

---

### File: `explainability_pipeline.py`

**Purpose**

Public entry point. Re-exports `ExplainabilityResult` from `common_schema` (not a redefinition). Delegates all work to `get_default_orchestrator()` — a process-wide singleton that avoids the overhead of rebuilding `GraphExplainer` / `ExplanationCache` / etc. on every article (PERF-6).

**Key Functions**

| Function | Signature | Description |
|----------|-----------|-------------|
| `run_explainability_pipeline` | `(text, predict_fn, *, model, tokenizer, tokens, attentions, config)` | Main entry point. Validates text, gets/creates orchestrator singleton, calls `orchestrator.explain()`, assembles `ExplainabilityResult`. |
| `explain_prediction_full` | `(text, predict_fn, model, tokenizer, use_lime, use_shap)` | Backward-compat wrapper: LIME + SHAP + bias/emotion + aggregation + consistency + metrics. |
| `explain_fast` | `(text, predict_fn)` | Minimal config: LIME only, no model required, no aggregation. |

**Dependencies:** `orchestrator`, `common_schema`  
**External:** none

---

### File: `orchestrator.py`

**Purpose**

Central coordinator. Runs every sub-explainer under `_run()` (safe isolation with timing), applies the FAITH-1 faithfulness gate on attention, and assembles the final explanation dict. The process-wide singleton `get_default_orchestrator()` is keyed by the SHA-256 hash of the config dataclass (via `json.dumps(asdict(config))`).

**Key Classes / Functions**

| Name | Description |
|------|-------------|
| `ExplainabilityConfig` | 17-field dataclass controlling which modules run. Key fields: `use_shap` (default False — too slow on CPU), `use_lime` (True), `use_attention_rollout` (True), `use_bias_emotion` (True), `use_graph_explainer` (True), `attention_faithfulness_threshold` (0.0 = disabled), `ig_steps` (8), `raise_on_majority_failure` (False), `aggregation_weights` (AggregationWeights). |
| `ExplainabilityOrchestrator.__init__` | Instantiates: `ExplanationCache`, `AttentionRollout`, `ExplanationAggregator`, `ExplanationConsistency`, `ExplanationMetrics`, `ExplanationMonitor`, `GraphExplainer`. All gated by config flags. |
| `ExplainabilityOrchestrator.explain` | Main pipeline method. Runs all enabled modules, applies FAITH-1 gate, builds aggregation, consistency, metrics, monitoring, and metadata. |
| `ExplainabilityOrchestrator.explain_fast` | LIME-only fast path. Caches base prediction before calling LIME (avoids extra model forward). |
| `_make_batch_predict_fn` | Wraps `predict_fn(text) → dict` into `(texts: List[str]) → List[dict]` for SHAP/LIME. Uses `predict_fn.batch_predict` when available (CRIT-12). |
| `_wrap_bias_ig` | Extracts `integrated_gradients` or `token_importance` from `explain_bias` output and wraps it in an `ExplanationOutput` so the aggregator and consistency stages can consume it (CRIT-8). |
| `_spearman_safe` | Nan-safe Spearman for FAITH-1 gate without scipy dependency. |
| `_compute_offsets` | Character-offset alignment between aggregated canonical tokens and original text for CRIT-11 ablation. Returns `None` if alignment breaks. |
| `get_default_orchestrator` | Thread-safe singleton factory. Config hash → cached `ExplainabilityOrchestrator`. |

**Dependencies:** all other explainability files, `src.graph.graph_explainer`  
**External:** `numpy`, `torch`, `threading`, `hashlib`

---

### File: `shap_explainer.py`

**Purpose**

SHAP-based local explanation of a single article's prediction. Uses `shap.maskers.Text` + `shap.Explainer` to compute token-level Shapley values, then calibrates and returns an `ExplanationOutput`.

**Key Functions**

| Function | Inputs | Outputs | Logic |
|----------|--------|---------|-------|
| `get_explainer(predict_fn)` | callable | `shap.Explainer` | LRU cache (max 8) keyed by stable predict_fn identity; builds `shap.maskers.Text` + `shap.Explainer` on miss. |
| `_get_shap_values(predict_fn, text)` | callable, str | `shap.Explanation` | Value cache (max 64) keyed by (predict_fn_key, sha1(text)); runs `explainer([text])` on miss. |
| `explain_text(predict_fn, text)` | callable, str | `ExplanationOutput` | Calls `_get_shap_values`, extracts `data[0]` (tokens) and `values[0]` (SHAP values), filters special tokens, calibrates via `calibrate_explanation`, returns `ExplanationOutput(method="shap")`. |
| `shap_predict_wrapper(texts, predict_fn)` | List[str], callable | `ndarray (N, 2)` | Wraps predict_fn into SHAP's expected `[p_real, p_fake]` format. Uses `batch_predict` when available. |
| `plot_explanation / save_explanation_html` | — | HTML file | Delegates to `shap.plots.text`. |

**SHAP value shape negotiation (`_process_shap_values`):**

| Shape | Meaning | Action |
|-------|---------|--------|
| 3D `(samples, seq_len, classes)` | Multi-class | Take `[:, :, -1]` (positive class) |
| 2D `(seq_len, 1)` | Binary single | Flatten `[:, 0]` |
| 2D `(seq_len, classes)` | Multi-class 2D | Take `[:, -1]` |
| 1D | Already correct | Pass through |

**Explainability technique:** SHAP (Shapley Additive Explanations) with `Text` masker (masks words by replacing with baseline mask string).  
**Dependencies:** `explanation_calibrator`, `utils_validation`, `common_schema`  
**External:** `shap` (optional — ImportError if absent), `numpy`

---

### File: `lime_explainer.py`

**Purpose**

LIME-based local explanation. Perturbs the input text (word removal), calls the model on each perturbation, fits a local linear regression to explain which words most affect the prediction.

**Key Functions**

| Function | Inputs | Outputs | Logic |
|----------|--------|---------|-------|
| `get_explainer(model_id)` | str | `LimeTextExplainer` | LRU cache (max 4) keyed by model_id. Creates `LimeTextExplainer(class_names=["Real", "Fake"])`. |
| `explain_prediction(predict_fn, text, num_features, num_samples)` | callable, str, int=8, int=25 | `ExplanationOutput` | Cache check (SHA-256 key) → `explainer.explain_instance()` → `exp.as_list()` → calibrate → `ExplanationOutput(method="lime")`. `num_samples=25` reduced from 256 (PERF-2). |
| `lime_predict_wrapper(texts, predict_fn)` | Sequence[str], callable | `ndarray (N, 2)` | Wraps predict_fn. Tries `batch_predict` → list call → per-text loop with 0.5 fallback on error. |
| `save_explanation_html` | — | HTML file | Calls `exp.save_to_file`. |

**Explanation cache:** bounded `OrderedDict` capped at 256 entries (key = `sha256(text|num_features|num_samples)`).

**Explainability technique:** LIME — local interpretable model-agnostic explanations. Fits ridge regression on perturbed neighborhood.  
**Dependencies:** `explanation_calibrator`, `common_schema`  
**External:** `lime.lime_text.LimeTextExplainer` (optional), `numpy`

---

### File: `attention_rollout.py`

**Purpose**

Propagates attention through all transformer layers using the rollout algorithm (Abnar & Zuidema, 2020). Returns per-token importance from the `[CLS]` token's perspective.

**Key Class: `AttentionRollout`**

| Method | Inputs | Outputs | Logic |
|--------|--------|---------|-------|
| `_validate_inputs` | attentions, tokens, sample_index, source_token_index | `seq_len` | Validates 4D tensors `(batch, heads, seq, seq)`, consistent batch/seq across layers. |
| `_aggregate_heads` | `(batch, heads, seq, seq)`, sample_index | `(seq, seq)` | `mean(dim=1)[sample_index]` |
| `_add_residual` | `(seq, seq)` | `(seq, seq)` | Add identity matrix → row-normalize |
| `_stack_add_residual_normalize` | List of `(seq, seq)` | `(layers, seq, seq)` | **PERF-4**: stack all layers → fused residual-add + normalize in 2 kernel launches (vs 24 per-layer calls). float16/bfloat16 upcast to float32. |
| `compute_rollout` | attentions, tokens, *, sample_index, source_token_index, mask_tokens, layer_weights, top_k | `ExplanationOutput` | Head-aggregate per layer → vectorised stack → residual → optional layer weights → matrix product all layers → `scores = rollout[source_token_index]` → calibrate → return. |

**Explainability technique:** Attention rollout — faithful attention-based attribution that accounts for residual connections and layer-wise propagation.  
**Dependencies:** `explanation_calibrator`, `common_schema`  
**External:** `torch`, `numpy`

---

### File: `attention_visualizer.py`

**Purpose**

High-level wrapper that combines attention extraction (model forward pass with `output_attentions=True`), aggregation, rollout, and visualization.

**Key Class: `AttentionVisualizer`**

| Method | Inputs | Outputs | Description |
|--------|--------|---------|-------------|
| `extract_attention` | `input_ids`, `attention_mask` | `{"attentions": List[Tensor]}` | Runs model forward with `output_attentions=True`. Handles `return_dict=True` and legacy tuple returns. |
| `aggregate_attention` | attentions, sample_index | `ndarray` | Stack all layers → mean over heads and layers → `(seq, seq)` heatmap. |
| `compute_rollout` | attentions, tokens | `ExplanationOutput` | Delegates to `AttentionRollout.compute_rollout`. |
| `plot_attention` | attention_matrix, tokens, *, title, save_path, normalize | figure | Matplotlib `imshow` heatmap with token labels on both axes. |
| `plot_token_importance` | tokens, scores, *, title, save_path | figure | Matplotlib bar chart of normalized scores. |
| `analyze` | input_ids, attention_mask, tokens | `{attention_matrix, rollout}` | Full pipeline: extract → aggregate → rollout. |

**External:** `torch`, `numpy`, `matplotlib` (lazy import GPU-5)

---

### File: `bias_explainer.py`

**Purpose**

Model-aware bias explainer that fuses three attribution signals (SHAP, Integrated Gradients, Attention Rollout) on the `MultiTaskTruthLensModel`. Routes through `model.encoder` + `model.heads[task]` — not `model(...).logits` which does not exist on the multitask wrapper (CRIT-1).

**Key Functions**

| Function | Description |
|----------|-------------|
| `_forward_logits(model, enc, task)` | Single forward pass returning task-head logits. Supports both multitask and vanilla HF models (CRIT-1). |
| `compute_shap(model, tokenizer, text, task)` | Cached `shap.Explainer` per `(id(tokenizer), task)` (PERF-3). Returns normalized attribution per token. |
| `compute_ig(model, tokenizer, text, target_idx, steps)` | **Real Riemann-sum IG** (FAITH-2). Interpolates `steps` embeddings between zero baseline and input, averages gradients along the path. Pads gradients on masked positions. |
| `compute_attention_rollout(model, tokenizer, text, task)` | Calls encoder with `output_attentions=True`, delegates to `AttentionRollout`. Slices off `[CLS]`/`[SEP]` to align with `tokenizer.tokenize(text)`. |
| `fuse_methods(shap_vals, ig_vals, attn_vals)` | Weighted fusion: SHAP=0.4, IG=0.3, Attention=0.3 (weights zeroed and renormalized when method unavailable). Returns `(fused, weights)`. |
| `explain_bias(model, tokenizer, text, task, use_shap, ig_steps)` | Main API. `use_shap=False` by default (too slow on CPU). `ig_steps=0` skips IG entirely. Returns dict with `token_importance`, `integrated_gradients`, `biased_tokens`, `sentence_bias_scores`, `attention_scores`, `bias_heatmap`, `bias_intensity`. |

**Explainability techniques:** SHAP (model-specific, tokenizer masker), Integrated Gradients (gradient × path integral), Attention Rollout  
**Dependencies:** `token_alignment`, `utils_validation`, `attention_rollout`  
**External:** `shap` (optional), `torch`, `numpy`

---

### File: `emotion_explainer.py`

**Purpose**

Emotion explainer that explicitly separates **heuristic** (lexicon-based, word-level) from **faithful** (gradient-based, subword-level) signals to prevent the CRIT-2 tokenization alignment bug where lexicon and gradient arrays were silently mixed at different granularities.

**Key Functions**

| Function | Description |
|----------|-------------|
| `compute_lexicon(tokens)` | Assigns `1.0` per emotion-lexicon word (from `EMOTION_TERMS`), `+0.5` per intensifier. Word-level. |
| `compute_gradients(model, tokenizer, text, task)` | Gradient × input on `model.encoder.embeddings`. Returns `(subword_tokens, scores)` pair — the tokens are subword (BPE), not word-level (CRIT-2/FAITH-5). |
| `fuse(lexicon, gradients)` | **Returns lexicon signal only** (CRIT-2: dropped 0.6×lex + 0.4×grad cross-tokenization mixing). `gradients` kwarg kept for API compatibility. |
| `emotion_distribution(tokens)` | Fraction of detected emotion-category words per emotion class. |
| `explain_emotion(text, model, tokenizer)` | Main API. Returns `EmotionExplanation` dict with `lexicon_intensity`, `gradient_importance=[]` (deprecated), `fused_importance` (lexicon), `sentence_scores`, `emotion_distribution`, `intensity_score`, `faithful=False`, `model_attribution={tokens, importance, faithful=True, token_space="subword"}`. |

**Explainability techniques:** Lexicon scoring (heuristic, `faithful=False`) + gradient×input (faithful, `faithful=True`)  
**Dependencies:** `src.features.emotion.emotion_schema.EMOTION_TERMS`  
**External:** `torch`, `numpy`

---

### File: `propaganda_explainer.py`

**Purpose**

Pattern-matching detection of eight classical propaganda techniques using a fixed lexicon. Explicitly marked `faithful=False` (CRIT-9) — this is a rule-based heuristic, not a model attribution.

**Techniques detected:**

| Technique | Weight | Examples |
|-----------|--------|---------|
| `name_calling` | 1.0 | terrorist, criminal, extremist |
| `glittering_generalities` | 0.6 | freedom, democracy, patriot |
| `fear_appeal` | 0.9 | danger, crisis, catastrophe |
| `loaded_language` | 1.0 | regime, hoax, conspiracy, rigged |
| `false_dilemma` | 0.7 | either, never, always, inevitable |
| `appeal_to_authority` | 0.5 | experts say, scientists confirm |
| `bandwagon` | 0.6 | everyone, majority, all Americans |
| `repetition` | 0.8 | tokens appearing > 2 times (boosted proportionally) |

**Key Functions**

| Function | Description |
|----------|-------------|
| `explain_propaganda(text, top_k)` | `_score_tokens` (single-word) + `_apply_phrase_scores` (multi-word) → calibrate → `ExplanationOutput(method="propaganda", faithful=False)`. Returns empty output if no patterns match. |
| `detect_techniques(text)` | Returns `{technique: [matched_tokens]}` dict without full ExplanationOutput. |

**Explainability technique:** Lexicon pattern matching (heuristic)  
**Dependencies:** `common_schema`, `explanation_calibrator`  
**External:** `numpy`

---

### File: `token_alignment.py`

**Purpose**

Merges subword tokens (WordPiece `##`, SentencePiece `▁`, BPE `Ġ`) into word-level tokens for human-readable output. Aggregates per-subtoken scores into a single word score.

**Key Function: `align_tokens`**

```python
align_tokens(
    tokens: Sequence[str],
    scores: Sequence[float],
    tokenizer_type: str = "wordpiece",  # "wordpiece" | "sentencepiece" | "bpe"
    aggregation: str = "mean",          # "mean" | "sum" | "max"
    normalize: bool = False,
    clip: bool = False,
    max_tokens: int | None = None,
    return_structured: bool = False,
    return_variance: bool = False,
) -> Tuple[List[str], List[float]] | Dict
```

| Tokenizer | Merge rule |
|-----------|-----------|
| WordPiece | Tokens starting with `##` are continuations; flush on new root token |
| SentencePiece | Tokens starting with `▁` (U+2581) begin new words |
| BPE (RoBERTa/GPT-2) | Tokens starting with `Ġ` (U+0120) begin new words |

**Aggregation:** `mean` (default), `sum`, or `max` by absolute value. `return_variance=True` returns per-word score variance across constituent subtokens.

**External:** `numpy`

---

### File: `explanation_aggregator.py`

**Purpose**

Fuses per-method `ExplanationOutput` objects into a single `AggregatedExplanation`. Implements four critical correctness fixes (CRIT-3/4/9/PERF-5).

**Key Class: `ExplanationAggregator`**

| Method | Description |
|--------|-------------|
| `__init__(weights, include_heuristic, config_path)` | Normalizes `AggregationWeights` to sum 1. Optionally loads weights from YAML (CFG-3). `include_heuristic=False` gates propaganda/emotion out of fusion. |
| `aggregate(shap, integrated_gradients, attention, lime, graph_explanation)` | Full pipeline: faithfulness filter → canonical token selection → `[methods × tokens]` matrix build → vectorised `weighted * confidence` sum (PERF-5) → graph contribution → normalize → per-token confidence = `1 - std` → agreement score → `AggregatedExplanation`. |

**Default fusion weights:**

| Method | Default weight |
|--------|---------------|
| SHAP | 0.35 |
| Integrated Gradients | 0.25 |
| Attention Rollout | 0.20 |
| LIME | 0.10 |
| Graph | 0.10 |

**Token alignment contract (CRIT-3/4):**
- Canonical token sequence: first non-empty source in `shap → ig → attn → lime` order
- Positional alignment when `len(src_tokens) == len(canonical)` (preserves duplicates)
- First-occurrence name lookup otherwise (best-effort for mismatched tokenisations)

**Dependencies:** `explanation_consistency`, `common_schema`  
**External:** `numpy`, `yaml` (optional, for config loading)

---

### File: `explanation_calibrator.py`

**Purpose**

Normalizes raw explanation scores (which may be signed SHAP values, positive attention scores, or lexicon counts) into a common [0, 1]-probability-like scale, and derives confidence and entropy metrics.

**Key Functions**

| Function | Description |
|----------|-------------|
| `normalize_scores(scores)` | L1 normalization: `abs(arr) / sum(abs(arr))`. Returns zeros on empty or all-zero input. |
| `compute_entropy(probs)` | Shannon entropy `H = -sum(p * log(p))`. |
| `compute_confidence(probs)` | `1 - H / log(512)` — **fixed reference entropy** `log(512)` regardless of token count (SCALE-3). A 5-token peaked explanation and a 100-token peaked explanation get the same confidence when equally concentrated. |
| `calibrate_by_method(scores, method)` | **FAITH-3**: all methods now receive identical L1 normalization. Removed the previous per-method `power 0.8 / 1.2` shaping that had no theoretical basis and changed token ranking irreversibly. |
| `calibrate_explanation(scores, method)` | Main entry point. Returns `{scores: ndarray, confidence: float, entropy: float}`. |

**External:** `numpy`

---

### File: `explanation_consistency.py`

**Purpose**

Measures agreement between explanation methods by computing pairwise correlation metrics across their shared token vocabularies.

**Key Class: `ExplanationConsistency`**

| Method | Description |
|--------|-------------|
| `_pearson(a, b)` | `np.corrcoef(a, b)[0, 1]`. Returns 0.0 if std < 1e-12. |
| `_spearman(a, b)` | `corrcoef(rank(a), rank(b))` using `argsort(argsort(x))` for true ranks (CRIT-10). Returns 0.0 on degenerate input. |
| `_cosine(a, b)` | `dot(a,b) / (norm(a)*norm(b) + EPS)`. |
| `_compare(a, b, conf_a, conf_b)` | Computes all three metrics on shared tokens. Confidence-weights by `min(conf_a, conf_b)`. |
| `_token_consistency(sources)` | **REC-4**: vectorised `np.nanstd` over `[n_sources × n_tokens]` matrix. Per-token consistency = `clip(1 - std, 0, 1)`. NaN for tokens present in only one source. |
| `compute(shap, ig, attention, lime)` | Returns dict with all pairwise metrics + `overall_agreement` + `token_agreement_mean`. Returns `{}` when fewer than 2 sources available. |

**Pairwise metric keys** (example when shap + lime available):
```
shap_vs_lime_pearson, shap_vs_lime_spearman, shap_vs_lime_cosine,
overall_agreement, token_agreement_mean
```

**External:** `numpy`

---

### File: `explanation_metrics.py`

**Purpose**

Faithfulness evaluation: quantifies how well the explanation reflects the model's actual decision logic by ablating tokens and measuring prediction changes.

**Key Class: `ExplanationMetrics`**

| Method | What it measures | How |
|--------|-----------------|-----|
| `faithfulness` | Correlation between importance scores and prediction drops per removed token | Pearson(scores, base - preds_per_removed_token) |
| `comprehensiveness` | How much the prediction drops when top-k tokens are removed | base - pred(text_without_top_k) |
| `sufficiency` | How well top-k tokens alone preserve the prediction | base - pred(top_k_tokens_only) |
| `deletion_score` | Average prediction drop as tokens deleted highest-to-lowest | base - mean(pred at each deletion step) |
| `insertion_score` | Cumulative prediction gain as tokens revealed highest-first | trapz(preds at each insertion step) |
| `evaluate` | Runs all 5 metrics, confidence-weights, normalizes to [0,1] via `(v+1)/2`, returns `overall_score` | — |

**CRIT-11 ablation paths:**

| Context | Ablation method |
|---------|----------------|
| `text` + `offsets` available | `_ablate_offsets`: replaces character spans in original text (character-accurate) |
| Fallback | `" ".join([t for j,t in enumerate(tokens) if j != i])` (legacy, may corrupt subwords) |

**REC-3:** `evaluate()` computes `base_proba` once if not supplied, then passes it to all 5 sub-metrics, collapsing 5 redundant model forwards into 1.

**External:** `numpy`

---

### File: `explanation_monitor.py`

**Purpose**

Lightweight production monitoring of explanation quality. Tracks running statistics over the last `max_history=500` importance score distributions.

**Key Class: `ExplanationMonitor`**

| Method | Description |
|--------|-------------|
| `update(scores)` | Normalizes scores (L1), appends to bounded history. Evicts oldest when full. |
| `mean / std / min / max` | Aggregate statistics over concatenated history. |
| `drift()` | L1 distance between last two history entries — detects explanation distribution shift. |
| `summary()` | Returns `{mean, std, min, max, drift, history_size}` |
| `reset()` | Clears history for testing or after model updates. |

**External:** `numpy`

---

### File: `explanation_cache.py`

**Purpose**

Two-level (memory + disk) explanation cache with TTL, LRU eviction, zlib compression, and versioning.

**Key Class: `ExplanationCache`**

| Method | Description |
|--------|-------------|
| `_make_key(text, model_version, method)` | `sha256(text|model_version|method)` — method-specific keys prevent cross-method cache collisions |
| `get(text, model_version, method)` | Memory LRU → disk read → deserialize → TTL check → version check → return |
| `set(text, data, model_version, method)` | Write to memory (evict to max_size) + write compressed bytes to disk |
| `stats()` | Returns `{hits, misses, hit_rate}` |
| `clear_memory / clear_disk` | Eviction utilities |

**Serialization:** `json.dumps` → `zlib.compress` (when `enable_compression=True`). Version tag `"v2"` invalidates stale disk entries after schema changes.

**External:** `hashlib`, `zlib`, `json`, `pathlib`, `threading`

---

### File: `explanation_report_generator.py`

**Purpose**

Writes per-article explanation reports as JSON and HTML. The HTML report includes token-level importance highlighting with red-intensity color coding.

**Key Class: `ExplanationReportGenerator`**

| Method | Description |
|--------|-------------|
| `save_json(article_id, explanation)` | Writes `{article_id, generated_at, version="v3", explanation}` to `{output_dir}/{safe_id}.json` |
| `save_html(article_id, explanation)` | Writes 9-section HTML: Prediction, Scores, Risks, Token Importance (highlighted), Explainability Metrics, Method Contributions, Confidence, Entropy, Monitoring |
| `_highlight_tokens(tokens, scores)` | `<span style="background:rgba(255,0,0,{s/max})">token</span>` for each token — opacity maps to relative importance |
| `generate(article_id, explanation, *, save_json, save_html)` | Calls both save functions, returns `{json: Path, html: Path}` |

**Output format:** `reports/explanations/{safe_article_id}.json` and `.html`

---

### File: `explanation_visualizer.py`

**Purpose**

Static (matplotlib) and interactive (Plotly) visualization of token importance.

**Key Class: `ExplanationVisualizer`**

| Method | Output | Description |
|--------|--------|-------------|
| `plot_token_heatmap` | PNG | 1×N color matrix with token labels on x-axis |
| `plot_importance_bar` | PNG | Horizontal bar chart, top-k tokens sorted by importance |
| `plot_multi_method_overlay` | PNG | Line plot overlaying SHAP, LIME, IG, Attention on same token axis |
| `plot_interactive` | Browser | Plotly scatter with hover — one trace per method |
| `visualize_aggregated` | multiple | Runs heatmap + bar + multi-method overlay in one call |

All matplotlib plots use lazy imports (`import matplotlib.pyplot as plt` inside method body) to avoid GPU process import cost (GPU-5).

**External:** `matplotlib`, `plotly` (optional)

---

### File: `model_explainer.py`

**Purpose**

Backward-compatibility wrapper. Contains `explain_prediction_full()` and `explain_fast()` that instantiate a fresh `ExplainabilityOrchestrator` per call (not using the process-level singleton). Present for callers that still use the old interface directly.

> **Note:** New code should use `explainability_pipeline.run_explainability_pipeline()` which routes through the singleton orchestrator.

---

### File: `utils_validation.py`

**Purpose**

Shared input validator used by SHAP, LIME, attention rollout, consistency, and metrics modules.

**Key Function: `validate_tokens_scores`**

```python
validate_tokens_scores(
    tokens, scores,
    enforce_range=False,   # raise if score outside [0, 1]
    normalized=False,      # raise if sum ≠ 1
    allow_duplicates=True, # if False: merge duplicates by summing scores
    auto_fix=False,        # coerce non-finite / out-of-range to 0.0
    return_fixed=False,    # return (tokens, scores) after fixes
)
```

Checks: type correctness, length match, all strings, all numeric, all finite, optional range `[0,1]`, optional sum-to-1, optional low-variance signal warning.

**External:** `numpy`, `math`

---

## 5. Explanation Types

| Type | Description | Example | Faithful? |
|------|-------------|---------|-----------|
| **Global** | Overall model behavior across the entire article | Feature importance ranking over all tokens, propaganda technique distribution | Partial |
| **Local — SHAP** | Token-level Shapley value for a single prediction | "The word *radical* pushed fake probability from 0.62 → 0.74" | Yes |
| **Local — LIME** | Token-level linear approximation for a single prediction | "Removing *invasion* reduced fake probability by 0.11" | Approximate |
| **Local — IG** | Gradient-based token attribution via path integration | Smooth gradient attribution along interpolation path from zero to input | Yes |
| **Local — Attention** | Layer-propagated attention weight per token | "[CLS] attends most to *crisis* across all 12 layers" | Conditional (FAITH-1 gated) |
| **Heuristic — Propaganda** | Pattern-matched propaganda technique per token | *terrorist*: name-calling (weight=1.0) | No (`faithful=False`) |
| **Heuristic — Emotion** | Lexicon-based emotional intensity per word | *incredibly*: intensifier (+0.5) | No (`faithful=False`) |
| **Aggregated** | Weighted combination of all faithful explanations | Final importance vector from 0.35×SHAP + 0.25×IG + 0.20×attention + 0.10×LIME + 0.10×graph | Best-effort |

---

## 6. Feature Importance Interpretation

### How scores are calculated

1. Each method produces a raw score vector over its tokens (SHAP values: signed; attention: positive; LIME: signed coefficients; lexicon: positive counts)
2. `calibrate_explanation()` applies L1 normalization → all scores become non-negative, summing to 1.0
3. The aggregator builds a `[methods × tokens]` matrix, multiplies each row by `weight × confidence`, sums across methods, normalizes
4. Final `final_token_importance[i]` = relative contribution of token `i` to the prediction

### Positive vs negative impact

- **Raw SHAP values** can be negative (token pushes toward "real") or positive (pushes toward "fake")
- After `calibrate_explanation`, scores are **absolute-valued** and L1-normalized — the directionality is lost but the magnitude is preserved
- For signed-value analysis, use `ExplanationOutput.raw` (available in LIME, contains `(token, signed_score)` list)

### Ranking

`ExplanationOutput.structured` is returned in **token-occurrence order**, not ranked. To get top-k:
```python
ranked = sorted(zip(out.tokens, out.importance), key=lambda x: x[1], reverse=True)
top_3 = ranked[:3]
```

---

## 7. Output Artifacts

### 7.1 `ExplainabilityResult` (in-memory / JSON via API)

```json
{
  "prediction": {"fake_probability": 0.87, "prediction": "FAKE", "confidence": 0.87},
  "shap_explanation": {
    "method": "shap",
    "tokens": ["breaking", "news", "radical", "attack"],
    "importance": [0.05, 0.02, 0.41, 0.52],
    "structured": [{"token": "radical", "importance": 0.41}, ...],
    "confidence": 0.81,
    "entropy": 1.23
  },
  "lime_explanation": {...},
  "attention_explanation": {...},
  "bias_explanation": {
    "token_importance": [{"token": "radical", "importance": 0.41}],
    "integrated_gradients": [...],
    "biased_tokens": ["radical", "attack"],
    "bias_intensity": 0.34,
    "bias_heatmap": [0.05, 0.02, 0.41, 0.52]
  },
  "emotion_explanation": {
    "lexicon_intensity": [...],
    "emotion_distribution": {"fear": 0.6, "anger": 0.4},
    "intensity_score": 0.21,
    "faithful": false,
    "model_attribution": {"tokens": [...], "importance": [...], "faithful": true}
  },
  "aggregated_explanation": {
    "tokens": ["breaking", "news", "radical", "attack"],
    "final_token_importance": [0.04, 0.01, 0.53, 0.42],
    "method_weights": {"shap": 0.35, "ig": 0.25, "attn": 0.20, "lime": 0.10, "graph": 0.10},
    "confidence_score": 0.77,
    "agreement_score": 0.68
  },
  "consistency_metrics": {
    "shap_vs_lime_spearman": 0.71,
    "overall_agreement": 0.69,
    "token_agreement_mean": 0.73
  },
  "explanation_metrics": {
    "faithfulness": 0.63,
    "comprehensiveness": 0.18,
    "sufficiency": 0.14,
    "deletion_score": 0.09,
    "insertion_score": 0.42,
    "overall_score": 0.74
  },
  "monitoring": {"mean": 0.22, "std": 0.08, "drift": 0.03, "history_size": 42},
  "explanation_quality_score": 0.74,
  "module_failures": [],
  "metadata": {"pipeline_version": "v5", "latency_ms": {"lime": 140, "aggregation": 3}}
}
```

### 7.2 HTML report (`ExplanationReportGenerator`)

- Token importance highlighted with red-intensity color coding
- Sections: Prediction, Scores, Risks, Token Importance, Metrics, Method Contributions, Confidence, Entropy, Monitoring
- Saved to `reports/explanations/{article_id}.html`

### 7.3 SHAP HTML (`shap_explainer.save_explanation_html`)

- `shap.plots.text` interactive force-plot style highlight
- Saved to `reports/shap.html`

### 7.4 LIME HTML (`lime_explainer.save_explanation_html`)

- `LimeExplanation.save_to_file` — browser-viewable explanation with class probabilities, word contribution bars
- Saved to `reports/lime_explanation.html`

### 7.5 Plot images (`ExplanationVisualizer`)

- `{prefix}_heatmap.png` — token importance as a 1-row color matrix
- `{prefix}_bar.png` — horizontal bar chart (top 20 tokens)
- `{prefix}_overlay.png` — line overlay of all methods on same token axis

---

## 8. Model Compatibility

### Which models are supported

| Model type | SHAP | LIME | IG | Attention Rollout |
|-----------|------|------|----|------------------|
| TruthLens `MultiTaskTruthLensModel` (roberta-base) | Yes (via tokenizer masker) | Yes | Yes (multitask path) | Yes |
| Any HF model with `.logits` output | Yes | Yes | Yes (fallback path) | Yes (if `output_attentions=True`) |
| Any model exposing `predict_fn(text) → {fake_probability}` | Yes | Yes | No | No |
| Sklearn / tree-based | No (not supported) | Yes | No | No |

### Limitations per method

| Method | Limitations |
|--------|------------|
| SHAP | Requires hundreds of model forwards per article. On CPU, ~5–30 seconds. Disabled by default (`use_shap=False`). |
| LIME | `num_samples=25` (reduced from 256 for speed). Lower samples = noisier attributions. |
| IG | Requires access to model embedding layer. `ig_steps=8` is a speed/accuracy tradeoff. |
| Attention Rollout | Requires `output_attentions=True` support. May not correlate with predictions (gated by FAITH-1). |
| Propaganda | Lexicon-only — cannot detect novel propaganda that isn't in the pattern list. |
| Emotion | Lexicon signal is heuristic. Gradient attribution requires model + tokenizer. |

---

## 9. Config Integration

All explainability behavior is controlled by `ExplainabilityConfig`:

```python
@dataclass
class ExplainabilityConfig:
    enabled: bool = True
    use_lime: bool = True
    use_shap: bool = False              # off by default (CPU cost)
    use_attention_rollout: bool = True
    use_bias_emotion: bool = True
    use_propaganda_explainer: bool = False  # off by default (heuristic)
    use_aggregation: bool = True
    use_consistency: bool = True
    use_explanation_metrics: bool = True
    use_graph_explainer: bool = True

    aggregator_include_heuristic: bool = False  # include propaganda in fusion?
    cache_enabled: bool = True
    cache_max_size: int = 128
    cache_dir: Optional[str] = None

    aggregation_weights: AggregationWeights = ...  # SHAP=0.35, IG=0.25, Att=0.20, LIME=0.10, Graph=0.10

    attention_faithfulness_threshold: float = 0.0  # 0 = no gate; 0.3 = drop if |Spearman(attn,IG)| < 0.3
    raise_on_majority_failure: bool = False
    ig_steps: int = 8                  # 0 = skip IG entirely (fastest mode)
```

**YAML config for aggregation weights (CFG-3):**
```yaml
explainability:
  aggregation_weights:
    shap: 0.35
    integrated_gradients: 0.25
    attention: 0.20
    lime: 0.10
    graph: 0.10
```

Load via `ExplanationAggregator(config_path="config/config.yaml")`.

---

## 10. Performance and Efficiency

### Computational cost per article

| Method | CPU cost | GPU cost | Notes |
|--------|----------|----------|-------|
| LIME (25 samples) | ~120–200 ms | ~30 ms | `num_samples=25` (reduced from 256) |
| SHAP | ~5–30 s | ~1–3 s | Disabled by default |
| IG (8 steps) | ~200–400 ms | ~50 ms | Linear in `ig_steps` |
| Attention Rollout | ~20 ms | ~10 ms | Cheap — no extra model forward |
| Aggregation | < 5 ms | — | Vectorised matrix ops |
| Consistency | < 5 ms | — | Pure numpy |
| Metrics (5 per article, batched) | ~1–5 s | ~200 ms | 5 ablation sweeps |

### Efficiency fixes implemented

| Fix | Impact |
|-----|--------|
| PERF-2: LIME `num_samples` 256 → 25 | ~10× speedup |
| PERF-3: SHAP explainer cached per `(tokenizer, task)` | Avoids masker rebuild per article |
| PERF-4: Attention rollout vectorised (stack + fused residual) | 24 kernel launches → 2 |
| PERF-5: Aggregator vectorised `[methods × tokens]` matrix | Eliminates per-token Python loop |
| PERF-6: Orchestrator singleton per config hash | No re-instantiation of GraphExplainer per article |
| CRIT-12: Batch predict via `predict_fn.batch_predict` | Single GPU call for all SHAP/LIME perturbations |
| REC-3: Base prediction computed once, forwarded | 5 redundant model forwards → 1 |
| GPU-5: Matplotlib lazy imports | Avoids matplotlib CUDA context at startup |

### Approximation strategy

- LIME uses a linear local approximation (not exact)
- IG uses Riemann sum with 8 steps (not exact integration — use 20–50 steps for research)
- SHAP uses the `shap.Explainer` partition-based estimator (approximate for large token counts)
- Attention rollout is exact given the model's actual attention weights

---

## 11. Validation of Explanations

### Input validation

Every explainer calls `validate_tokens_scores(tokens, scores)` before returning. It checks:
- Type correctness (str tokens, numeric scores)
- Length match
- All finite values
- Near-zero variance warning (low-signal explanation)

`auto_fix=True` mode coerces non-finite values to 0.0 instead of raising.

### Cross-method consistency (`ExplanationConsistency`)

Cross-method Pearson, Spearman, cosine correlation. Reported in `consistency_metrics`. An `overall_agreement` > 0.6 indicates the methods agree on which tokens are important.

### Faithfulness gate (FAITH-1)

When `attention_faithfulness_threshold > 0`, attention rollout is dropped from aggregation if its Spearman correlation with IG falls below the threshold. This prevents unreliable attention patterns from diluting the aggregated signal.

### Faithfulness metrics (`ExplanationMetrics`)

Measures whether token importance actually reflects the model's decision:
- **Faithfulness > 0.5**: Strong — removing high-importance tokens substantially changes prediction
- **Comprehensiveness > 0.1**: Removing top-5 tokens drops prediction
- **Sufficiency ≈ 0**: Top-5 tokens alone preserve most of the prediction

### Schema synchronization

`ExplanationOutput` validator enforces that `importance[i] == structured[i].importance` (tolerance 1e-6). This prevents the calibrator and the flat list from drifting.

### `faithful` flag (CRIT-9)

Every `ExplanationOutput` carries a `faithful: bool` flag:
- `True` (default): SHAP, LIME, IG, Attention — derived from model computations
- `False`: Propaganda, Emotion-lexicon — derived from rule-based heuristics

The aggregator only fuses `faithful=True` sources by default (`include_heuristic=False`).

---

## 12. Bias and Fairness Insights

### How explanations help detect bias

1. **Bias explainer**: `biased_tokens` = tokens with fused importance > 0.05. High fused importance on identity-related terms (*radical*, *invasion*, *regime*) suggests the model latches onto loaded language rather than factual content.

2. **Propaganda explainer**: Surfaces specific loaded-language and fear-appeal tokens that may disproportionately affect predictions on articles about certain groups or topics.

3. **Emotion distribution**: `emotion_distribution` shows which emotional categories are activated. An article about a marginalized group triggering high `fear` or `anger` emotion class proportions warrants review.

4. **Cross-task attention**: When the same article has high `ideology` and `bias` task scores, the bias explainer can surface which tokens drove both — revealing ideological framing driving the fake-news classification.

### Identifying sensitive feature influence

- Pass `token_importance` output to `src/evaluation/fairness.py` for group-level auditing
- Compare `aggregated_explanation.final_token_importance` across demographic subgroups in your evaluation dataset

### Ethical considerations

- The propaganda and emotion explainers use **fixed English lexicons** — performance degrades on non-English text or domain-specific jargon
- SHAP explanations explain a **local linear approximation** — high importance on a token does not prove causation
- Explanations should be presented alongside confidence scores, not as definitive verdicts

---

## 13. Extensibility Guide

### Adding a new explainability method

1. **Create your explainer** (e.g. `src/explainability/my_explainer.py`):
   ```python
   from src.explainability.common_schema import ExplanationOutput, TokenImportance
   from src.explainability.explanation_calibrator import calibrate_explanation

   def explain_my_method(predict_fn, text) -> ExplanationOutput:
       tokens = ...   # list of str
       raw_scores = ...  # list of float
       cal = calibrate_explanation(raw_scores, method="custom")
       structured = [TokenImportance(token=t, importance=float(s))
                     for t, s in zip(tokens, cal["scores"])]
       return ExplanationOutput(
           method="custom",
           tokens=tokens,
           importance=cal["scores"].tolist(),
           structured=structured,
           confidence=cal["confidence"],
           entropy=cal["entropy"],
           faithful=True,  # or False if heuristic
       )
   ```

2. **Wire it into `ExplainabilityConfig`**: add a `use_my_method: bool = False` field

3. **Call it in `orchestrator.py`** inside `explain()`:
   ```python
   if self.config.use_my_method:
       my_out, t, ok = self._run("my_method", lambda: explain_my_method(predict_fn, text))
       _record("my_method", ok, t)
       explanation["my_method_explanation"] = my_out
   ```

4. **Add it to `ExplainabilityResult`** in `common_schema.py`:
   ```python
   my_method_explanation: Optional[Any] = None
   ```

5. **Add it to the aggregator** (if faithful):
   - Add a weight field to `AggregationWeights`
   - Pass it as a kwarg to `ExplanationAggregator.aggregate()`
   - Extend the `method_names` list inside `aggregate`

### Adding a new propaganda pattern

Add entries to `PROPAGANDA_PATTERNS` dict in `propaganda_explainer.py`:
```python
"whataboutism": ["but what about", "you also", "they do it too"],
```
And a corresponding weight in `TECHNIQUE_WEIGHTS`.

### Tuning aggregation weights

Pass a custom `AggregationWeights` to `ExplainabilityConfig`:
```python
from src.explainability.explanation_aggregator import AggregationWeights
config = ExplainabilityConfig(
    aggregation_weights=AggregationWeights(shap=0.5, lime=0.5, attention=0.0, integrated_gradients=0.0, graph=0.0)
)
```

---

## 14. Common Pitfalls and Risks

| Pitfall | Cause | Mitigation |
|---------|-------|-----------|
| Inconsistent importance between methods | SHAP and LIME use different perturbation strategies | Check `consistency_metrics.overall_agreement`; < 0.4 means methods disagree |
| SHAP returns all-zero scores | Single-word article or all tokens masked | Check `ExplanationOutput.tokens` is non-empty before displaying |
| Attention scores don't match LIME/SHAP | Attention and prediction don't always correlate in transformers | Use FAITH-1 gate (`attention_faithfulness_threshold=0.3`) to auto-drop |
| High `fake_probability` on neutral article | Loaded lexicon in propaganda patterns triggering high scores | `propaganda_explanation.faithful=False` — treat as signal, not verdict |
| Slow response from `/explain` endpoint | SHAP enabled, CPU inference | Set `use_shap=False` in config (already the default) |
| Importance scores don't sum to 1 | Pre-calibration values passed to downstream code | Always use `ExplanationOutput.importance` (post-calibration), not raw model outputs |
| Correlated features mislead importance | High SHAP on "the" in a biased model | Check `explanation_metrics.faithfulness` — low value = tokens don't actually affect prediction |
| Cross-tokenization alignment errors | Mixing subword IG with word-level lexicon scores | `emotion_explainer.fuse()` now returns only the lexicon signal; access gradient separately via `model_attribution` |
| Cache staleness after model update | Old explanations served from disk cache | Use `model_version` kwarg in `ExplanationCache.get/set`, or call `clear_disk()` after deployment |
| Over-trusting explanation quality score | `overall_score` is an average of 5 ablation metrics, itself an approximation | Report alongside `module_failures` — if most modules failed, quality score is unreliable |

---

## 15. Example Usage

### Minimal — LIME only (fast mode)

```python
from src.explainability.explainability_pipeline import explain_fast

def my_predict_fn(text):
    return {"fake_probability": 0.87, "prediction": "FAKE", "confidence": 0.87}

result = explain_fast("Breaking news: Radical extremists attack the nation!", my_predict_fn)

# Top feature from LIME
lime = result.lime_explanation
ranked = sorted(zip(lime.tokens, lime.importance), key=lambda x: x[1], reverse=True)
print(f"Top token: {ranked[0][0]!r} (importance={ranked[0][1]:.3f})")
# → Top token: 'extremists' (importance=0.412)
```

### Full pipeline with model

```python
from src.explainability.explainability_pipeline import run_explainability_pipeline, ExplainabilityConfig

config = ExplainabilityConfig(
    use_lime=True,
    use_shap=False,          # too slow on CPU
    use_bias_emotion=True,
    use_attention_rollout=True,
    use_aggregation=True,
    use_consistency=True,
    use_explanation_metrics=True,
    attention_faithfulness_threshold=0.3,  # gate attention on IG correlation
)

result = run_explainability_pipeline(
    text="Scientists confirm new radical cure for pandemic.",
    predict_fn=my_predict_fn,
    model=model,
    tokenizer=tokenizer,
    config=config,
)

# Aggregated top-3 tokens
agg = result.aggregated_explanation
ranked = sorted(zip(agg.tokens, agg.final_token_importance), key=lambda x: x[1], reverse=True)
print("Top 3 tokens influencing prediction:")
for token, score in ranked[:3]:
    print(f"  {token!r}: {score:.3f}")

# Explanation quality
print(f"Explanation quality: {result.explanation_quality_score:.2f}")
print(f"Method agreement: {result.consistency_metrics.get('overall_agreement', 0):.2f}")
print(f"Failed modules: {result.module_failures}")
```

### Generate HTML report

```python
from src.explainability.explanation_report_generator import ExplanationReportGenerator

gen = ExplanationReportGenerator(output_dir="reports/explanations")
paths = gen.generate(
    article_id="article_001",
    explanation=result.model_dump(),
    save_json=True,
    save_html=True,
)
print(f"Report saved: {paths['html']}")
```

### Visualize multi-method overlay

```python
from src.explainability.explanation_visualizer import ExplanationVisualizer

viz = ExplanationVisualizer()
viz.visualize_aggregated(
    aggregated_output={
        "tokens": agg.tokens,
        "final_token_importance": agg.final_token_importance,
    },
    method_outputs={
        "lime": result.lime_explanation.importance,
        "attention": result.attention_explanation.importance,
    },
    save_prefix="reports/article_001",
)
```

---

## 16. Simple Explanation for Non-Technical Reviewers

### What does TruthLens actually do?

When you give TruthLens an article, it reads every word and decides whether the article is likely real or fake news. It gives you a probability — for example, "87% likely fake."

### But why did it say that?

That's where the explainability system comes in. Instead of just giving you a number, TruthLens also highlights **which specific words made it suspicious**.

Imagine the AI is like a detective. After reading a news article, it doesn't just say "this looks like a lie" — it also points to the evidence:

> "I'm 87% sure this is fake news, and here's why: the word **'extremists'** was the biggest red flag (importance 0.41), followed by **'radical'** (0.37), and **'attack'** (0.22). These three words together pushed my score strongly toward 'fake.'"

### How do we know the AI is telling the truth about its reasoning?

The system doesn't just ask the AI "why did you decide this?" It **tests** the reasoning by running an experiment:

1. It removes the word "extremists" from the article
2. It re-checks the AI's score — if the score drops significantly, that proves the word truly mattered
3. It does this for every highlighted word

This is called a **faithfulness test**, and it's how we verify that the highlighted words actually drove the decision rather than being invented explanations.

### What about the different colored words?

The explainability system uses three independent methods (SHAP, LIME, and Integrated Gradients) that each highlight important words differently. If all three agree that "extremists" is important, that's a very trustworthy signal. We report an **agreement score** (0 to 1) — above 0.6 means the methods are consistent.

### What does "biased tokens" mean in the bias analysis?

This shows words that the AI found particularly loaded — terms that tend to appear in propaganda or biased reporting, like *"regime," "invasion," "threat."* Finding these words doesn't prove the article is fake, but it tells you the model is reacting to emotionally charged language rather than objective facts.

### The one-line pitch

> "Our model is not a black box — for every decision it makes, we can tell you exactly which words influenced it, prove that those words actually mattered by testing what happens when we remove them, and show that three independent explanation methods all agree."
