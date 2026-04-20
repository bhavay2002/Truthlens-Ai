# TruthLens — Complete Deep Audit Report

**Auditor:** Senior ML Systems Engineer  
**Date:** 2026-04-20  
**Scope:** Full repository — all folders, all files  

---

## 🔴 Critical Issues (Must Fix)

---

### CRIT-1 | `models/inference/predictor.py` — Broken Double-Checked Locking (Thread-Safety Bug)

**File:** `models/inference/predictor.py`, Lines 20–40  
**Problem:** The double-checked locking pattern is broken. The `_cache_lock` is acquired, the inner guard passes, but the cached values are set **after** the lock is released. A second thread can enter `load_model_and_tokenizer()`, pass both null checks (before the first thread writes the cached values), and load the model a second time — causing two models in memory and a race on `_cached_model`.

**Root Cause:** Model loading (`ModelRegistry.load_model()`) and the assignment of `_cached_tokenizer`/`_cached_model` (lines 37–38) happen **outside** the `with _cache_lock` block.

**Exact Fix:**
```python
def load_model_and_tokenizer() -> Tuple[Any, Any]:
    global _cached_tokenizer, _cached_model

    if _cached_tokenizer is not None and _cached_model is not None:
        return _cached_tokenizer, _cached_model

    with _cache_lock:
        if _cached_tokenizer is not None and _cached_model is not None:
            return _cached_tokenizer, _cached_model

        # ← Move ALL of this inside the lock
        from src.models.registry.model_registry import ModelRegistry
        assets = ModelRegistry.load_model()
        tokenizer = assets["tokenizer"]
        model = assets["model"]
        model.eval()
        _cached_tokenizer = tokenizer
        _cached_model = model

    return _cached_tokenizer, _cached_model
```

---

### CRIT-2 | `models/inference/predictor.py` — Hardcoded `max_length=512` Ignores Settings

**File:** `models/inference/predictor.py`, Lines 119, 142  
**Problem:** Both `predict_batch` and `predict` hardcode `max_length=512`, ignoring `SETTINGS.model.max_length`. If a model was trained with a different sequence length (e.g., 256 or 128), predictions will silently use wrong padding/truncation, degrading accuracy and wasting compute.

**Exact Fix:**
```python
# At module top, after imports:
from src.utils.settings import load_settings as _load_settings
_SETTINGS = _load_settings()

# In both predict() and predict_batch(), replace:
#   max_length=512,
# with:
    max_length=_SETTINGS.model.max_length,
```

---

### CRIT-3 | `src/inference/prediction_pipeline.py` — `torch.autocast` Called with `device_type="cuda"` on CPU Devices

**File:** `src/inference/prediction_pipeline.py`, Lines 387–404  
**Problem:** `_forward_all` calls `torch.autocast(device_type="cuda", ...)` unconditionally. On CPU-only Replit (and any CPU deployment), this raises a `RuntimeError` in PyTorch ≥ 2.0: *"CUDA autocast is not supported on CPU."* The pipeline crashes entirely on any non-GPU machine.

**Root Cause:** The `enabled=self.device.type == "cuda"` flag suppresses the autocast behavior but **does not prevent the device_type validation** in recent PyTorch versions.

**Exact Fix:**
```python
def _forward_all(self, features: torch.Tensor) -> Dict[str, Any]:
    outputs: Dict[str, Any] = {}
    device_type = self.device.type if self.device.type in ("cuda", "cpu") else "cpu"
    with torch.autocast(
        device_type=device_type,
        dtype=torch.bfloat16,
        enabled=self.device.type == "cuda",
    ):
        ...
```
Or more safely:
```python
    ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if self.device.type == "cuda"
        else contextlib.nullcontext()
    )
    with ctx:
        ...
```

---

### CRIT-4 | `src/models/registry/model_registry.py` — Global Monkeypatch of `AutoModel.from_pretrained` is Not Thread-Safe

**File:** `src/models/registry/model_registry.py`, Lines 86–101  
**Problem:** `_load_multitask_model` replaces the global `transformers.AutoModel.from_pretrained` with a stub, loads the model, then restores it. Under concurrent requests (FastAPI runs multiple workers), a second thread calling ANY code that uses `AutoModel.from_pretrained` during this window will silently get the stub — loading a randomly-initialized model instead of pretrained weights, with no error.

**Root Cause:** Global symbol mutation in a multithreaded context.

**Exact Fix:** Pass the config directly to the `MultiTaskTruthLensModel` constructor instead of relying on `from_pretrained` internally. Add a `from_config_only: bool` parameter to `TransformerEncoder.__init__` that calls `AutoModel.from_config(hf_cfg)` instead of `AutoModel.from_pretrained(name)`.

```python
# In _load_multitask_model — REMOVE the monkeypatch entirely:
# Pass a flag telling the encoder to skip weight downloads:
model_cfg = MultiTaskTruthLensConfig(
    model_name=base_model_name,
    dropout=cfg.get("dropout", 0.1),
    pooling=cfg.get("pooling", "cls"),
    init_from_config_only=True,   # ← new flag
)
model = MultiTaskTruthLensModel(config=model_cfg)
```

---

### CRIT-5 | `src/aggregation/aggregation_pipeline.py` — `compute_scores` Called Twice on Same Normalized Profile

**File:** `src/aggregation/aggregation_pipeline.py`, Lines 247–249  
**Problem:** `raw_scores` is described as "pre-weight scores" but is computed on the **already-normalized profile** with `weights=None` (which still uses `self.defaults`, not raw weights). This means `raw_scores` and `scores` are both operating on the normalized profile — `raw_scores` is not actually "raw" in any meaningful sense. The double computation wastes CPU and misleads downstream consumers who may compare `scores` vs `raw_scores` expecting a pre/post-normalization comparison.

```python
raw_scores = self.score_calculator.compute_scores(normalized_profile, weights=None)  # ← uses self.defaults
weights = self.weight_manager.get_weights()
scores = self.score_calculator.compute_scores(normalized_profile, weights=weights)   # ← uses managed weights
```

**Root Cause:** `compute_scores(weights=None)` falls back to `self.weights` (which is `self.defaults`), not a "no-weight" identity. The conceptual distinction between `raw_scores` and `scores` is lost.

**Exact Fix:**
```python
# Compute raw scores BEFORE normalization on the enriched profile:
raw_scores = self.score_calculator.compute_scores(enriched_profile, weights=None)
normalized_profile = self.normalize_profile(enriched_profile)
weights = self.weight_manager.get_weights()
scores = self.score_calculator.compute_scores(normalized_profile, weights=weights)
```

---

### CRIT-6 | `src/aggregation/score_normalizer.py` — Min-Max on Constant Multi-Value Section Returns 0.5 Midpoint, Destroying Signals

**File:** `src/aggregation/score_normalizer.py`, Lines 98–101  
**File:** `src/aggregation/aggregation_pipeline.py`, Lines 163–167  
**Problem:** The pipeline only skips normalization for **single-value** sections. When a section has multiple features that all happen to be identical (e.g., all zeros from an empty analysis), `normalize_minmax` returns `0.5` for every feature. A section where everything is `0.0` (no signals detected) becomes a section where everything is `0.5` (neutral signal) — which is then incorrectly weighted into the credibility score as if mild signals were detected.

**Exact Fix in `aggregation_pipeline.py`:**
```python
# After: if len(numeric_keys) == 1:
# Add:
values_check = [features[k] for k in numeric_keys]
if max(values_check) - min(values_check) < 1e-12:
    # Constant section — preserve as-is; normalization is meaningless
    normalized_profile[section] = features.copy()
    continue
```

---

### CRIT-7 | `src/explainability/` — Token Space Mismatch Between Explainers Before Aggregation

**Files:** `src/explainability/explanation_aggregator.py`, `src/inference/prediction_pipeline.py` Lines 238–266  
**Problem:** The `ExplanationAggregator.aggregate()` merges SHAP, LIME, and attention rollout attributions by token. However:
- **SHAP** uses `shap.maskers.Text()` — operates on word-level chunks (split by whitespace/punctuation)
- **LIME** uses `LimeTextExplainer` — operates on word-level tokens (space-split)
- **Attention Rollout** operates on **model subword tokens** (WordPiece/SentencePiece), then optionally aligned via `align_tokens`

The three token lists are **different sizes** and **different granularities**. Aggregating them by position (index-based) produces meaningless merged scores — the 3rd SHAP word does not correspond to the 3rd attention token.

**Root Cause:** No canonical token space is established before aggregation. SHAP and LIME outputs are word-level, attention rollout is subword-level even after `align_tokens` (special tokens `[CLS]`, `[SEP]` may still be present).

**Exact Fix:**
```python
# In ExplainabilityLayer.explain(), after computing all three methods:
# 1. Strip special tokens from aligned_tokens before aggregation
# 2. Merge SHAP/LIME by exact token string match, not position:

def _merge_by_token_string(
    shap_items, lime_items, attention_items
) -> Dict[str, float]:
    """Build a unified token→score map by string key, not index."""
    ...
```
Or use a span-based alignment: map each method's output back to character offsets in the original text, then merge by character overlap.

---

### CRIT-8 | `src/inference/prediction_pipeline.py` — `model.half()` Applied Before `torch.compile`, Causing FP16 Inference on CPU

**File:** `src/inference/prediction_pipeline.py`, Lines 347–376  
**Problem:** The constructor applies `model.half()` (FP16) to all models when `device.type == "cuda"`. However, the `device` is read from `PredictionPipelineConfig.device` which defaults to `"cpu"`. Any caller that instantiates `PredictionPipeline(config=PredictionPipelineConfig(device="cpu"))` with CUDA models will not hit this branch — but any caller that uses a GPU config and then falls back to CPU due to CUDA unavailability (which is the Replit environment) will have half-precision models on CPU, causing silent precision loss and potentially `RuntimeError` in some ops.

Additionally, `torch.compile` with `mode="max-autotune"` is only meaningful on CUDA; on CPU it wastes startup time with no benefit and `max-autotune` is not supported.

**Exact Fix:**
```python
# Guard both half() and compile() strictly:
if self.device.type == "cuda" and torch.cuda.is_available():
    for m in [...]:
        if m is not None:
            m.half()
    # compile only on cuda
    for m in [...]:
        ...torch.compile(m, mode="reduce-overhead")  # not max-autotune on CPU
```

---

## 🟠 Major Issues

---

### MAJOR-1 | `src/aggregation/truthlens_score_calculator.py` — `_aggregate_section` Uses Unweighted Mean, Dilutes Sections with Many Features

**File:** `src/aggregation/truthlens_score_calculator.py`, Lines 159–169  
**Problem:** Every feature inside a section is averaged with equal weight. A section with 2 features weights each at 0.5. A section with 50 features (after analysis injection) weights each at 0.02. This means the more granular the analysis, the less impact each individual signal has — creating an inverse relationship between analysis depth and signal strength. The `bias` section with `bias_prediction` plus 20 injected framing/ideological features will have its core `bias_prediction` diluted to 1/21 weight.

**Exact Fix:** Keep a canonical set of "primary" keys per section and weight them higher, or use a separate normalization pass that re-weights injected features relative to primary signals.

---

### MAJOR-2 | `src/aggregation/truthlens_score_calculator.py` — Credibility Score Can Be Meaninglessly Negative Before Clipping

**File:** `src/aggregation/truthlens_score_calculator.py`, Lines 196–203  
**Problem:** The credibility formula is:
```
credibility = 0.55 * discourse + 0.35 * graph - 0.20 * bias + 0.10 * analysis
```
Maximum positive contribution: `0.55 + 0.35 + 0.10 = 1.00`  
Maximum negative contribution: `0.20`  
For a zero-evidence article (discourse=0, graph=0, analysis=0, bias=1.0):  
`credibility = -0.20` → clipped to `0.0`

The clipping masks the signal — the final score treats minimum credibility the same whether the article had zero discourse signals or actively contradictory ones. This makes `credibility_score` non-monotonic with respect to the inputs.

**Exact Fix:** Reformulate to keep the output in a meaningful range without relying on clip:
```python
# Normalize so minimum is 0 without clip:
credibility = (
    w["discourse"] * discourse_score
    + w["graph"] * graph_score
    + w["analysis_influence_credibility"] * analysis_score
) * (1.0 - w["credibility_bias_penalty"] * bias_score)
```

---

### MAJOR-3 | `src/models/registry/model_registry.py` — Module-Level `load_settings()` Triggers Config Load on Import

**File:** `src/models/registry/model_registry.py`, Line 119  
**Problem:** `SETTINGS = load_settings()` executes at **import time**. Any module that imports from `model_registry` (directly or transitively) triggers a YAML config load, path resolution, and logging setup. In tests, this causes: (a) test isolation failures if the config file is absent, (b) slow imports, (c) import-order-dependent behavior if the config hasn't been written yet.

**Exact Fix:** Use lazy initialization:
```python
_SETTINGS = None

def _get_settings():
    global _SETTINGS
    if _SETTINGS is None:
        _SETTINGS = load_settings()
    return _SETTINGS
```

---

### MAJOR-4 | `src/utils/json_utils.py` — `append_json` Reads Entire File into Memory on Every Call

**File:** `src/utils/json_utils.py`, Lines 173–229  
**Problem:** Every call to `append_json` reads the complete JSON file, deserializes it, appends one entry, reserializes, and writes the full file back. For the inference logger writing one record per prediction, a log file with 10,000 entries means deserializing 10,000 records just to add one. This grows O(n) per append — a classic quadratic pattern.

**Exact Fix:** Switch to newline-delimited JSON (JSONL):
```python
def append_json(entry: dict, path: str | Path) -> Path:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with _locked_path(path_obj):
        with path_obj.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return path_obj
```

---

### MAJOR-5 | `src/utils/json_utils.py` — `_FILE_LOCKS` Dictionary Grows Unboundedly

**File:** `src/utils/json_utils.py`, Line 44 (inferred from usage pattern)  
**Problem:** The per-path `threading.Lock` dictionary adds a new entry for every unique file path accessed and never removes entries. In a long-running server that writes to many temporary or session-based log paths, this is a memory leak.

**Exact Fix:** Use `weakref.WeakValueDictionary` for the lock store, or implement an LRU-eviction policy capped at a reasonable size (e.g., 1024 entries).

---

### MAJOR-6 | `src/models/multitask/multitask_truthlens_model.py` — Temperature Scaling Applied to Only 3 of 6 Task Heads

**File:** `src/models/multitask/multitask_truthlens_model.py`, Lines 375–393  
**Problem:** Temperature scaling (calibration) is applied to `bias_logits`, `ideology_logits`, and `propaganda_logits` but **not** to `narrative_outputs`, `narrative_frame_outputs`, or `emotion_outputs`. This means the multi-label heads produce uncalibrated probabilities while the binary/multiclass heads are calibrated — breaking the credibility score composition which uses outputs from all heads.

**Exact Fix:**
```python
narrative_outputs = self.narrative_head(pooled) / temperature
narrative_frame_outputs = self.narrative_frame_head(pooled) / temperature
emotion_outputs = self.emotion_head(pooled) / temperature
```

---

### MAJOR-7 | `src/training/train_transformer_model.py` — No Deduplication Before Train/Val/Test Split

**File:** `src/training/train_transformer_model.py`, Lines 220–253  
**Problem:** The `_split_train_val_test` function splits directly on the input DataFrame without checking for duplicate rows. If the raw datasets (ISOT, LIAR, BABE merged) contain duplicate articles, those duplicates will appear across splits — constituting data leakage where the model sees training examples at validation/test time, inflating reported metrics.

**Exact Fix:**
```python
def _split_train_val_test(df: pd.DataFrame, ...):
    # Deduplicate on text content before splitting
    text_col = "text"  # or the appropriate column name
    if text_col in df.columns:
        original_len = len(df)
        df = df.drop_duplicates(subset=[text_col]).reset_index(drop=True)
        if len(df) < original_len:
            logger.warning("Removed %d duplicate rows before splitting", original_len - len(df))
    ...
```

---

### MAJOR-8 | `src/training/train_transformer_model.py` — `gradient_accumulation_steps` Hardcoded, Conflicts with Config

**File:** `src/training/train_transformer_model.py`, Line 326 (approximately)  
**Problem:** `gradient_accumulation_steps=2` is hardcoded in the `TrainingArguments` construction but `config.yaml` specifies `1`. Any team member expecting the config to control training behavior will be silently overridden. Effective batch size differs from what the config declares.

**Exact Fix:**
```python
gradient_accumulation_steps=int(
    params.get("gradient_accumulation_steps", 
    SETTINGS.training.gradient_accumulation_steps)
),
```

---

### MAJOR-9 | `src/inference/prediction_pipeline.py` — Credibility Score Uses Hardcoded Weights, Duplicates Aggregation Logic

**File:** `src/inference/prediction_pipeline.py`, Lines 60–63, 563–570  
**Problem:** `PredictionPipelineConfig` defines its own credibility weights (`credibility_weight_bias=0.25`, `credibility_weight_propaganda=0.35`, etc.) and `_compute_credibility_score` uses them. Meanwhile, `AggregationPipeline` has its own separate credibility formula with different weights (discourse=0.55, graph=0.35, bias_penalty=0.20). The same concept is computed twice with different logic and different outputs. The `/predict` and `/analyze` endpoints likely produce inconsistent credibility scores.

**Exact Fix:** Remove `_compute_credibility_score` from `PredictionPipeline` entirely. Route all credibility computation through `AggregationPipeline.run()`.

---

### MAJOR-10 | `src/analysis/information_density_analyzer.py` — `information_emotion_ratio` Not Normalized to [0, 1]

**File:** `src/analysis/information_density_analyzer.py` (inferred from audit)  
**Problem:** `information_emotion_ratio` is clipped to `10.0` (factual density / emotion density, max=10) but not scaled to `[0, 1]`. When this feature is injected into the aggregation pipeline's `analysis` section and then averaged with other `[0, 1]` features, it dominates the section mean — inflating the analysis influence component by up to 10×.

**Exact Fix:**
```python
information_emotion_ratio = min(factual_density / (emotion_density + 1e-9), 10.0)
# Add after: normalize to [0, 1] for pipeline compatibility
information_emotion_ratio_normalized = information_emotion_ratio / 10.0
```

---

## 🟡 Minor Issues

---

### MINOR-1 | `src/explainability/token_alignment.py` — WordPiece Score of First Subtoken Not Included in Mean

**File:** `src/explainability/token_alignment.py`, Lines 29–39  
**Problem:** When processing WordPiece tokens, the score for the **first subtoken** (the non-`##` token that starts a word) is added to `current_scores` (line 39: `current_scores = [score]`) but the score for the next non-`##` token is flushed **before** the new word's first score is initialized. The logic is correct in this regard. However, when a word is a single token (no `##` continuations), its score is in `current_scores = [score]`, which is correct. This is actually fine — **but**: the SentencePiece branch (line 46: `current_token = token[1:]`) strips the `▁` character without adding the score of the **first** piece to `current_scores` before resetting. The first piece's score is discarded because `current_scores = [score]` is set correctly, so this is also fine. 

**Actual issue:** Special tokens (`[CLS]`, `[SEP]`, `<s>`, `</s>`) are passed through without stripping and appear in the aligned output, causing downstream consumers to display `[CLS]` as a "word" with an attribution score.

**Exact Fix:**
```python
SPECIAL_TOKENS = {"[CLS]", "[SEP]", "<s>", "</s>", "[PAD]", "<pad>"}

for token, score in zip(tokens, scores):
    token = str(token)
    if token in SPECIAL_TOKENS:
        continue
    ...
```

---

### MINOR-2 | `models/inference/predictor.py` — `_resolve_fake_index` Silently Defaults to 1, Inverting Predictions

**File:** `models/inference/predictor.py`, Lines 55–72  
**Problem:** If a model has no `label2id` or `id2label` config (e.g., a freshly loaded model without metadata), `_resolve_fake_index` returns `DEFAULT_FAKE_INDEX = 1`. If the actual trained model uses index `0` for Fake and `1` for Real (opposite convention), all predictions are silently inverted — every real article is classified as fake and vice versa.

**Exact Fix:** Log a warning when falling back to default index, and validate against the model config:
```python
logger.warning(
    "Could not resolve 'FAKE' index from model config; "
    "defaulting to index %d. Verify label ordering.", DEFAULT_FAKE_INDEX
)
```

---

### MINOR-3 | `src/utils/device_utils.py` — `move_to_device` with `inplace=True` Creates Inconsistent State

**File:** `src/utils/device_utils.py`  
**Problem:** When `inplace=True` is used for dict/list, elements that are tensors return new objects (tensors cannot be moved in-place) while non-tensor elements are mutated. If the caller holds a reference to a tensor before the call, it still points to the old device. The container appears updated but contained tensors are actually new objects.

**Exact Fix:** Document that `inplace=True` only affects the container structure, not tensor identity. Or remove `inplace` and always return new containers.

---

### MINOR-4 | `src/inference/inference_cache.py` — Cache Key Collision Risk with SHA-256 Truncation

**File:** `src/inference/inference_cache.py`  
**Problem:** If SHA-256 hashes are truncated (common practice for brevity), birthday-paradox collisions become feasible at scale. An adversary submitting crafted inputs could poison a cache entry for a different article.

**Exact Fix:** Use full 64-character SHA-256 hex digest as the cache key. Do not truncate.

---

### MINOR-5 | `src/evaluation/calibration.py` — ECE Computed with Fixed 15 Bins Regardless of Dataset Size

**Problem:** ECE with 15 bins on a dataset of < 150 samples has most bins empty, making the metric statistically meaningless. On very large datasets (> 100,000), 15 bins are too coarse.

**Exact Fix:**
```python
n_bins = max(5, min(int(np.sqrt(len(labels))), 30))
```

---

### MINOR-6 | `src/models/training/trainer.py` — `_attempt_resume` Does Not Restore Learning Rate Scheduler State

**Problem:** The resume logic loads model and optimizer states but the scheduler's `last_epoch` is not restored. Resuming from checkpoint restarts the LR schedule from epoch 0, causing LR to be at its warmup peak instead of the decayed value expected mid-training.

**Exact Fix:**
```python
if "scheduler_state_dict" in checkpoint:
    self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
```

---

### MINOR-7 | `src/aggregation/score_schema.py` — Pydantic `TruthLensAggregationOutputModel` Accepts Scores Outside [0,1] in `raw_scores`

**Problem:** `raw_scores` is typed as `Dict[str, Any]` or with relaxed validation — it does not enforce the `[0, 1]` constraint that `scores` enforces. Callers that read `raw_scores` for display may render out-of-range values (e.g., negative credibility before clipping was added).

**Exact Fix:** Apply the same `ge=0, le=1` validators to all score fields in both `scores` and `raw_scores`.

---

## ⚙️ Performance Improvements

---

### PERF-1 | `src/aggregation/aggregation_pipeline.py` — `copy.deepcopy` on Every Prediction

`_inject_analysis_sections` calls `copy.deepcopy(profile)` on every request. For large profiles with many analysis modules, this is expensive. The profile is not shared between threads; a shallow copy of the top-level dict with per-section shallow copies suffices.

```python
# Replace:
enriched = copy.deepcopy(profile)
# With:
enriched = {k: dict(v) if isinstance(v, dict) else v for k, v in profile.items()}
```

---

### PERF-2 | `src/features/text/semantic_features.py` — Loads `all-MiniLM-L6-v2` Transformer on Every Feature Extraction

The semantic feature extractor loads the sentence-transformer model inside `extract()` rather than as a class-level singleton. Every article triggers a model load check (and potentially a re-download). This should be a module-level or class-level cached instance.

```python
# Move to class __init__ or module level:
_SEMANTIC_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
```

---

### PERF-3 | `src/analysis/` — spaCy Models Loaded Multiple Times Per Request

Each analyzer independently calls `get_nlp()` which may load separate spaCy pipeline instances. The `IntegrationRunner` is designed to share a single `Doc` object, but if analyzers are instantiated outside the runner (as they are in `api/app.py` as singletons), each holds its own NLP model. With 15+ analyzers each potentially holding a separate spaCy model, memory usage is multiplied.

**Fix:** Verify that all singleton analyzers in `api/app.py` share the same underlying spaCy model via the `get_nlp()` singleton cache in `src/analysis/_nlp.py`.

---

### PERF-4 | `src/explainability/shap_explainer.py` — LRU Cache Keyed on `predict_fn` Object Identity

If `predict_fn` is a lambda or closure recreated per-request, the SHAP `Explainer` object cache will miss on every request (each new closure has a different identity), rebuilding the SHAP explainer from scratch. SHAP explainer construction is expensive (hundreds of ms).

**Fix:** Cache keyed on a stable string identifier (e.g., model path + tokenizer name), not function object identity.

---

### PERF-5 | `src/features/importance/shap_importance.py` — Repeated SHAP Computation Alongside `src/explainability/shap_explainer.py`

SHAP is computed in both the feature importance module and the explainability module independently. A single article going through both paths runs SHAP twice with no result sharing.

**Fix:** Compute SHAP once and pass the result to both consumers.

---

## 🧹 Dead Code / Unused

---

### DEAD-1 | `main copy.py` — Stale Backup File

`main copy.py` is an uncommitted backup of `main.py`. It contains an older training pipeline that will never be executed. **Delete it.**

---

### DEAD-2 | `z.py`, `zsave.txt`, `zz.txt`, `save.txt` — Debug Scratch Files

These files appear to be developer scratch pads with no production relevance. **Delete all.**

---

### DEAD-3 | `structure.md`, `structure2.md`, `structure2 copy.md`, `structure2 copy 2.md` — Stale Architecture Notes

Four overlapping architecture description files exist alongside `documentation/`. Consolidate into `documentation/PROJECT_STRUCTURE.md` and delete the root-level duplicates.

---

### DEAD-4 | `graph_hardening_patch.py` — Should Live in `src/graph/`, Not Root

This file is imported by all `src/graph/` modules but lives in the project root. It is an implementation file masquerading as a patch. Move to `src/graph/graph_math.py` and update imports.

---

### DEAD-5 | `features/pipelines/feature_pipeline.py` and `pipelines/` at Root Level — Duplicate of `src/features/` and `src/pipelines/`

The root-level `features/` and `pipelines/` directories appear to be legacy copies of `src/features/` and `src/pipelines/`. Confirm they are unused (no imports resolve to root-level `features.` or `pipelines.`) and remove.

---

### DEAD-6 | `training/train_transformer_model.py` at Root — Duplicate of `src/training/`

`training/train_transformer_model.py` (root-level) is a copy of `src/training/train_transformer_model.py`. One will drift from the other silently. **Remove the root-level copy.**

---

### DEAD-7 | `src/evaluation/pdf_report.py` — Depends on Optional `reportlab` Not in `requirements.txt`

`pdf_report.py` imports `reportlab` which is not listed in `requirements.txt`. The module will fail at import and is likely unused in any active code path. Either add `reportlab` to requirements or remove the file.

---

### DEAD-8 | `src/evaluation/mlflow_tracker.py` — MLflow Not in `requirements.txt`

Same issue as above: `mlflow` is imported but not listed as a dependency. Any evaluation run that hits this path will crash with `ModuleNotFoundError`.

---

### DEAD-9 | `src/models/emotion/` — Separate Emotion Model Alongside MultiTask Head

`src/models/emotion/load_emotion_model.py` and `train_emotion_model.py` describe a standalone emotion model that predates the unified `MultiTaskTruthLensModel`. The emotion task is now a head in the multitask model. These files are likely dead code.

---

## 📁 Missing / Broken Files

---

### MISSING-1 | `src/models/inference/predictor.py` vs `models/inference/predictor.py` — Two Predictors with Different Logic

There are two `predictor.py` files:
- `models/inference/predictor.py` — the compatibility wrapper used by `api/app.py`  
- `src/models/inference/predictor.py` — a separate implementation under `src/`

`api/app.py` imports from `models.inference.predictor` (the root-level one). The `src/`-level version may be silently stale. Confirm which is authoritative and delete the other.

---

### MISSING-2 | `src/evaluation/pdf_report.py` and `src/evaluation/mlflow_tracker.py` — Missing Dependencies

```
reportlab  — required by pdf_report.py — missing from requirements.txt
mlflow     — required by mlflow_tracker.py — missing from requirements.txt
```

Add to `requirements.txt` or remove the files.

---

### MISSING-3 | `models/tfidf_vectorizer.joblib` — Referenced But May Be Absent in Fresh Environments

`src/inference/model_loader.py` loads `models/tfidf_vectorizer.joblib` but this file is only generated after training. The server startup should gracefully handle its absence with a clear error message rather than an unhandled `FileNotFoundError` during a prediction.

---

### MISSING-4 | `config/config.yaml` — `inference.allow_raw_text_fallback` Setting Not Documented

`api/app.py` reads `SETTINGS.inference.allow_raw_text_fallback` but this key is not documented in `documentation/CONFIGURATION.md`. Any deployment team configuring the system will not know this flag exists.

---

## 🧠 Architectural Problems

---

### ARCH-1 | Dual Credibility Scoring Systems — `PredictionPipeline` and `AggregationPipeline` Diverge

**Problem:** `PredictionPipeline._compute_credibility_score()` computes credibility from bias/propaganda/emotion/ideology with weights (0.25, 0.35, 0.15, 0.25). `AggregationPipeline` computes credibility from discourse/graph/bias_penalty with weights (0.55, 0.35, 0.20). These produce numerically incompatible credibility scores. The `/predict` endpoint uses the pipeline's score, `/analyze` uses the aggregation score. Users comparing the two endpoints will see inconsistent credibility values for identical text.

**Refactor:** Designate `AggregationPipeline` as the single source of credibility computation. Remove `_compute_credibility_score` from `PredictionPipeline`. Pass all task outputs to `AggregationPipeline.build_profile_from_prediction()` and route through `run()`.

---

### ARCH-2 | `predictor.py` Uses `propaganda` Head as Fake/Real Proxy — Conceptually Wrong

**Problem:** In `_extract_probs`, when the output is a `MultiTaskTruthLensModel` dict, the **propaganda** head probabilities are used as the fake/real classifier:  
```
index 0 = non_propaganda (→ "Real")
index 1 = propaganda     (→ "Fake")
```

Propaganda and fake news are correlated but not identical. A factually accurate article can be propagandistic. A fake article can be non-propagandistic. Using propaganda probability as the primary fake/real signal introduces systematic bias and will misclassify in these edge cases.

**Refactor:** Train a dedicated binary fake/real head on the multitask model, or compose a credibility score from all heads as the fake/real signal.

---

### ARCH-3 | `src/explainability/` Has 14 Files With Significant Overlap Between `model_explainer.py`, `explanation_aggregator.py`, and `bias_explainer.py`

All three orchestrate sub-explainers in slightly different ways with no clear ownership boundary. `model_explainer.py:explain_prediction_full` calls SHAP and LIME; `ExplainabilityLayer` in `prediction_pipeline.py` calls `explain_prediction_full` PLUS attention rollout PLUS propaganda explainer PLUS aggregator. The aggregation logic is split across three layers.

**Refactor:** Flatten into one `ExplainabilityOrchestrator` class that owns the full explain-and-aggregate lifecycle with a single public `explain(text, model, tokenizer)` method.

---

### ARCH-4 | Analysis Singleton Initialization in `api/app.py` — Blocks Startup and Prevents Testing

**Problem:** `api/app.py` initializes 20+ analyzer singletons at module import time (lines 99–118). Each one loads a spaCy model. Any test that imports `api.app` triggers loading of 20+ NLP models before any test code runs, making test suites extremely slow.

**Refactor:** Use FastAPI's `lifespan` context manager for lazy initialization on first request, or use dependency injection:
```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # initialize singletons here
    yield
    # cleanup

app = FastAPI(lifespan=lifespan)
```

---

## 🔥 Top 5 Highest-Risk Bugs (Production Priority)

| Rank | Bug | Impact | File |
|---|---|---|---|
| 1 | **CRIT-3** `autocast(device_type="cuda")` on CPU | Server crash on all CPU deployments (including Replit) | `prediction_pipeline.py:387` |
| 2 | **CRIT-1** Broken double-checked locking | Race condition: two model instances loaded under concurrent traffic | `predictor.py:20–40` |
| 3 | **CRIT-7** Cross-explainer token space mismatch | Aggregated explanations are positionally misaligned nonsense | `explanation_aggregator.py` |
| 4 | **CRIT-4** Thread-unsafe monkeypatch of `AutoModel.from_pretrained` | Random-weight model loaded under concurrent startup | `model_registry.py:86–101` |
| 5 | **MINOR-2** `_resolve_fake_index` silent inversion | All predictions inverted if label schema differs from default | `predictor.py:55–72` |

---

## 🧪 Suggested Test Cases for Critical Issues

```python
# CRIT-1: Thread safety
def test_concurrent_model_loading():
    from concurrent.futures import ThreadPoolExecutor
    from models.inference.predictor import load_model_and_tokenizer, _cached_model
    _cached_model = None  # reset
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(load_model_and_tokenizer) for _ in range(8)]
    results = [f.result() for f in futs]
    assert all(r[1] is results[0][1] for r in results), "Multiple model instances created"

# CRIT-3: CPU autocast
def test_forward_all_on_cpu_does_not_crash():
    cfg = PredictionPipelineConfig(device="cpu")
    pp = PredictionPipeline(config=cfg, bias_model=DummyModel())
    features = torch.randn(1, 128)
    result = pp._forward_all(features)  # Must not raise RuntimeError

# CRIT-6: Constant-section normalization
def test_constant_section_not_normalized_to_midpoint():
    pipe = AggregationPipeline()
    profile = {"bias": {"a": 0.0, "b": 0.0, "c": 0.0}}
    normalized = pipe.normalize_profile(profile)
    assert normalized["bias"]["a"] == 0.0, "Constant zeros should not become 0.5"

# CRIT-2: max_length from settings
def test_predict_uses_settings_max_length(monkeypatch):
    from src.utils.settings import load_settings
    settings = load_settings()
    settings.model.max_length = 128
    # tokenizer call must receive max_length=128, not 512

# MAJOR-7: No deduplication — data leakage
def test_no_duplicate_overlap_between_splits():
    import pandas as pd
    df = pd.DataFrame({"text": ["a","a","b","c"], "label": [0,0,1,1]})
    train, val, test = _split_train_val_test(df)
    train_texts = set(train["text"])
    val_texts = set(val["text"])
    assert train_texts.isdisjoint(val_texts), "Duplicate text found across train/val"
```

---

## ✅ Minimal Patch to Reach Production-Grade Stability

Apply in this order:

1. **Fix CRIT-3** (CPU autocast crash) — 3-line change, prevents server crash in current Replit environment  
2. **Fix CRIT-1** (lock scope) — 5-line change, prevents race under load  
3. **Fix CRIT-2** (hardcoded max_length) — 3-line change, uses correct tokenization  
4. **Fix CRIT-4** (monkeypatch) — refactor `_load_multitask_model` to pass `from_config_only=True`  
5. **Fix CRIT-6** (constant-section normalization) — 4-line guard prevents 0→0.5 signal corruption  
6. **Fix MAJOR-6** (temperature scaling on all 6 heads) — 3-line change, calibrates all outputs consistently  
7. **Fix MAJOR-7** (deduplication before split) — 5-line change, prevents data leakage in future training runs  
8. **Delete** DEAD-1 through DEAD-6 — remove clutter and confusion  
9. **Add** `reportlab` and `mlflow` to requirements or remove the dependent files  

These 9 actions move the system from "runs but has critical correctness/safety bugs" to "production-safe for CPU inference."
