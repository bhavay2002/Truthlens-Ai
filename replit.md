# TruthLens AI

## Project Overview
TruthLens AI is a multi-layer AI platform for misinformation detection and news credibility analysis. It combines deep linguistic analysis, narrative extraction, propaganda detection, and graph-based reasoning to provide an interpretable "Credibility Score."

## Architecture
- **Backend**: FastAPI REST API (`api/app.py`) served via Uvicorn
- **Language**: Python 3.12
- **ML/NLP**: PyTorch, Hugging Face Transformers, spaCy, NLTK, LIME, SHAP
- **Port**: 5000

## Recent Refactors (audit fixes applied)
- **HARDEN-1** (training-stability audit): AMP `GradScaler` state now persisted in checkpoints (`scaler_state_dict`) and restored on `load_checkpoint`. Without this, the loss scale resets on every resume and produces the "calm → spike → calm" pattern visible in the training log.
- **HARDEN-2**: `CheckpointManager.cleanup_old_checkpoints` enforces `max_checkpoints >= 2` (was `>0`). Single-checkpoint retention risks catastrophic loss if the only surviving file is corrupt.
- **HARDEN-3**: `_train_epoch` emits a `WARNING` when raw loss exceeds `TRUTHLENS_SPIKE_RATIO` (default 5×) the running mean — instruments spike-batch visibility.
- **HARDEN-4**: Pre-clip grad norm logged every `log_every_steps` to surface exploding-grad before it spikes the loss.
- **HARDEN-5**: `main._evaluate_on_test` now raises a `RuntimeError` instead of silently `continue`-ing when bias logits are missing — that path indicates a model contract violation, not a data issue.
- **HARDEN-6**: `load_data` now warns on out-of-range / NaN values for `BIAS_LABEL`, `PROPAGANDA_LABEL`, `IDEOLOGY_LABEL` so silently-corrupted label columns are visible in the log.
- **HARDEN-7** (output-contract audit): `Trainer.load_checkpoint` now raises `RuntimeError` when any task-head weights (`bias_head`, `ideology_head`, `propaganda_head`, `narrative_head`, `narrative_frame_head`, `emotion_head`) are absent from the loaded state dict. Previously a `strict=False` resume would silently disable a head and produce the "no bias logits returned by model" eval warning.
- **HARDEN-8** (data-contract audit): `load_data` now drops empty-text rows (with a per-split count log) and raises if a split is fully empty. Empty rows tokenize to padding-only batches and produce meaningless gradients.
- **HARDEN-9**: Spike-loss warning now also reports per-task losses and a batch signature (input-id checksum) so the offending head and batch can be located instead of just observed.
- **HARDEN-10** (defensive instrumentation subsystem): new `src/training/instrumentation.py` module wired into `Trainer`. Provides:
  * `LossTracker` — per-task **bias-corrected** EMA (Adam-style: init at 0, correct by `1-(1-α)^t`) with NaN/Inf rejection. Replaces the previous fixed `0.9/0.1` running mean.
  * `LossStats` — windowed mean + variance per task (`window=50`), exposes instability that EMA alone hides.
  * `GradTracker` — windowed history of total / mean / per-parameter grad norms; `detect_grad_anomaly` classifies `EXPLODING` / `VANISHING` / `NORMAL` and warns on anything but `NORMAL`.
  * `SpikeDetector` — hybrid `ratio | z-score` detector, fixes the false positives near zero EMA that the pure-ratio version produced.
  * `validate_labels`, `check_optimizer`, `apply_clipping` — fail-fast utilities.
  * `dump_batch` — atomic `.pt` post-mortem with inputs, logits, per-task losses, smoothed losses, loss stats, and LR snapshot. Capped at `TRUTHLENS_MAX_DEBUG_DUMPS` (default 20) per run; written to `<checkpoint_dir>/debug_dumps/`.
  * Wiring: `Trainer.__init__` constructs the trackers; `_train_epoch` updates them every step, fires the spike detector on the worst-offending task (so a single batch dumps once not 6×), and the optimizer-step block now logs `total_norm | mean_norm | mean_var` and warns on grad anomalies.
  * Tested in `tests/test_instrumentation.py` (10 unit tests).
- **HARDEN-11** (instrumentation phases 4-7): `src/training/instrumentation.py` extended with:
  * `AnomalyClassifier` — priority-ordered multi-signal classifier returning one of `nan_loss`, `nan_logits`, `exploding_gradients`, `vanishing_gradients`, `negative_labels`, `invalid_labels`, `high_variance`, `loss_spike`, `logit_collapse`, `normal`. Order is deliberate so a NaN loss is never reported as a loss spike.
  * `anomaly_severity` — coarse `critical | high | medium | low` triage bucket.
  * `GradNorm` (Chen et al., 2018, paper-aligned) + `compute_task_grad_norms` helper that computes per-task gradient norms over the **shared backbone** (the only correct formulation). Weights satisfy the `sum(weights) == T` constraint. Opt-in: not wired into the default training loop because changing loss weighting alters convergence; enable explicitly when the dominance detector flags imbalance.
  * `GradHookManager` — per-parameter backward-hook collector with `attach(filter_fn=…)` / `aggregate()` / `reset()` / `detach()`. Detach is mandatory before model deletion to avoid hook leaks.
  * `TaskDominanceDetector` — EMA-smoothed max/min ratio detector that replaces the previous `_task_loss_dominance_summary`. Smoothing prevents single-step flapping.
  * Wiring: `Trainer` now uses `AnomalyClassifier` + `anomaly_severity` whenever the spike detector fires (so dumps are tagged with the actual failure category), and `TaskDominanceDetector` replaces the previous noisy max/min summary. Tunables: `TRUTHLENS_DOMINANCE_RATIO` (default `5.0`).
  * 17 new unit tests in `tests/test_instrumentation.py` (53 tests total in the module).
- **HARDEN-12** (control plane — 12 new components in `src/training/instrumentation.py`):
  * **Detection (auto-wired, observation-only):**
    1. `SilentCollapseDetector` — EMA-smoothed; fires only when *both* a sudden raw-loss drop (`loss < loss_ratio × loss_ema`) and a degraded smoothed metric (`metric_ema < metric_floor`) hold for `patience` consecutive updates. Both conditions are required so a legitimate loss drop with healthy F1 doesn't false-alarm.
    2. `classify_collapse_type(logits)` — returns `mode_collapse` (all argmax identical) / `confidence_collapse` (max prob < 0.4) / `unknown`.
    3. `GradientConflictDetector` + `flatten_grads` — pairwise cosine similarity over flattened shared-backbone grad vectors; reports pairs with `sim < threshold` (default 0).
    4. `SpikeCluster` + `spike_severity` — sliding-window spike *density* tracker (deque-based) with `low/medium/high/critical` buckets. Wired into `Trainer._train_epoch` (logs warning when density > 0.2 over a 50-step window).
    5. `TaskDominanceDetector` extended: now returns `type=grad_dominance` or `type=grad_zero_collapse` when a task's smoothed grad fully vanishes (was silently swallowed before).
  * **Action (opt-in helpers, NOT auto-wired):** these mutate optimizer/loss state, so `Trainer` does not invoke them automatically — callers must opt in once the convergence implications are understood.
    6. `handle_task_dominance(result, optimizer, task_weights)` — decays dominant task weight (×0.7), boosts suppressed weight (×1.3), optionally decays dominant head LR (×0.5).
    7. `handle_silent_collapse()` — returns the manual-inspection checklist (label distribution, data leakage, class imbalance, augmentation errors). Deliberately advisory, not auto-mutating, because the right response depends on which check fails.
    8. `resolve_conflicts(conflicts, weights, rate=0.1)` — soft-damps both conflicting tasks' weights by `rate × |sim|`.
    9. `TaskBalancer(nn.Module)` — Kendall et al. (2018) homoscedastic-uncertainty balancer with `nn.ParameterDict` log-vars (registered as parameters so they actually train, fixing the doc's bug).
  * **Control (composable brain):**
    10. `BatchAnalyzer` — priority-ordered single-string classifier for fused signal dicts (distinct from per-tensor `AnomalyClassifier`); exposes `analyze` and `analyze_multi`.
    11. `FailureClassifier` — returns `(root_cause, [all_flags])` so reports can show co-occurring symptoms; `FailureMemory` stores capped per-type history with `recent`/`distribution` queries; `detect_failure_trend` flags persistent patterns.
    12. `AutoDebugEngine` — wires detectors + classifier + memory into a single `step(signals)` call, swallowing per-detector `TypeError` so a misconfigured detector cannot crash the training step. `HealthScore` (weighted, signals → [0,1]) + `SmoothedHealth` (EMA) wired into `Trainer` and logged every `TRUTHLENS_HEALTH_LOG_EVERY` steps (default 100).
  * **Trainer wiring:** spike density, dominance type, and the composite health score are emitted automatically. Tunables: `TRUTHLENS_SPIKE_WINDOW` (50), `TRUTHLENS_SPIKE_DENSITY` (0.2), `TRUTHLENS_HEALTH_LOG_EVERY` (100).
  * **Tests:** 36 new unit tests in `tests/test_instrumentation.py` (89 total in the broader related suite, all passing).
  * **Deliberately NOT auto-wired:** `TaskBalancer`, `handle_task_dominance`, `handle_silent_collapse`, `resolve_conflicts`. These change loss/optimizer dynamics; auto-mutation hides bugs instead of surfacing them. The "auto-recovery layer" the source doc proposes (LR annealing on anomaly clusters, batch quarantine, head reinitialization) is intentionally out of scope for the same reason — observability first, automation only on demand.
- **PERF-3**: All 14 singleton analyzers share one spaCy `en_core_web_sm` model via `get_shared_nlp()` in `src/analysis/_nlp.py`. All `disable_components` defaults unified to `()` so the cache key is always `("en_core_web_sm", ())`. Previous state: 4 separate pipeline instances.
- **ARCH-1**: `PredictionPipeline._compute_credibility_score()` and its dead private task methods (`_predict_bias`, `_predict_ideology`, `_predict_propaganda`, `_predict_emotion`) removed. Credibility computation is now exclusively owned by `AggregationPipeline`. `predict_with_aggregation()` reads `truthlens_credibility_score` directly from aggregation output.
- **ARCH-3**: `ExplainabilityLayer` (was in `prediction_pipeline.py`) and `explain_prediction_full`/`explain_fast` (was in `model_explainer.py`) consolidated into `ExplainabilityOrchestrator` in `src/explainability/orchestrator.py`. Single `explain()` method owns the full lifecycle: SHAP → LIME → bias/emotion → attention rollout → propaganda → aggregation → consistency. Backward-compat shims kept in both files.
- **CRIT-P2-1** (second-pass audit): `models/inference/predictor.py` `predict_batch` was collapsing N texts into 1 averaged result and returning a single dict — causing silent data loss and a `KeyError` crash in the `/batch-predict` fallback path. Fixed: per-sample tensor slicing after the batch forward pass; returns `[[real0,fake0], [real1,fake1], …]` as `app.py` expects.
- **CRIT-P2-2** (second-pass audit): `src/models/inference/predictor.py` `build_fake_real_output` called `probs.argmax(dim=-1).item()` on a (N,2) tensor — raises `RuntimeError` for N>1. Fixed: `probs.mean(dim=0)` collapses batch first; `argmax` and `max` are then safe for any batch size.

## Training-Log Audit (15-issue fix pass, April 2026)
- **#1/#13 Resume key mismatch (showstopper)**: `CheckpointManager.save_checkpoint` saved under key `model` while `src/training/checkpointing.load_checkpoint` strictly required `model_state_dict`, so every resume silently failed and training restarted from scratch. Saver now writes the canonical keys (`model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`); `Trainer.load_checkpoint` accepts both old and new keys. `Trainer._attempt_resume` now **raises** when an on-disk checkpoint fails to load (escape hatch: `TRUTHLENS_ALLOW_RESUME_FAIL=1`).
- **#3 Test-eval pipeline broken**: `main._evaluate_on_test` was reading `outputs["heads"]["bias"]`, but the model returns `outputs["bias"]["logits"]` — every batch produced `logits=None`, so `y_true` stayed empty and the run logged "Test evaluation skipped: no bias logits". Fixed with the correct path; old shape kept as backward-compat fallback.
- **#7 `checkpoint-1000000001` artifacts**: best model now lives in `<ckpt_dir>/best/` (was already in place); plus checkpoint listing is now numerically sorted (`checkpoint-100` > `checkpoint-2`), removing latent "wrong latest resumed" risk.
- **#14 Validator wired**: `validate_checkpoint` (encoder + head prefixes) is now invoked inside `CheckpointManager.save_checkpoint` so unusable payloads fail loudly *before* serialization.
- **#4 DtypeWarning**: `main.load_data` pins explicit dtypes for text + label columns in addition to `low_memory=False`.
- **#10 torch.compile**: passed `dynamic=True` to bound recompiles under the bucket sampler.
- **#15 Reproducibility**: `set_seed` honors `TRUTHLENS_DETERMINISTIC=1` and the missing `Optional` import in `seed_utils` is fixed.
- All 9 checkpoint-related tests pass (`tests/test_checkpoint_manager.py`, `tests/test_training_checkpointing.py`, `tests/test_checkpointing.py`).

## Deep ML Systems Audit (April 2026)
- **Predictor key matching**: `_format_outputs` now uses `key.endswith("_logits")` instead of a loose `"logits" in key` substring match, so keys like `per_sample_logits_norm` are no longer mis-softmaxed.
- **Model wrapper precision**: Removed unconditional `.half()` and `torch.compile(mode="max-autotune")` side effects. Both are now opt-in via `use_half_precision` and `compile_mode` constructor kwargs. Autocast uses the device's actual `device.type` (not hardcoded `"cuda"`) with bf16/fp16 chosen by capability.
- **Trainer module-level side effects**: TF32/cuDNN precision flags moved into a `_configure_tf32()` helper called from `Trainer.__init__` (only when CUDA is present). Importing the module no longer mutates global PyTorch state. `torch.autograd.set_detect_anomaly(False)` removed.
- **Trainer per-step sync**: Replaced per-step `raw_loss.item()` accumulation with an on-device tensor accumulator (`loss_accum`) in both `_train_epoch` and `_validate_epoch`. CPU sync now happens once per epoch in the final return.
- **Propaganda explainer safety**: Removed `self.model.zero_grad()` on the shared singleton. Switched to `torch.autograd.grad` with per-sample softmax-max sum as the target. Eval/train mode preserved and restored. Duplicate merged tokens now accumulate instead of producing `_1/_2` suffix keys.
- **Emotion explainer isolation**: Integrated Gradients now runs on a local `copy.deepcopy` of the model, preventing any eval/train-state mutations from leaking into the serving singleton. Falls back to the shared model only if deepcopy fails.
- **Aggregation pipeline**: Removed the duplicated `compute_scores` call pattern (now one raw pass + one weighted pass). Applied weights logged at DEBUG. Pre-clip saturation warnings added so upstream scale bugs surface in logs instead of being silently clamped.
- **Predict API singleton**: Lazy loader now guarded by `threading.Lock` with double-checked locking, preventing duplicate model loads under concurrent request threads. Removed the `predict.batch_predict = batch_predict` function-attribute monkey-patch.
- **Explanation aggregator cleanup**: Dropped unused `_safe_corr` static method (dead code — `ExplanationConsistency` already handles correlation).
- **Missing model artifacts**: Generated `models/truthlens_model/config.json` (derived from `roberta-base` with Fake/Real `id2label`/`label2id`) and full tokenizer files (`vocab.json`, `merges.txt`, `tokenizer_config.json`, `special_tokens_map.json`) so the model registry boots cleanly. Actual `pytorch_model.bin` weights still need to be trained before `/predict` produces meaningful scores.

## Third ML Systems Audit (April 2026 — checkpointing & training durability)
- **Atomic, durable saves**: `save_model` in `main.py` and both `_atomic_save` paths in `src/training/checkpointing.py` now write to a `.tmp` sibling, `os.replace` into place, then `fsync` the parent directory. NaN/Inf weight guards (`_validate_finite`) refuse to serialize broken state.
- **Drive sync hardening**: `main.save_model._sync_to_drive` (file-level, MD5+size) and `src/training/checkpointing._copy_to_drive` (tree-level, copytree → staging → backup → atomic swap) both retry up to 3 times, validate sizes, and **raise** on persistent failure instead of silently downgrading to a warning.
- **Checkpoint payload schema (C5)**: every checkpoint — file-based (`save_checkpoint`) and manager-based (`CheckpointManager.save_checkpoint`) — now carries `epoch`, `step`, `loss` (val_loss → train_loss → loss), `config`, `pytorch_version`, `model_state_dict`/`model`, optional `optimizer`, and optional `scheduler`. `load_checkpoint` returns the full payload so callers can restore optimizer/scheduler/loss/config.
- **PyTorch 2.6 weights_only fallback**: `_safe_torch_load` and `CheckpointManager.load_checkpoint` first try `weights_only=True`, then fall back to `weights_only=False` on `UnpicklingError` so legitimate metadata fields (`pytorch_version`, config dicts) load cleanly.
- **Trainer wiring**: `TrainerConfig.checkpoint_dir` defaults to `MODELS_DIR/checkpoints`. `Trainer.train` saves per-epoch, marks the best epoch via `step >= 1e9 + epoch`, and prunes to the latest 3 — best-marked checkpoints are preserved by `cleanup_old_checkpoints`. Removed duplicate `torch.compile` call.
- **Test set evaluation (M2)**: Trainer now builds a `test_loader`, runs `_evaluate_on_test` after the final epoch, and writes the report alongside training artifacts.
- **Gradient accumulation flush (M3)**: step counter starts at `-1`, optimizer/scheduler step on `(step + 1) % grad_accum == 0`, so the final partial batch flushes correctly.
- **Scheduler step (M4)**: `try: scheduler.step() except TypeError: scheduler.step(float(loss.item()))` for ReduceLROnPlateau compatibility.
- **TF32/SDP opt-in (M5)**: All TF32 / cuDNN / flash-SDP mutations are wrapped in `configure_training_precision()` in `trainer.py`, `training_step.py`, `train_transformer_model.py`, and `src/training/checkpointing.py`. Importing the modules no longer mutates global PyTorch state.
- **NaN-reset zero_grad (M6)**: training loop calls `optimizer.zero_grad(set_to_none=True)` after a non-finite loss to prevent stale-gradient pollution.
- **HF Trainer dedicated dir (M7)**: `HF_OUTPUT_DIR = MODELS_DIR / "hf_trainer"` separates HF Trainer checkpoints from custom checkpoints; `get_last_checkpoint(HF_OUTPUT_DIR)` is now well-defined.
- **CUDA gating (C8/M1)**: `cudnn.benchmark`, DataLoader `pin_memory`, and `collate_fn` pinning are all gated on `torch.cuda.is_available()` so CPU-only environments don't crash.
- **Async writer surfacing (m8)**: `AsyncCheckpointWriter.last_error` exposes the most recent worker exception; `CheckpointManager.save_checkpoint` polls and re-raises before queueing the next save, so silent background failures cannot accumulate.
- **Logging rotation (m3)**: `setup_logging` uses `RotatingFileHandler(maxBytes=50MB, backupCount=5, delay=True)` to bound disk usage.
- **Misc**: `TOKENIZERS_PARALLELISM=false`, env-driven model/log paths, redundant `torch.compile` removed.

## Senior ML Systems / GPU Audit (April 2026, second pass)
- **Token alignment signed-score corruption (CRITICAL)**: `src/explainability/token_alignment.py` ended with `np.clip(s, 0.0, 1.0)`, which silently destroyed every negative SHAP / IG / LIME attribution (and capped legitimate magnitudes >1). Removed the clip; final step now only neutralizes non-finite values and preserves sign + magnitude. `nan_to_num(posinf=1.0)` replaced with `posinf=0.0` so upstream Inf doesn't masquerade as a confident positive score. `max` aggregation switched to absolute-max so the largest-magnitude signal (and its sign) survives merging.
- **CheckpointManager writer bug (CRITICAL)**: `cleanup_old_checkpoints` called `self._writer.close()`, which permanently killed the async saver thread and broke every subsequent save. Replaced with `self._writer.flush()` — we still wait for in-flight writes before deleting, but the manager stays usable.
- **CheckpointManager wasted pinned memory**: `_to_cpu` called `t.pin_memory()` on state-dict tensors destined for `torch.save` → disk. Pinned memory is a scarce OS resource for async H2D transfer; pinning for disk I/O is pure waste and can OOM on large checkpoints. Now detaches to CPU only.
- **CheckpointManager dedup hash collisions**: `_hash_state` only hashed the first 10 elements of each tensor — collision-prone during fine-tuning where early params drift slowly. Now hashes shape, dtype, plus head and tail slices.
- **Inference engine module side-effect**: `src/inference/inference_engine.py` set `torch.backends.cudnn.benchmark = True` unconditionally at import time. Gated behind `torch.cuda.is_available()`.
- **Model wrapper stray pin_memory**: `ModelWrapper._move_to_device` called `pin_memory()` whenever the source was CPU, even when the target device was CPU — wasted pages and slower CPU→CPU path. Pinning is now gated on the target being CUDA; `non_blocking` is also gated so CPU-only deploys don't hit the pinned-memory code path at all.
- **Model wrapper logits key match**: `_extract_predictions` used `"logits" in key` substring match and `key.replace("logits", "")`, which could mis-match or mangle keys like `logits_norm`. Now matches `key == "logits"` or `key.endswith("_logits")` and uses a precise suffix strip.
- **Predictor redundant stable-softmax**: `_format_outputs` and `_extract_fake_probs` did `logits = logits - logits.max(...)` before `torch.softmax`. PyTorch's softmax is already numerically stable and internally subtracts the row-max. Removed the redundant pass in both sites.
- **Predictor base-name mangling**: `_format_outputs` used `key.replace("logits", "probabilities")`, which would corrupt any key where "logits" appeared elsewhere. Now uses a precise suffix strip (safe because the matching is `endswith("_logits")`).
- **Model registry double-name path bug**: `ModelRegistry.load_model` unconditionally appended `model_name` to `settings.model.path`, producing `models/truthlens_model/truthlens_model` and a `FileNotFoundError` whenever the configured path already pointed at the model dir. Registry now accepts either a per-model subdir or a direct model dir (detected via `config.json`).
- **Predictor accepts HF single-head classifiers**: `_extract_fake_probs` previously refused anything that didn't expose a `fake_logits` / `fakenews_logits` / `misinformation_logits` head, even when the loaded model's own `config.id2label` declared Fake/Real semantics. Added a guarded fallback: if the model emits plain `logits` and `id2label` contains a fake-label candidate, we use that tensor directly — honoring the actual model config instead of fabricating a head. Unrelated classifiers (no fake-label candidates) are still rejected.

## Key API Endpoints

### Core prediction
- `POST /predict` — Predict fake/real for news text (1-hour memory cache)
- `POST /batch-predict` — Predict for up to 50 texts at once; cache-aware
- `POST /analyze` — Full deep analysis (bias, emotion, narrative, graph, LIME)
- `POST /report` — Lightweight structured report (bias + emotion + credibility)

### Calibration
- `GET /calibration/info` — Describe available calibration strategies (temperature scaling, isotonic regression)
- `POST /calibration/metrics` — Compute ECE, MCE, Brier Score, NLL from probability + label arrays

### Ensemble
- `GET /ensemble/info` — Describe available ensemble strategies
- `POST /ensemble/predict` — Combine probability vectors from multiple models (average / weighted_average / majority_vote)

### Export
- `GET /export/info` — Describe ONNX, TorchScript, and quantization options
- `POST /export/onnx` — Export loaded model to ONNX (requires trained model)
- `POST /export/torchscript` — Export loaded model to TorchScript (requires trained model)

### Inference meta
- `GET /inference/model-info` — InferenceEngine model metadata (params, device, label map)
- `POST /cache/clear` — Clear the in-memory prediction cache

### Utility
- `GET /` — Index; lists all available endpoints
- `GET /health` — Detailed health check (model availability, cache size)
- `GET /project-view` — Project structure and configuration info
- `GET /docs` — Interactive Swagger API documentation

## Project Structure
```
api/          - FastAPI application (app.py)
src/          - Core source code
  analysis/   - Bias, narrative, propaganda, discourse analysis
  features/   - Feature engineering (bias, emotion, lexical, semantic)
  models/
    calibration/  - CalibrationMetrics, TemperatureScaler, IsotonicCalibrator
    ensemble/     - EnsembleModel, WeightedEnsembleModel, StackingEnsembleModel
    export/       - ONNXExporter, TorchScriptExporter, QuantizationEngine
  inference/  - InferenceCache, InferenceEngine, InferenceLogger,
                ResultFormatter, ReportGenerator, FeaturePreparer,
                ModelLoader, PredictionPipeline
  explainability/ - SHAP/LIME explanations
  graph/      - Entity/narrative graph analysis
  utils/      - Configuration, logging, device, time, JSON utilities
models/       - Trained model artifacts
  inference/  - predictor.py (predict, predict_batch)
  registry/   - ModelRegistry
config/       - YAML configuration files
tests/        - Pytest test suite
```

## Integrated Subsystems (api/app.py singletons)

| Singleton | Class | Purpose |
|---|---|---|
| `INFERENCE_CACHE` | `InferenceCache` | 1-hour in-memory cache for /predict, /analyze, /report |
| `INFERENCE_LOGGER` | `InferenceLogger` | Structured JSON inference event logging |
| `RESULT_FORMATTER` | `ResultFormatter` | Validates /predict response schema |
| `REPORT_GENERATOR` | `ReportGenerator` | Powers /report endpoint |
| `CALIBRATION_METRICS` | `CalibrationMetrics` | Powers /calibration/metrics |
| `ONNX_EXPORTER` | `ONNXExporter` | Powers /export/onnx |
| `TORCHSCRIPT_EXPORTER` | `TorchScriptExporter` | Powers /export/torchscript |
| `_INFERENCE_ENGINE` | `InferenceEngine` | Lazy singleton for /batch-predict, /inference/model-info |
| `_QUANTIZATION_ENGINE` | `QuantizationEngine` | Lazy singleton (fbgemm backend guard) |

## Important Notes
- Model must be trained before `/predict`, `/analyze`, and export endpoints work fully
- Health endpoint shows "degraded" when no model is trained — this is expected
- The tokenizer requires `sentencepiece` or `protobuf` — a pre-existing environment issue
- `models/inference/predictor.py` contains `predict` and `predict_batch` (the RoBERTa backend)
- Export endpoints return 503 when InferenceEngine cannot be initialised (missing model)
- All `src/` subdirectories have populated `__init__.py` files

## Running
```
python -m uvicorn api.app:app --host 0.0.0.0 --port 5000 --reload \
  --reload-dir api --reload-dir src --reload-dir config --reload-dir models
```

## Configuration
Main config: `config/config.yaml` — model paths, training params, API settings

## Replit Migration Notes
The following files were created during Replit migration to resolve missing modules:
- `graph_hardening_patch.py` — Graph utility helpers (normalize_graph_adjacency, to_undirected, spectral_eigen_embedding, etc.) used by all `src/graph/` modules
- `src/models/ensemble/ensemble_model.py` — EnsembleModel / EnsembleConfig (average & majority-vote strategies)
- `src/models/ensemble/weighted_ensemble.py` — WeightedEnsembleModel / WeightedEnsembleConfig
- `src/models/ensemble/stacking_ensemble.py` — StackingEnsembleModel / StackingEnsembleConfig

### Replit Agent → Replit migration (April 2026)
- Installed missing Python deps via pip: `portalocker`, `spacy`, `sentence-transformers`, `datasets`, `mlflow` (torch/transformers/fastapi/uvicorn already present).
- Added `src/analysis/integration_runner.py` with `AnalysisIntegrationRunner.analyze_text(text)` aggregating all 15 single-text analyzers.
- Added missing utility helpers to `src/utils/`: `device_summary` / `device_name` / `set_cuda_device` (`device_utils.py`), `ensure_non_empty_text_column` / `ensure_positive_int` (`input_validation.py`), `safe_mean` / `normalize_score` (`metrics_utils.py`).
- Rewrote `src/utils/settings.py` to load `config/config.yaml` as a nested AttrDict so dotted access (`SETTINGS.model.path`, `SETTINGS.paths.tfidf_vectorizer_path`, `SETTINGS.api.*`, `SETTINGS.inference.*`, `SETTINGS.training.text_column`) works without the legacy `tasks` requirement.
- Added missing schema key lists to `src/analysis/feature_schema.py`: `INFORMATION_OMISSION_KEYS`, `NARRATIVE_CONFLICT_KEYS`, `NARRATIVE_PROPAGATION_KEYS`, `NARRATIVE_TEMPORAL_KEYS` (registered in the schema registry).
- Made `ScoreExplainer.__init__` tolerate `model=None` / `tokenizer=None`, and added a profile-only `explain_profile(profile, top_k)` method so `AggregationPipeline` no longer crashes on construction.
- App listens on `0.0.0.0:5000`; the `Start application` workflow boots cleanly and `GET /` returns the endpoint index.

## Multi-Task Training Playbook (applied)
All four phases of the misinformation multi-task playbook + corrections are wired in
and ON by default. Set the corresponding env var to `0` to disable any one.

| Env var | Default | Effect |
|---|---|---|
| `TRUTHLENS_SKIP_EMPTY_BATCH` | `1` | `MultiTaskLoss` returns a zero loss instead of raising when every head is masked out (trainer skips the optimizer step). Set `0` for strict mode. |
| `TRUTHLENS_OVERSAMPLE` | `1` | Train loader uses `WeightedRandomSampler` with inverse-frequency weights on the bias label. Set `0` to use the length-bucketed sampler. |
| `TRUTHLENS_EMA_TASK_WEIGHTING` | `1` | `MultiTaskLoss` multiplies each task's static weight by `min(1/cov_ema, cap)` so under-supervised heads get gradient-weight boosts. |
| `TRUTHLENS_EMA_ALPHA` | `0.1` | EMA smoothing factor for per-task coverage. |
| `TRUTHLENS_EMA_FLOOR` | `0.05` | Minimum coverage used in the inverse — caps boost magnitude. |
| `TRUTHLENS_EMA_CAP` | `10.0` | Hard upper bound on the EMA multiplier. |
| `TRUTHLENS_TASK_BALANCER` | `1` | Attaches a Kendall-uncertainty `TaskBalancer` so per-task scaling is learned via log-variances. |

Always-on additions (no env flag needed):
- `MultiTaskLoss` accepts `pos_weight` per BCE/multi-label task via `from_task_settings`.
- `load_data()` logs a per-task / per-split label distribution audit (`[label-audit]`) and warns on >95% single-class collapse and zero-positive multi-label classes.
- `MultiTaskLoss` divides each task loss by its smoothed running mean **before** weighting/balancing so harder tasks (e.g. ideology at ~1.2) cannot drown out easier ones (~0.6) by raw magnitude (#8 of the playbook). Disable via `MultiTaskLoss.loss_normalization = False`.

Default-on (was opt-in) since the latest playbook items #7–9:

| Env var | Default | Effect |
|---|---|---|
| `TRUTHLENS_LOG_GRAD_NORMS` | `1` | Per-task gradient-norm probe on the shared encoder every `TRUTHLENS_GRAD_NORMS_EVERY` steps; warns when max/min ratio > `TRUTHLENS_GRAD_NORM_DOMINANCE_WARN`. Result is also fed into `HealthScore.grad_unfair`. |

`HealthScore` is now multi-dimensional (#9 of the playbook). The legacy "all-clear" path still scores 1.0, but the score now also subtracts weight for `low_coverage` (any task EMA coverage < 5%), `grad_unfair` (std/mean of per-task grad norms > 1.0), and `loss_unstable`, in addition to spike / spike_cluster / dominance / conflicts / silent_collapse.

Refinement-phase fixes (#10–12):

- **#10 Pooler bypass.** `TransformerEncoder` sets `config.add_pooling_layer = False` before `AutoModel.from_pretrained`, so RoBERTa never instantiates the random-init `pooler.dense` (we use raw CLS / configured pooling and never read `pooler_output`). On startup the encoder logs either `"Encoder pooler bypassed for <model>"` or, if the model class ignored the flag, a warning that the pooler module is still present.
- **#11 Anomaly-logging rate limit + severity gate.** Spike batch dumps are now gated by *both* a hard cap (`TRUTHLENS_MAX_DEBUG_DUMPS=20`) **and** a "log every Nth spike OR every major spike" rule. A spike is "major" when the raw task loss exceeds its EMA by `TRUTHLENS_MAJOR_SPIKE_RATIO` (default 3.0×); otherwise we only dump every `TRUTHLENS_SPIKE_LOG_EVERY` (default 10) spike. Warning lines still fire on every spike — only the on-disk `.pt` dumps are throttled.
- **#12 Pre/post-clip gradient visibility.** The trainer used to call `_grad_tracker.update()` *after* `clip_grad_norm_`, so per-parameter norms always read ≈ `max_grad_norm` (the famous "grad_norm always ≈ 1.0" illusion). The order is now: tracker → anomaly classify → clip → log `grad_norm pre=… post=…`. A separate warning fires when `pre_clip > max_grad_norm × TRUTHLENS_HIDDEN_EXPLOSION_RATIO` (default 5.0), surfacing instability that clipping was previously masking.
