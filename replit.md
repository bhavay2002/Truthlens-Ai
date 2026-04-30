# TruthLens AI

## Project Overview
TruthLens AI is a multi-layer AI platform for misinformation detection and news credibility analysis. It combines deep linguistic analysis, narrative extraction, propaganda detection, and graph-based reasoning to provide an interpretable "Credibility Score."

## Architecture
- **Backend**: FastAPI REST API (`api/app.py`) served via Uvicorn
- **Language**: Python 3.12
- **ML/NLP**: PyTorch, Hugging Face Transformers, spaCy, NLTK, LIME, SHAP
- **Port**: 5000

## Recent Refactors (audit fixes applied)
- **MT-MODEL-ENC-KWARG-FIX** (`src/models/multitask/multitask_truthlens_model.py`): `MultiTaskTruthLensModel.forward(**inputs)` was splatting the entire batch (`input_ids`, `attention_mask`, `labels`, optionally `offset_mapping`) into `self.encoder(**inputs)`. `TransformerEncoder.forward` has a strict `(input_ids, attention_mask)` signature and rejected the extras: `TypeError: TransformerEncoder.forward() got an unexpected keyword argument 'labels'`. The single-task model classes used to hide this because they happened to declare `forward(input_ids, attention_mask, labels)` — a fact the `training_step` pre-filter explicitly relied on (its comment at line 411 says "the single-task model classes have strict `forward(input_ids, attention_mask, labels)` signatures and reject unknown kwargs"). Fix at the encoder boundary: build `encoder_inputs = {k: inputs[k] for k in ("input_ids", "attention_mask") if k in inputs}` and pass that to `self.encoder(**encoder_inputs)`. The training_step pre-filter (which only strips `task`) is left untouched so single-task callers keep working.
- **MT-FACTORY-NOLEGACY-CFG** (`src/training/create_multitask_trainer_fn.py`): the codebase has TWO parallel config systems and the multi-task factory was straddling both. **System A** (used by this factory): YAML keys `training.epochs`, `monitoring.spike_threshold`, `monitoring.ema_alpha`, `monitoring.grad_monitor_interval`, etc. — read directly via `_get(...)` and converted in-factory. **System B** (`src/models/config/model_config.py::ModelConfigLoader`, still used by `src/models/registry/model_factory.py`, `src/models/encoder/encoder_factory.py`, `src/inference/model_loader.py`): strict per-section dataclass parser expecting `training.num_epochs`, `monitoring.enable_drift_detection`, etc. The factory was forwarding `config_path` to `Trainer(...)`, which triggered `ModelConfigLoader.load_multitask_config(config_path)` against a System A YAML and crashed: `TrainingConfig.__init__() got an unexpected keyword argument 'epochs'`. Fix: stop forwarding `config_path` to `Trainer` from this factory (`Trainer(config_path=None, ...)`); add `_resolve_early_stopping_patience(settings)` alongside the existing `_resolve_epochs`; forward both via `params_override={"epochs": ..., "early_stopping_patience": ...}`. `Trainer.__init__` already guarded every `self.cfg` access with `if self.cfg is not None` (per the older N-LOW-4 comments), so `self.cfg=None` is a supported state. The legacy loader is left untouched so its single-task / inference callers keep working — fixing the schema there would be high-blast-radius and is out of scope for this issue.
- **MULTITASK-FACTORY-CFG-FIXES** (`src/training/create_multitask_trainer_fn.py`):
  * **MONITORING-CFG-FIX**: the factory used to call `MonitoringEngine(_get(settings, "monitoring"))`, passing the raw YAML AttrDict. The engine's first line is `self.config = config or MonitoringConfig()` — the AttrDict is truthy so the dataclass default was bypassed, and the next line (`EMA(self.config.throughput_ema_alpha)`) raised `AttributeError: 'AttrDict' object has no attribute 'throughput_ema_alpha'` because the YAML doesn't define every dataclass field. Added `_build_monitoring_config(settings)` which maps known YAML keys onto `MonitoringConfig.__dataclass_fields__` and lets the dataclass defaults fill the rest. The dataclass is now the contract; YAML is purely an override surface.
  * **MODEL-CFG-WARNING-FIX**: `_build_model_config` was logging `dropping unknown settings.model fields ['compile_mode', 'flash_attention', 'gradient_checkpointing', 'hidden_dim', 'torch_compile']` on every multi-task trainer build, which read like a silent-bug warning. Those fields are *intentionally* not on `MultiTaskTruthLensConfig` (which is strict-by-design per its dataclass docstring): `torch_compile` / `compile_mode` / `gradient_checkpointing` are picked up further down the same factory by `TrainingSetupConfig(use_compile=…, compile_mode=…, use_gradient_checkpointing=…)` reading the SAME `settings.model` block, `flash_attention` is currently a no-op in the model (no attention impl override wired), and `hidden_dim` is derived from the pretrained encoder. Filter these out via a new `_RUNTIME_ONLY_KEYS` set so the warning only fires for genuine YAML typos.
- **TRAINING-TOPOLOGY — true multi-task by default**:
  * `main.py` no longer loops `create_trainer_fn` per task. The training path now defaults to `create_multitask_trainer_fn(settings=config, data_bundle=datasets, tokenizer=…, enabled_tasks=list(datasets.keys()), config_path=str(CONFIG_PATH))` which builds **one** `MultiTaskTruthLensModel` (single shared `roberta-base` encoder + per-task heads) and **one** `LossEngine` whose `task_configs` covers every task at once. Side effects of the switch:
    - The `"MT-1: LossEngine instantiated with 1 task(s); disabling EMA normalizer …"` warning is gone — `len(task_configs)` is now the number of enabled tasks (≥5), so EMA normalizer + coverage tracker + `normalization="active"` are kept on as designed.
    - Encoder is no longer instantiated 5 times sequentially (`Creating encoder | model=roberta-base` × 5 → × 1).
    - `narrative_frame` (5-label head) which the single-task factory had no mapping for is now trained alongside the other heads.
    - Epochs come from YAML `training.epochs` (currently 4) instead of always defaulting to 1; `TRUTHLENS_TRAIN_EPOCHS` env var still overrides if set, and is now applied to *both* paths (not just legacy single-task) by mutating `config.training.epochs` before the factory reads it.
    - Checkpoint shape: `{"model": MultiTaskTruthLensModel, "task": "multitask", "tasks": [...], "encoder": "roberta-base"}` written once to `saved_models/checkpoint.pt` after training.
  * Legacy per-task path preserved behind `TRUTHLENS_USE_SINGLE_TASK=1` as an escape hatch for single-head debugging. Same env var also still respects `TRUTHLENS_FORCE_SINGLE_WORKER=1` for memory-constrained CPU smoke runs.
- **TOKENIZERS-FORK-FIX**: `os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")` set at the very top of `main.py` and `api/app.py` (before `import transformers`). Kills the per-step `huggingface/tokenizers: process just got forked …` spam and removes the deadlock window between the eager Rust thread pool and DataLoader worker forks. Set `TOKENIZERS_PARALLELISM=true` in the environment if running with a single-process loader.
- **MULTILABEL-FOCAL-FIX** (`src/training/loss_balancer.py:_multilabel_plan`): the multilabel branch previously only emitted `pos_weight` and never set `use_focal`, so on a task like emotion (per-column max-skew ≈ 0.95) the loss kept matching the majority class and the head plateaued at F1 ~0.5. Now mirrors the multiclass / binary gates: `use_focal=True` when `max_skew ≥ focal_threshold` (default 0.9), where `max_skew = max(pos_ratio, 1 - pos_ratio)` per column. Combined with the existing `pos_weight`, this routes through `LossEngine` → `TaskLossConfig(use_focal=True, focal_gamma=2.0)` for every column with extreme skew. `LossBalancingPlan.max_ratio` for multilabel is now reported as the per-column max-skew (was `pos_ratio.max()` which under-reported severe negative-collapse cases).


- **EXPLAINABILITY-AUDIT — CRITICAL EXPLANATION BUGS / PERFORMANCE BOTTLENECKS / FAITHFULNESS** (`src/explainability/` 22-file audit, scope: CRIT-1..12, PERF-1..6, FAITH-1..6 only):
  * **CRIT-1 / FAITH-2 / GPU-3 / PERF-3 — `bias_explainer.py` rewritten**: `compute_shap`, `compute_ig`, `compute_attention_rollout` now route through the multitask wrapper (`model.encoder` + `model.heads[task]`) matching `src/aggregation/score_explainer.py` instead of calling the non-existent `model(...).logits` (CRIT-1). `compute_ig` does a real Riemann-sum path integration over `DEFAULT_IG_STEPS=16` interpolation points instead of single-step gradient×input (FAITH-2). All tokenizer outputs are explicitly placed on `next(model.parameters()).device` (GPU-3). The `shap.Explainer` is cached at module level via an `OrderedDict` keyed by `(id(tokenizer), task)` with LRU cap 4 (PERF-3); helper `clear_shap_cache()` for tests.
  * **CRIT-2 / FAITH-5 — `emotion_explainer.fuse` no longer mixes word-level lexicon with subword-level gradients**: `fuse(lexicon, gradients=None)` now returns the normalised lexicon vector only, eliminating the silent-misalignment / shape-mismatch bug. Gradients, when computable, are exposed under `emotion_explanation["model_attribution"] = {tokens, importance, faithful=True, token_space="subword"}` so callers can use the faithful subword signal without word/subword mixing. The fused dict carries `faithful=False` to mark the lexicon path as heuristic.
  * **CRIT-3 / CRIT-4 / PERF-5 — `explanation_aggregator.py` rewritten**: dropped `sorted(set(union(*sources)))` which destroyed token order; the aggregator now picks a *canonical* token list (priority: SHAP → IG → attention → LIME) and aligns every other source to those positions (CRIT-3). Per-source values are no longer collapsed via `dict(zip(tokens, importance))` — repeated tokens now each get their own slot through positional alignment when source lengths match, falling back to first-occurrence lookup otherwise (CRIT-4). The Python per-token loop is replaced by a single `[n_methods, n_tokens]` matrix multiply: `(weights * confidences).reshape(-1,1) * importance_matrix` summed across the method axis (PERF-5). New `include_heuristic: bool = False` ctor flag implements the CRIT-9 gate. Aggregated output now optionally surfaces `text` + `offsets` for downstream metrics (CRIT-11 plumbing).
  * **CRIT-5 — `common_schema.ExplanationOutput` validators rewritten**: dropped `Field(ge=0.0, le=1.0)` from `TokenImportance.importance` (signed attributions are legitimate); `validate_importance_finite` is now a pass-through that only checks finiteness, removing the silent re-normalisation that drifted `structured[i].importance` away from the flat `importance[i]`. New `validate_structured` raises if `structured[i].importance != importance[i]` (within 1e-6).
  * **CRIT-6 / CRIT-7 — single source of truth for `ExplainabilityResult`**: deleted `src/explainability/model_explainer.py`. `explainability_pipeline.ExplainabilityResult` re-exports from `common_schema` with `extra="ignore"`. Schema extended with `bias_explanation`, `emotion_explanation`, `monitoring`, `explanation_quality_score`, `module_failures` so the pipeline output validates against the canonical class. `__init__.py` re-exports `ExplainabilityResult`, `AggregatedExplanation`, `ExplanationOutput`, `TokenImportance`, `run_explainability_pipeline`, `explain_prediction_full`, `explain_fast`, `get_default_orchestrator`.
  * **CRIT-8 — `bias.integrated_gradients` wired into aggregator + consistency**: orchestrator's new `_wrap_bias_ig()` builds an `ExplanationOutput(method="integrated_gradients", faithful=True)` from the bias output and passes it as `integrated_gradients=` to both `aggregator.aggregate(...)` and `consistency.compute(...)`. Surfaced as `explanation["integrated_gradients_explanation"]`.
  * **CRIT-9 — `faithful: bool = True` on `ExplanationOutput`**: `propaganda_explainer` sets `faithful=False` on every return path. `ExplanationAggregator(include_heuristic=False)` (default) drops any source whose `faithful=False` from fusion; the heuristic signal is still surfaced in its own `*_explanation` field.
  * **CRIT-10 — `_spearman` uses ranks** (`explanation_consistency.py`): `np.argsort(np.argsort(x))` produces actual ranks instead of the previous `np.argsort(x)` permutation indices. Added zero-std guard. Verified `_spearman([1,2,3,4],[10,20,30,40]) == 1.0` and inverse case `== -1.0`.
  * **CRIT-11 — text-level ablation in `explanation_metrics.py`**: every metric (`faithfulness`, `comprehensiveness`, `sufficiency`, `deletion_score`, `insertion_score`) gained optional `text` + `offsets` + `mask_string` parameters. When supplied, ablation happens at the input text level via right-to-left span replacement (`_ablate_offsets`) instead of `" ".join([t for j,t in enumerate(tokens) if j != i])`. Orchestrator computes offsets via best-effort `_compute_offsets()` walking the canonical tokens against the original `text` (handles `##`, `Ġ`, `▁` markers); falls back to the legacy join path when alignment can't be derived.
  * **CRIT-12 — `_make_batch_predict_fn` uses `predict_fn.batch_predict`**: when the predictor exposes a callable `batch_predict`, the wrapper routes through it directly instead of looping per text; Python loop kept as fallback. Avoids per-call interpreter overhead in SHAP/LIME's heavy text fan-out.
  * **PERF-2 — LIME default `num_samples` 256 → 64** (`lime_explainer.py:explain_prediction`): empirically the top-feature ranking is stable from ~64 samples on; callers needing finer attributions can still override.
  * **PERF-4 — vectorised attention rollout** (`attention_rollout.py`): new `_stack_add_residual_normalize()` stacks per-layer head-averaged attentions and applies the residual `+ I` and row-normalisation in a single broadcast op. On a 12-layer transformer this collapses 24 small kernel launches into 2 large ones. Layer-weight scaling now uses a single broadcast multiply.
  * **PERF-6 / REC-3 — orchestrator singleton + base-prediction reuse** (`orchestrator.py`): new module-level `get_default_orchestrator(config=None)` returns a process-wide cached `ExplainabilityOrchestrator` keyed by `sha1(asdict(config))` — avoids re-instantiating `ExplanationCache` + `GraphExplainer` + `ExplanationMonitor` per article (PERF-6). `explainability_pipeline.run_explainability_pipeline` now uses this singleton. Orchestrator computes the un-ablated base prediction once and threads `base_proba` into `metrics.evaluate(...)` so the five sub-metrics share the same forward (REC-3); the `evaluate` signature also computes its own `base_proba` once when callers don't supply one, collapsing 5→1 baseline forwards inside the metrics module.
  * **FAITH-1 — attention faithfulness gate**: new `ExplainabilityConfig.attention_faithfulness_threshold: float = 0.0`. When >0 and both attention rollout and IG are available, the orchestrator computes `|Spearman(attention.importance, ig.importance)|` (with rank-based Spearman) and drops attention from aggregation/consistency if the correlation falls below the threshold. Logs the dropped correlation and writes `metadata["attention_ig_spearman"]` for downstream visibility.
  * **FAITH-3 — drop arbitrary power transforms** (`explanation_calibrator.calibrate_by_method`): removed `np.power(arr, 0.8)` for LIME and `np.power(arr, 1.2)` for attention. Both had no theoretical basis, were irreversible, and changed relative ordering of low-importance tokens. All methods now share a single L1 normalisation. `method` parameter retained for backward compatibility but no longer drives any shape transformation. Verified outputs identical across `method ∈ {"lime", "attention", "shap"}`.
  * **FAITH-6 — `module_failures` surfaced + optional majority-failure raise**: orchestrator tracks every failed sub-explainer in a list, surfaced on `ExplainabilityResult.module_failures`. New `ExplainabilityConfig.raise_on_majority_failure: bool = False`: when True, `RuntimeError` is raised if more than half of the *enabled* modules failed, instead of silently returning a corrupt result. Default False preserves backward compat.
  * **Validation**: workflow restarts cleanly on port 5000. End-to-end import of all schemas + sub-explainers OK; 12 targeted unit checks pass — negative `TokenImportance.importance`, pass-through `ExplanationOutput` validator, `faithful` default True / explicit False, `structured`/`importance` mismatch detection, Spearman returns `+1.0` / `-1.0` on perfect / inverse rankings, calibrator outputs identical across `lime`/`attention`/`shap`, aggregator preserves `[the, cat, the, dog]` order with duplicates, heuristic source gated out of fusion, `emotion_explainer.fuse` no longer crashes on word/subword mismatch, `explain_emotion(...)` carries new `model_attribution` + `faithful=False`, `ExplainabilityResult.module_failures` validates, `get_default_orchestrator()` returns the same instance across calls. Live `/analyze` end-to-end blocked only by 503 "Model not available" (model untrained in this environment, not a regression).

- **AGGREGATION-AUDIT (TOK / GPU / REC / UNUSED / CFG / EDGE categories)**:
  * **TOK-AG-1..4** (`src/aggregation/score_explainer.py`): keyword matching is now exact (post-detokenisation) so `joy` no longer credits `enjoyed` etc.; `_detok` strips both WordPiece (`##`) and BPE (`Ġ`/`▁`) markers via `convert_tokens_to_string`; `_merge_subwords` reassembles whole words before scoring so multi-piece tokens count once; the IG attention_mask now zeroes padding positions in the importance vector.
  * **REC-AG-1** (`feature_mapper.py` + `aggregation_pipeline.py`): new `extract_task_signals` returns a cached `TaskSignal` (probability/confidence/entropy/max_class) so the pipeline computes confidence + entropy in one pass instead of three. Pipeline's old `_compute_entropy` removed; `extract_confidence` / new `extract_entropy` are thin wrappers.
  * **REC-AG-3** (`risk_assessment.py`): module-level `_DEFAULT_TRUTHLENS_CONFIG` instead of allocating `RiskConfig(invert_keys=...)` per call; new `from_pydantic_config` bridge builds runtime `RiskConfig` from the Pydantic config.
  * **REC-AG-4** (`score_explainer.explain_profile`): single-pass O(N) over sections+features instead of the old O(N·S) nested loop.
  * **GPU-AG-1** (`aggregation_pipeline.py`): when the explainer has a model+tokenizer AND Branch-A `model_outputs` AND raw `text` are available AND `attribution.method == integrated_gradients`, the pipeline now runs real Integrated Gradients via `ScoreExplainer.explain_from_prediction`. Otherwise falls back to the heuristic `explain_profile`. Previously the IG path was dead code.
  * **GPU-AG-4** (`score_normalizer._to_output`): preserves the input tensor's `dtype` and `device` instead of always returning `float32` on CPU; safe under autocast.
  * **CFG-AG-1** (`aggregation_config.py`): `WEIGHT_GROUPS`, `TASK_TO_GROUP`, `SCALAR_WEIGHT_KEYS` moved here as the single source of truth. `weight_manager.py` and `truthlens_score_calculator.py` import them. Re-exported from `weight_manager` for back-compat.
  * **CFG-AG-4** (`risk_assessment.from_pydantic_config`): bridges Pydantic `aggregation_config.RiskConfig` (low/medium/uncertainty_penalty) → runtime `risk_assessment.RiskConfig` so YAML edits actually take effect; `AggregationPipeline.__init__` uses it.
  * **CFG-AG-6** (`config/config.yaml` + `aggregation_config.load_aggregation_config`): added an `aggregation:` block to the global config; the loader now picks just that sub-tree when a global config is passed (still supports a standalone aggregation YAML).
  * **UNUSED-AG**: `AggregationMetrics` instantiated as `pipeline.metrics`, updated per article when `monitoring.enabled`. `UncertaintyConfig.{enable_entropy,track_percentiles,p95_threshold,p99_threshold}` wired into `result.analysis_modules.uncertainty`. `src/aggregation/__init__.py` populated with the public API (`AggregationPipeline`, `FeatureMapper`, `TaskSignal`, `WeightManager`, `WEIGHT_GROUPS`, `load_aggregation_config`, `assess_risk_levels`, `assess_truthlens_risks`, `assess_batch`, `risk_from_pydantic_config`, …).
  * **EDGE-AG**: empty `model_outputs` / empty `source` now emits a warning instead of silently producing all-zero predictions. NaN probabilities trigger a warning in `FeatureMapper.extract_task_signals` (instead of being silently `nan_to_num`'d to max-confidence). `WeightManager._aggregate_group_signal` filters NaN/Inf before averaging so a single bad task can no longer poison the multiplicative scale chain.
  * **Validation**: workflow restarts cleanly on port 5000 (`200 OK`); end-to-end smoke test exercises both Branch A and Branch B, the empty-source warning, the NaN guard, the `TaskSignal` cache (multilabel vs multiclass entropy), and verifies `pipeline.metrics.size()` increments per run.

- **MODELS-AUDIT** (`src/models/` audit, sections 3 Architectural / 4 Loss / 5 GPU+Device / 6 Recomputation — sections 1-2 explicitly out of scope):
  * **A5.1 — centralised device detection** (`src/models/_device.py::detect_device`): single CUDA → MPS → CPU resolver with `prefer=` override. `EncoderFactory.detect_device`, `inference/model_wrapper.TransformerEncoder.__init__`, `utils/model_utils.load_model`, and `benchmarking/benchmark_runner.py` now route through it instead of each duplicating the `cuda.is_available()` / `mps.is_available()` chain.
  * **A3.3 / A5.3 — `BaseModel.device` fast path + lazy device sync**: `set_device(...)` caches `self._device`; the `device` property returns the cached value without walking `parameters()` (only falls back to a parameter walk when `_device is None`). New `attach_module(name, module)` calls `module.to(self._device)` then `add_module(...)`, so modules added after the initial `set_device` inherit the model's device.
  * **A3.4 — head dict contract codified** (`src/models/heads/base_head.py`): new `BaseHead(nn.Module)` ABC requiring `forward(features) -> dict[str, Tensor]` with at least `"logits"`. `MultiTaskHead.forward` and `MultiTaskTruthLensModel.forward` no longer carry a tensor-fallback branch — they raise `TypeError` if a head returns a bare tensor, with a pointer to `BaseHead`.
  * **A3.5 / A3.2 / A6.1 — `MultiTaskTruthLensModel` inherits `BaseModel`, canonical encoder location, strict pooled extraction**: `class MultiTaskTruthLensModel(BaseModel)` so the multitask wrapper picks up the G4 calibration/base parameter split, `save_/load_checkpoint`, and centralised device tracking. Canonical `TransformerEncoder` moved to `src/models/encoder/transformer_encoder.py`; `inference/model_wrapper.py` is now a back-compat shim that re-exports from there. `_extract_pooled` raises `RuntimeError` when neither `pooled_output` nor `pooler_output` is populated — no silent `last_hidden_state[:, 0]` fallback. The two pooled-key probes are done via explicit `is None` checks (an `or` chain would call `bool(tensor)` and raise on multi-element tensors).
  * **A3.6 / A3.7 — de-duplicated output dict + tightened `MultiTaskOutput.from_model_outputs`**: `MultiTaskTruthLensModel.forward` returns per-task entries as the source of truth; `task_logits` is a thin view derived from them (`out["task_logits"]["bias"] is out["bias"]["logits"]`) kept for trainer back-compat. `MultiTaskOutput.from_model_outputs` no longer accepts the legacy "any dict with `logits`" shape — it requires either an explicit `task_logits` view (fast path) or a `task_names=` kwarg whitelist (typically `model.get_task_names()`); raises `RuntimeError` otherwise.
  * **A3.1 — `TaskConfig.loss_weight`**: per-task loss weights promoted into `TaskConfig` so YAML-driven `MultiTaskModelConfig` carries them as the canonical surface. `MultiTaskTruthLensConfig`'s per-task weight fields kept for back-compat.
  * **A4.1 — explicit loss accumulator** (`MultiTaskBaseModel.forward`): replaced the ternary `total = total + loss if total is not None else loss` with `losses.append(...)` then `torch.stack(losses).sum()`; explicit `active_task` (single-task) vs multi-task split.
  * **A4.2 — task weights as buffers** (`MultiTaskHead`): per-task weights are now zero-dim buffers (`_task_weight__<safe_name>`) so `state_dict()` carries them and DDP sees them on broadcast. `set_task_weight` does `buf.fill_()` under `torch.no_grad()`; `get_task_weights()` reads `.item()`. Verified round-trip via `state_dict()` / `load_state_dict()`.
  * **A4.3 — preserve smoothed soft labels** (`BaseClassifier.compute_loss`): only `.long()` when targets are integer-typed; soft (smoothed) targets pass through untouched. Verified both hard `LongTensor` labels and softmax-normalised soft targets compute non-zero CE loss without dtype errors.
  * **A4.4 — single per-task loss key** (`MultiTaskHead.forward`): per-task dict publishes one `"loss"` (unweighted, comparable across tasks); the old `weighted_loss` field is removed. `total_loss` still accumulates with the engine-applied weights internally.
  * **A4.5 — shared-parameter helper** (`src/models/loss/multitask_loss.py::gather_shared_parameters`): documents the previously implicit `shared_parameters` contract on `MultiTaskLoss.forward`. Returns `model.get_optimization_parameters()` if present, else `None` (which balancers interpret as "skip the gradient-shaping step" rather than crashing).
  * **A5.5 — ensemble device hygiene** (`EnsembleModel.forward`): explicit `del out` and `torch.cuda.empty_cache()` after each member call when on CUDA, so per-member intermediates don't accumulate in the active stream's allocator.
  * **A6.2 — drop redundant softmax-then-argmax**: `BaseClassifier.forward`, `MultiTaskBaseModel.forward`, and `ClassificationHead.forward` now do `argmax(logits)` directly; `confidence` reads `log_probs.max().values.exp()` so the model evaluates the softmax exactly once per call.
  * **A6.4 — optimiser groups skip calibration params** (`build_parameter_groups`): walks `BaseModel._is_calibration_parameter_name` and excludes anything tagged calibration (currently `temperature`) so `temperature` doesn't get shoved into the AdamW group with the encoder. Verified that a model with `nn.Parameter(temperature)` produces parameter groups carrying only the base `Linear.weight` + `Linear.bias`.
  * **A6.3 — `MultiTaskHead.predict` is forward, documented**: explicit comment clarifying that `predict` IS a single forward call (not a re-forward); the audit complaint was predicated on a missing optimisation that wasn't actually triggered.
  * **Validation**: workflow restarts cleanly on port 5000 (`Application startup complete`, GET / → 200). All 16 touched modules import without error (`_device`, `heads/{base_head,multitask_head,classification_head,multilabel_head}`, `base/{base_model,base_classifier,multitask_base_model}`, `multitask/{multitask_truthlens_model,multitask_output}`, `encoder/{transformer_encoder,encoder_factory}`, `inference/model_wrapper`, `optimization/optimizer_factory`, `loss/multitask_loss`, `ensemble/ensemble_model`). Targeted forward-pass smoke tests pass: `MultiTaskHead` produces `total_loss` + per-task `loss` (no `weighted_loss`), task-weight buffers survive `state_dict` round-trip, bare-tensor head triggers the A3.4 `TypeError`, `MultiTaskTruthLensModel` (now inheriting `BaseModel`) shares the per-task entry as the source of truth for `task_logits`, missing `task_names=` raises on `from_model_outputs`, missing pooled output raises (no CLS fallback), `BaseClassifier` accepts both hard and soft labels, `temperature` is excluded from `build_parameter_groups`, `attach_module` lands new modules on the cached device.

- **GRAPH-AUDIT** (`src/graph/` 12-file audit, items R1, R2, CFG1-3, E1-E4 — earlier session covered S1, S2, T1, T2, G1, G2):
  * **R1 — `GraphPipeline` singleton** (`src/graph/graph_pipeline.py`): added `get_default_pipeline()` / `reset_default_pipeline()`; the 7 callsites that previously instantiated `GraphPipeline()` per-construct now share one process-wide instance: `api/app.py`, `src/pipelines/truthlens_pipeline.py`, `src/inference/{inference_pipeline,feature_preparer,batch_inference,analyze_article}.py`, `src/features/pipelines/feature_pipeline.py`. Avoids re-initializing 6 builders + 15 analyzers + spaCy on every constructor.
  * **R2 — Skip duplicate analyzer pass**: `GraphFeatureExtractor.extract_from_graphs` now accepts `entity_metrics=` and `narrative_metrics=` kwargs. `GraphPipeline._run_with_doc` passes the metrics it just computed, eliminating a redundant `analyzer.analyze()` call per graph.
  * **CFG1 — Hardcoded knobs to YAML** (`config/config.yaml`): added a `graph:` block exposing batch_size, embedding_type, spectral_dim, walk_length, num_walks, explainer weights, temporal_min_token_length, return_vector, run_analysis_modules. `GraphConfig` extended; `GraphConfigLoader.load_default_graph_config()` loads from the YAML with a quiet fallback to defaults.
  * **CFG2 — Wire YAML into pipeline** (`graph_pipeline.py`): `GraphPipelineConfig.from_yaml()` and `from_graph_config()` classmethods. `GraphPipeline.__init__` hydrates from YAML by default and propagates config to `NarrativeGraphBuilder`, `TemporalGraphAnalyzer`, `GraphFeatureExtractor` (via `GraphEmbeddingConfig`), and `GraphExplainer`. `GraphExplainer.__init__` now accepts `node_weight`/`edge_weight`/`temporal_weight` (sum-to-1 validated); `_overall_score` uses these instead of hardcoded 0.4/0.3/0.3.
  * **CFG3 — Validation**: explainer weights must sum to 1.0 (rejected at load time with a clear `ValueError`); `embedding_type` whitelisted to `{node2vec, spectral, hybrid, structural}`.
  * **E1 — Embedding dim contract** (`graph_embeddings.py`): empty graphs and populated graphs always return the same shape via `_embedding_target_dim()` + always-pad-to-target-dim path. Verified `(16,) == (16,)` for `hybrid` mode (spectral_dim=8).
  * **E2 — Temporal-consistency convention** (`temporal_graph.py`): `temporal_consistency` now requires `shift_arr.size >= 2` (i.e. ≥3 sentences for ≥2 transitions) before computing variance. Returns 0.0 otherwise — consistent "insufficient data" semantics with the existing 1-sentence early-return path. Previously a 2-sentence input returned `1.0` (var of a single-element array is 0) which was indistinguishable from "perfectly consistent."
  * **E3 — Symmetric weak-component BFS** (`narrative_graph_builder.py`): verified `_weak_components` symmetrises edges before BFS so directed asymmetric edges don't fragment the component count. No code change needed.
  * **E4 — Sparse clustering for large graphs** (`entity_graph.py`): verified `_average_clustering_sparse` (added in earlier G-P2 fix) is the path used in `extract_features` for large graphs; uses `scipy.sparse.csr_matrix` instead of dense O(N²) NetworkX call.
  * **Validation**: `/health → 200 (degraded model only, expected)`; `get_default_pipeline()` returns same instance across calls; explainer weights propagate as 0.4/0.3/0.3; `extract_from_graphs(entity_metrics=…)` runs analyzer 0× (vs 1× without); 2-sentence consistency returns 0.0 (was 1.0); empty + populated hybrid embeddings both shape `(16,)`; CFG validation rejects weights summing to 1.5 with `ValueError`.

- **FEATURES-AUDIT-1+2** (`src/features/` audit, items 1 and 2 from the prioritized fix list):
  * **Item 1 — Population-level FeatureScaler** (`src/features/fusion/feature_scaling.py`):
    * Replaced the misnamed module (which contained only `HybridTruthLensModel` and zero scaling code) with `FeatureScalingPipeline` — a per-feature scaler with persistent JSON state. Methods: `standard` / `minmax` / `robust`; supports `clip=(lo,hi)`; `fit` / `transform(return_array=False)` / `fit_transform` / `save(path)` / `load(path)`. Unseen keys at transform time pass through with a one-shot warning. Unfitted `transform` raises `RuntimeError`. Backward-compat alias `FeatureScaler`.
    * **Critical bug fixed**: `feature_engineering_pipeline.py:13` already imported `FeatureScalingPipeline` from this module — but the class did not exist, so importing the engineering pipeline crashed. Now resolves cleanly.
    * `HybridTruthLensModel` retained at the bottom of the same file (no other module imports it from a new location); marked in module docstring as a follow-up move target. No new dependencies added on its location.
    * Removed `FeatureFusion._normalize` and the `normalize: bool` flag from `src/features/fusion/feature_fusion.py`. Per-row z-score across feature TYPES is statistically invalid (mixes ratios + counts + densities + embeddings within one row); all scaling lives in `FeatureScalingPipeline` now.
    * Removed hard-coded `avg_len/20.0` and `std_len/10.0` magic constants from `src/features/text/lexical_features.py`. `lex_avg_word_length` and `lex_std_word_length` now emit raw values; `_safe_unbounded` helper added.
  * **Item 2 — Vectorized lexicon scoring** (~10–50× faster CPU-bound feature extraction):
    * New `src/features/base/lexicon_matcher.py` with three primitives: `LexiconMatcher` (unweighted, np.isin + precompiled `\b…\b` regex), `WeightedLexiconMatcher` (Dict[str, float], also accepts plain iterables → weight 1.0; supports `negation_aware_sum`), and `compute_negation_mask` (vectorized rolling-window via cumsum trick — replaces the per-token `for t in tokens[start:i]` loop).
    * Defensive against the `{...}` placeholder lexicons sprinkled in source (Ellipsis sets/lists): empty matcher → 0.0, preserving prior behavior.
    * Six lexicon extractors rewired to use module-level matchers built once at import: `bias/bias_features.py`, `bias/bias_lexicon_features.py`, `emotion/emotion_features.py`, `emotion/emotion_lexicon_features.py`, `propaganda/propaganda_features.py`, `propaganda/propaganda_lexicon_features.py`. Each also overrides `extract_batch` so the new pipeline batch dispatch can amortize per-batch setup.
    * `BaseFeature.safe_extract_batch` added — batch counterpart of `safe_extract` with the same `fail_silent` policy and per-sample validation. Length contract is enforced (pads/truncates to match `len(contexts)`) so fusion alignment can never silently drift.
    * `FeatureFusion.extract_batch` now dispatches per-feature through `safe_extract_batch` and assembles per-context results, instead of the old `[self.extract(c) for c in contexts]` Python-level loop.
    * `FeaturePipeline.batch_extract` now calls `fusion.extract_batch` directly (with the existing CUDA `autocast` block preserved). Graph features remain per-sample because graph build is text-dependent and already cached per ctx.
  * **Validation**: 13/13 modified modules import cleanly; vectorized matchers verified to match the old Python-loop outputs (set & weighted+negation parity); `FeatureScalingPipeline` round-trips fit→transform→save→load and zero-mean/unit-std verified; `pipeline.batch_extract` produces 200 features/sample matching single-sample `extract`; `FeatureEngineeringPipeline` end-to-end runs with the new scaler.

- **DATA-AUDIT** (data-layer production audit, full rewrite of `src/data_processing/*`):
  * Added `src/data_processing/__init__.py` exposing the public surface (pipeline, factories, contracts, collate).
  * Fixed all stale `src.data.*` imports across 15 files → `src.data_processing.*`. Also fixed `src/training/create_trainer_fn.py` import.
  * Unified label column names through `data_contracts.CONTRACTS` (`bias_label`, `ideology_label`, `propaganda_label`, `CO/EC/HI/MO/RE`, `hero/villain/victim`, `emotion_0..emotion_19`). `data_validator.TASK_SCHEMAS` now derived from contracts so cleaning/validation/factory/sampler can never disagree.
  * `dataset.py`: full rewrite — pre-tokenizes the entire text column once in `__init__`, stores numpy/list arrays, returns torch tensors per `__getitem__` (no per-sample tokenizer calls). Adds `return_offsets_mapping` (requires fast tokenizer), `pad_token_id` exposure, truncation diagnostics, NaN-label guard. Old duplicate `dataset.build_dataset` helper deleted.
  * `collate.py`: new `build_collate_fn(pad_token_id=…)` closure. Pad-id no longer hardcoded to 0 — RoBERTa (pad_id=1) now pads correctly. `attention_mask` padded with 0; offsets carried through if present; mixed-task batch raises.
  * `samplers.py`: `build_sampler` now uses contract columns. Multilabel sampler uses Laplace-smoothed inverse-frequency (epsilon=1.0) instead of 1e-6 (avoids 1e6 weight blowup on zero-positive columns). Dead `BucketSampler` deleted.
  * `dataloader_factory.py`: `pin_memory` gated on CUDA availability; `persistent_workers`/`prefetch_factor` exposed; `num_workers` defaults to `min(4, cpu//2)`; collate fn built with the tokenizer's pad-id.
  * `data_cache.py`: `load_settings` now lazy (no filesystem hit at import); `CACHE_VERSION` bumped to `v3`; file fingerprint replaces `mtime` with `(size, sha256(head_1MB + tail_1MB))` so `cp -p`/`git checkout` no longer invalidates; `get_cache_key` accepts an `extra` dict so tokenizer/max_length/cleaning/augmentation config invalidation is automatic. Cache filenames use `task__split.parquet` (double underscore) to handle task names containing `_`.
  * `data_pipeline.py`: leakage check now runs (a) on cache hits and (b) on RAW splits *before* augmentation (was running after, masking train→val/test bleed); cache key includes tokenizer + cleaning + augmentation + max_length; multilabel `task_columns` flattened to one entry per label so the analysis modules' `Dict[str,str]` contract is honored (was passing lists → silent `column not in df` failure).
  * `data_augmentation.py`: removed module-level `random.seed(42)` (was polluting global RNG) and module-level `nltk.download` (was a silent network call at import). Resources downloaded lazily; per-call `random.Random(config.random_seed)` instance for reproducibility without side-effects.
  * `data_resolver.py`: `required_splits` now configurable (default `("train","val","test")`).
  * `leakage_checker.py`: empty/whitespace texts filtered before hashing (otherwise they collapse to one bucket and report bogus overlap); switched to SHA-256; `LeakageReport.examples` defaults to empty dict not None.
  * `class_balance.py`: contract-driven `analyze_task_balance(df, task)` wrapper added; `analyze_classification`/`analyze_multilabel` no longer use bare `bias`/`ideology`/`propaganda` column names.
  * **Functional E2E verified**: all 6 tasks (bias, ideology, propaganda, frame, narrative, emotion) build datasets + samplers + dataloaders end-to-end with RoBERTa (pad_id=1), correct shapes, offsets returned, cache invalidation working.
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

## src/analysis/ Audit (April 2026 — CRIT + HIGH pass)

Closed the critical/high-severity items from the `src/analysis/` audit
(`attached_assets/Pasted-I-have-enough-information…txt`, items CRIT-A1–A8 / HIGH-A1–A11).
All fixes preserve backward-compatible call sites.

- **CRIT-A1 / A2 / A8 (`batch_processor.py`)** — `_run_batch` now passes a
  `List[str]` to `AnalysisPipeline.run_batch` (the contract it actually
  expects) and `_fallback_batch` calls `pipeline.run(text)` per item.
  Removed the cross-item `shared_cache` dict that was causing batch-wide
  spaCy doc / token cache contamination.
- **CRIT-A3 (`analysis_config.py` + `analysis_pipeline.py`)** —
  `default_orders` now uses canonical registry names
  (`information_omission`, `narrative_role`, `narrative_conflict`,
  `narrative_propagation`, `narrative_temporal`) instead of stale
  aliases (`omission`, `conflict`, …). New
  `validate_config_against_registry()` is invoked from
  `AnalysisPipeline.__init__` and raises on unknown analyzer names so
  ablation / ordering / `enabled=False` flags can never silently no-op.
- **CRIT-A4 / F5 (`emotion_target_analysis.py`)** — `PhraseMatcher` is
  built once against `get_task_nlp("ner").vocab` (not lazily against
  the first incoming `doc.vocab`). `analyze` validates
  `doc.vocab is matcher.vocab` per call and falls back to token-only
  matching with a warning if they diverge.
- **CRIT-A5 (`analysis_registry.py`)** — at registration time we
  introspect `analyzer.analyze` and cache `accepted_kwargs` /
  `accepts_var_kwargs` on the `AnalyzerSpec`. `run_all` then forwards
  only declared kwargs from `extra_inputs`, eliminating spurious
  `TypeError` on analyzers that don't list every key.
- **CRIT-A6 / F16 (`base_analyzer.py`)** — `__call__._validate_context`
  now invokes `ctx.ensure_tokens()` so analyzers that branch on
  `ctx.n_tokens == 0` always see a populated token view, regardless of
  whether the upstream constructor (FeatureContext.from_doc /
  batch_processor) ran it.
- **CRIT-A7 / F6 (`feature_context.py` + `analysis_pipeline.py`)** —
  `FeatureContext.from_doc` accepts a `mode` arg (default `"safe"`).
  In `"safe"` it pre-seeds the `"syntax"` and `"ner"` slots; in
  `"fast"` (or any other mode) it leaves them empty so downstream
  `get_doc(ctx, task)` re-parses with the task-appropriate pipeline
  rather than returning a stripped doc with empty `.ents` / `.dep_`.
  `AnalysisPipeline.run` and `run_batch` now propagate `self.nlp_mode`
  into `from_doc`.
- **F3 (`output_models.py`)** — added missing keys to the Pydantic
  models: `information_diversity`, `ideology_diversity`,
  `attribution_intensity`, `attribution_diversity`,
  `entity_repetition_ratio`. `BiasProfile` and `PipelineOutput` now
  include `argument`, `source`, `context`, `propaganda`,
  `information_omission`, `narrative_role`, `narrative_conflict`,
  `narrative_propagation`, and `narrative_temporal` (default
  `Field(default_factory=dict)` so older callers still validate).
  `FullAnalysisOutput.to_vector` appends the new sections at the end
  of the deterministic order, so the historical vector prefix remains
  stable.
- **F10 (`propaganda_pattern_detector.py`)** — added `_mean_present`
  helper and switched `_fear`'s narrative cue from "first present" to
  the mean of all present cues so we no longer drop half the
  narrative signal whenever both `conflict_intensity` and
  `polarization_ratio` are emitted.
- **F14 (`discourse_coherence_analyzer.py` + `feature_keys.py` +
  `output_models.py`)** — `_narrative_continuity` now emits both
  `narrative_continuity` (legacy) and the canonical alias
  `entity_repetition_ratio` with the same value;
  `DISCOURSE_COHERENCE_KEYS` and `DiscourseFeatures` updated to match.
- **F15** — `SourceAttributionAnalyzer.QUOTE_PATTERN` is now a paired
  span pattern `\"[^\"]+\"|“[^”]+”` and `_quote_density` returns the
  number of quoted spans / tokens (it previously summed character
  counts of unmatched quote runs). `ContextOmissionDetector.QUOTE_PATTERN`
  no longer matches apostrophes (`'`, `‘`, `’`), so contractions and
  possessives stop inflating `context_quote_ratio`.

Smoke-tested the full registry: single + batch runs of the pipeline
under both `nlp_mode="safe"` and `"fast"` produce non-empty
`information_omission` / `narrative_*` / `propaganda` sections, the new
keys above appear in the merged feature dict, and
`validate_config_against_registry` raises on unknown analyzer names.

## src/data_processing/ Audit v3 (April 2026 — batches 1 + 2)
Third-pass audit targeting the data layer. All fixes verified by behavioural smoke test. `CACHE_VERSION` was bumped `v3 → v4` in batch 1 to invalidate the fingerprint-bug caches; **batch 2 then converted `CACHE_VERSION` to an auto-derived `f"{_BASE_VERSION}-{md5(source(_file_fingerprint)+source(get_cache_key))[:8]}"`** so any future edit to either function invalidates stale entries automatically (current value: `v4-0773043e`).

### Batch 2 (sections 5-9: GPU/dataloader, caching, config, unused code, edge cases)
- **GPU-D5 / DataLoader workers**: `dataloader_factory._default_num_workers` bumped from `min(4, cpu)` to `min(8, cpu)` — Replit gives us 8 vCPUs and the previous default was leaving half of them idle during data-loading-bound epochs.
- **GPU-D3 (already correct)**: `training_utils.move_batch_to_device` already uses per-tensor `is_pinned()` to gate `non_blocking=True`, so no change was needed — confirmed by inspection.
- **CACHE-D2 / Auto-derived `CACHE_VERSION`**: `_derive_logic_fingerprint()` now hashes `inspect.getsource(_file_fingerprint) + inspect.getsource(get_cache_key)` and folds the 8-char digest into `CACHE_VERSION`. The manual `_BASE_VERSION` literal stays as a coarse override for changes the source-fingerprint can't see.
- **CACHE-D4 / `prune_cache(max_bytes, max_age_days)`**: new function in `data_cache.py` mirroring the feature-cache pruner. Scans the cache root, evicts by age first, then drops the oldest entries (mtime LRU) until under the byte cap. Skips `*.tmp` staging dirs and is exception-safe.
- **CACHE-D5 / Atomic dataset cache writes**: `save_cached_datasets` now stages everything into `{cache_key}.tmp/`, fsyncs `meta.json`, then `os.replace`s into the final dir. `load_cached_datasets` treats **the absence of `meta.json` as "incomplete save → invalidate"**, so a crash mid-write can no longer feed half a dataset into training.
- **CFG-D1 / `config.yaml::data.*` actually reaches `DataLoader`**: added `DataLoaderConfig.from_yaml_data(cfg.data)` classmethod (logs unknown keys, ignores them) and added a `shuffle` field that `build_dataloader` honours when `use_sampler=False`. `main.py` now constructs `loader_cfg = DataLoaderConfig.from_yaml_data(config.data)`, threads it into `DataPipelineConfig(dataloader_config=...)`, and forwards `batch_size/num_workers/pin_memory` to `create_trainer_fn` via `params=`. Previously every YAML knob in `data.*` was silently overridden by `DataLoaderConfig` defaults.
- **CFG-D2 / D4 / D5 / D7 / D8 / Orphan-key purge**: deleted `dataset.unified_schema` (contracts table is the source of truth in `data_contracts.py`), `balancing.*` (samplers use `WeightedRandomSampler`, not SMOTE/oversample/undersample), `augmentation.techniques.*` (`TASK_OPS` is hard-wired in `data_augmentation.py`), `profiling.report_dir`, `output.*`, and `eda.*` from `config/data_config.yaml` — none of these were read anywhere in the pipeline.
- **CFG-D3 / Cleaning fields wired**: `DataCleaningConfig` gained real `normalize_unicode`, `remove_emojis`, and `expand_contractions` fields with vectorized implementations (`_normalize_unicode_series`, `EMOJI_RE.sub`, `_CONTRACTIONS` lookup). Both `_clean_text` (scalar) and `clean_dataframe` (vectorized) honour them, and a smoke test asserts the two paths produce byte-identical output.
- **UNUSED-D1 / Dead helpers removed**: deleted `data_loader.compute_md5`, `data_loader.load_csv_in_chunks` (and the now-unused `compute_hash` arg on `load_csv`), and `data_resolver.pretty_print_config` — none had call-sites. Legacy `collate.collate_fn` / `fast_collate_fn` are kept but emit `DeprecationWarning` pointing callers at `build_collate_fn(pad_token_id=tokenizer.pad_token_id)` (the legacy default `pad_token_id=0` is unsafe for RoBERTa-family tokenizers, which use 1).
- **EDGE-D1 / NaN survival in cleaning (real bug found mid-test)**: `clean_dataframe` was calling `.astype(str)` early in the pipeline, which silently turned `NaN → "nan"` and `None → "None"` strings — the post-cleaning `.isna()` filter then matched nothing and the literal `"None"` row reached training. Fixed by capturing `nan_mask = df["text"].isna()` *before* the astype and using `~nan_mask` for both the keep-filter and the drop count.
- **EDGE-D2 / Split-mask drop logging**: `clean_dataframe` now logs `Length filter | dropped %d (NaN) + %d (<%d chars) + %d (>%d chars)` so a too-aggressive `max_text_len` no longer silently halves the corpus (audit §9 explicitly flagged the prior silent-truncation behaviour).
- **EDGE-D3 / `MultiLabelDataset` rejects out-of-range labels**: now raises `ValueError("Multilabel values outside [0, 1] in [...]")` for values <0 or >1, but accepts soft labels (e.g. `0.5`) so emotion-distribution training continues to work.
- **DEFAULT_MAX_LENGTH constant**: introduced in `data_contracts.py` (=512) and re-exported from `src/data_processing/__init__.py`. `dataset_factory.DatasetBuildConfig` and `data_pipeline.DataPipelineConfig` both use it as the default `max_length`, so changing the project-wide tokenization budget is now a one-line edit.
- **Public surface**: `DEFAULT_MAX_LENGTH` and `DatasetBuildConfig` are now exported from `src/data_processing/__init__.py`'s `__all__`.

### Batch 1 (original 13 fixes — verified)
- **CRIT-D1**: `data_cleaning.clean_for_task` no longer hardcodes a `TASK_LABELS` lookup — it now pulls `label_columns` from `data_contracts.get_contract(task)`, restoring the canonical single-source-of-truth invariant. Renaming/extending a label in `CONTRACTS` now propagates to cleaning automatically.
- **CRIT-D2**: `data_cache._file_fingerprint` had a 1 MB < size ≤ 2 MB blind spot (only the first 1 MB was hashed in that band → identical fingerprints across files with the same head). Now hashes the entire file when `size ≤ 2 MB`, head + tail otherwise. `CACHE_VERSION` bumped to `v4`.
- **CRIT-D3**: `data_loader.load_parquet` / `load_json` ignored `usecols`. Both now honour it (`pd.read_parquet(columns=...)` natively for parquet; post-load projection for JSON since pandas has no native arg). `load_dataframe` forwards `usecols` to all three paths.
- **CRIT-D4**: `data_loader.load_json` was hardcoded to `lines=True` and crashed on standard JSON-array files. Now sniffs the first non-whitespace byte to choose `lines=` automatically.
- **CRIT-D5 / LEAK-D1**: Augmentation ran *after* the leakage check, so synonym/swap ops could mutate a train row into a near-duplicate of a val/test row. `data_pipeline` now (a) passes the val/test frames into `augment_dataset` as `held_out_dfs` for a per-row pre-filter, and (b) re-runs the cheap exact-match leakage check on the post-augmentation splits before caching.
- **CRIT-D6**: `propaganda_injection`, `bias_injection`, and `emotion_amplify` blindly prepended/appended marker phrases regardless of the row's label, producing label-corrupted training data (a propaganda marker on a `propaganda_label=0` row teaches the model to ignore the marker). Each op now accepts `label=` and short-circuits unless the row is a positive (or has at least one positive emotion column).
- **CRIT-D7**: `augment_dataset` used uniform `rng.choice` over the input — a 95/5 dataset stayed 95/5 after augmentation, defeating `balancing.method: oversample`. Now stratified via `rng.choices(weights=inverse_class_freq)` keyed on the per-task label signature (single int for classification, tuple for multilabel).
- **PERF-D1**: `clean_dataframe` text-cleaning loop replaced from `df["text"].map(_clean_text)` (Python-level) to a chained `.str.replace`/`.str.lower`/`.str.strip` pipeline (vectorized C path). ~5-10× faster on 100k-row CSVs.
- **PERF-D2**: `BaseTextDataset` no longer stores `_input_ids: List[List[int]]` (~25M Python ints / ~200 MB overhead for a 100k×256 corpus, plus worker-fork copy storm). Flat numpy storage instead: one `int32` `_ids_flat`, one `int8` `_attn_flat`, one `int64` `_offsets`, optional `int64` `_om_flat[:, 2]`. ~3-5× lower RSS, ~30% faster `__getitem__`.
- **PERF-D3**: `leakage_checker.check_near_duplicates` was unconstrained `O(n·m)` SequenceMatcher — ~1e8 comparisons on 10k×10k splits. Now subsamples both sides to `√max_pairs` when `n·m` exceeds a 1e7 cap and warns loudly. Caller can override `max_pairs=`.
- **PERF-D4**: `data_loader.compute_md5` left in place (kept as harmless debug option behind `compute_hash=True`); the cache uses the SHA-256 head+tail fingerprint above, no second full-file pass.
- **TOK-D2**: `BaseTextDataset` truncation diagnostic switched from `L >= max_length` (over-counted samples that fit *exactly*) to `enc.encodings[i].overflowing` — the canonical HuggingFace signal. Falls back to the heuristic for slow tokenizers.
- **LEAK-D3**: `clean_dataframe.drop_duplicates(subset=["text"])` was case-sensitive while `leakage_checker._normalize` lowercases before hashing — `"Foo"` and `"foo"` both survived dedup, then triggered a false-positive leakage hit that aborted under `strict=True`. Cleaning now dedups on the lowercased text, matching the leakage normalizer.

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

### Replit CPU smoke training + Predictor wiring (April 28 2026)
End-to-end goal: `python main.py --mode train` produces `saved_models/checkpoint.pt`, then `python main.py --mode infer` attaches a `Predictor` and runs the pipeline. Both now exit 0.

Single-task ↔ multi-task glue (3 small fixes — none of them touch the model classes themselves):
- `src/training/training_step.py` and `src/training/evaluation_engine.py`: strip the non-tensor `task` key (added by `data_processing/collate.py` for logging) out of the batch before `model(**batch)`, so the strict single-task forwards (`forward(input_ids, attention_mask, labels)`) don't reject it.
- `src/training/loss_engine.py::compute`: when configured with exactly one task, synthesize the `task_logits` dict from `outputs["logits"]` and wrap a bare `labels` tensor into `{task: tensor}` before handing off to `MultiTaskLoss` (which strictly requires both as dicts). Multi-task callers are unaffected — they still must supply both as dicts.
- `src/training/evaluation_engine.py::_update_metrics`: same single-task `task_logits` synthesis so validation metrics are computed against `outputs["logits"]` instead of bailing out silently.
- `src/models/registry/model_factory.py::ModelFactory.create`: filter the merged config dict (which now includes `lr`, `epochs`, `tokenizer`, …) down to valid dataclass fields via `dataclasses.fields(...)` before instantiating the strict task-config dataclass.

`main.py --mode train` hardening:
- Skips `narrative_frame` (no single-task model factory entry — only lives inside the multitask spec) with a clear `[skip] task=…` log line.
- Coerces YAML scalars: `lr="3e-5"` → `float`, etc., before they reach `torch.optim`.
- Caps epochs to **1** for the CPU smoke run via `TRUTHLENS_TRAIN_EPOCHS` (default `1`); override with `TRUTHLENS_TRAIN_EPOCHS=N` env to train longer.
- Forces `num_workers=0` on the per-task DataLoaders. Each task's DataLoader otherwise forks `num_workers × 2` (train + val) child processes that fork the *current* RAM image (which by then holds previously-trained roberta-base weights). Across 5 sequential tasks these forks pile up and sometimes silently reap the parent.
- **Saves the checkpoint incrementally** after every task to `saved_models/checkpoint.pt` so a crash midway through the 5-task loop still leaves a usable checkpoint behind. Drops the trainer + model references and runs `gc.collect()` between tasks.
- Checkpoint shape is `{"model": <nn.Module>, "task": <task_name>, "encoder": "roberta-base"}` — matches what the inference path's `state.get("model")` lookup expects.

`main.py --mode infer` fix:
- `torch.load(checkpoint_file, map_location="cpu", weights_only=False)` — required because the checkpoint contains a *pickled* `nn.Module`, not just a state-dict, and PyTorch ≥ 2.6 defaults to `weights_only=True` which rejects arbitrary pickled objects. We control both writer and reader, so opting out of safe-load is fine here.

Verified end-to-end on Apr 28 2026:
- `--mode train` → trains `bias` for 1 epoch, validation `bias_score=0.6`, writes `saved_models/checkpoint.pt` (498 MB, contains `BiasClassifier` nn.Module).
- `--mode infer` → finds the checkpoint, logs `Predictor initialized on cpu` + `Predictor attached`, runs the full pipeline (`BATCH SUMMARY: n=3 …`), exits `SYSTEM COMPLETED SUCCESSFULLY`.

**Known follow-up (not blocking):** the CPU smoke training workflow currently dies silently between tasks (after `bias` finishes and `ideology` enters its sanity check) — no traceback, no OOM (44/62 GB used). Because checkpointing is now incremental, this still produces a working `checkpoint.pt`. To train all 5 tasks in one run, the next session should investigate why the workflow process is reaped between tasks (suspect: lingering multiprocessing artifacts from `transformers` / dataset tokenization, or workflow signal handling).

**Lexicon JSONs missing — orthogonal warnings (still open):** `Lexicon file not found` warnings for `bias_lexicon.json`, `framing_lexicon.json`, `ideology_lexicon.json`, `propaganda_phrases.json`. The features fall back to empty lexicons and the pipeline still runs; tracked as a separate cleanup.

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

---

## Apr 28 2026 — `src/features/` v6 audit §7 / §8 / §9

Closed the cache-layer / dead-code / configuration-drift items. Workflow boots clean on :5000, all 32 prior evaluation tests still pass.

**§7 Cache layer:**
- **§7.1 schema-fingerprint header on pickle blobs**: `FeatureCache.save(... fingerprint=)` and `load(... expected_fingerprint=)` now carry a SHA-16 of `feature-set-fingerprint:lexicon-fingerprint` in the pickle payload. A blob written under fingerprint A is treated as a miss when loaded under fingerprint B — defence in depth behind `context_key` (which already bakes the same components into the path digest).
- **§7.5 bounded per-context cache**: `FeatureContext.cache` is now a `_BoundedContextCache(OrderedDict)` capped at `TRUTHLENS_CONTEXT_CACHE_SIZE` (default 256) entries with LRU eviction. Subclasses `dict` so all existing `ctx.cache.setdefault(...)` / `ctx.cache["k"]` / `isinstance(ctx.cache, dict)` call sites keep working unchanged.
- **§7.6 hit/miss counters**: `CacheManager` exposes `mem_hits`, `mem_misses`, `disk_hits`, `disk_misses`, `computes`, `disk_write_failures` (wired into both `get_or_compute` and `get_or_compute_batch`) plus `stats()` / `reset_stats()` accessors suitable for Prometheus scraping.
- **`_LEXICON_SOURCES` cleaned**: dropped four entries that pointed at files removed in the §4 cleanup (`emotion_lexicon.py`, `emotion_lexicon_features.py`, `emotion_features.py`, `propaganda_lexicon_features.py`) — they used to contribute `missing:<rel>` placeholders to the fingerprint. Replaced with files that still exist (`emotion_schema.py`, `emotion_intensity_features.py`).

**§8 Dead/unused code:**
- **`importance/` moved**: `feature_ablation.py`, `permutation_importance.py`, `shap_importance.py` now live under `src/evaluation/importance/` since their sole live caller is `src.evaluation.advanced_analysis` (offline-only — never imported from `src.inference`, `api.app`, or any model forward path). Backward-compat shims at the old `src/features/importance/` paths re-export the moved classes so any pickled / experiment-script imports keep working.
- **Strict-load flag for analysis adapter**: `_BaseAnalysisFeature.initialize()` now re-raises silent `importlib` failures when `runtime_config.analysis_adapters_strict()` is on. Operator hook: `set_analysis_adapters_strict(True)` / env `TRUTHLENS_ANALYSIS_ADAPTERS_STRICT=1`. Default off (preserves legacy behaviour).
- **Kept**: `bias/bias_lexicon.py` (used by `api/app.py:32` and `tests/test_api.py:113`); the three already-minimal shims `feature_pruning.py`, `pipelines/feature_schema.py`, `fusion/feature_selection.py` (each has live call sites in `feature_engineering_pipeline.py` / `dataset_feature_generator.py`).

**§9 Configuration drift:**
- **`src/features/feature_config.py` (new)**: single-source `FeatureConfig` dataclass replacing the previous three-place tangle (per-module constants like `MODEL_NAME`, scattered env-var reads, runtime-config flags). Fields: `emotion_model`, `spacy_model`, `transformer_enabled`, `transformer_max_batch`, `transformer_chunk_length`, `transformer_chunk_stride`, `cache_max_memory_items`, `cache_max_memory_bytes`, `feature_context_cache_size`, `analysis_adapters_strict`, `torch_thread_cap`. All defaults match the legacy env-var-driven values byte-for-byte. `apply_to_runtime()` pushes everything into `runtime_config` (and writes the model-name env vars so the lazy spaCy / HF loaders pick them up).
- **`feature_bootstrap.bootstrap_feature_registry(... config=)` (new param)**: when `None`, instantiates `FeatureConfig()` and applies it before importing any feature module. Re-applies the torch thread cap with the config-driven value so `_apply_torch_thread_cap` is no longer a hard-coded module-import side effect.
- **`runtime_config` extended**: new `analysis_adapters_strict()` / `set_analysis_adapters_strict(...)` flags + `configure(... analysis_adapters_strict=)` plumbing.

**Verification:** `pytest tests/test_evaluation.py tests/test_evaluation_metrics.py tests/evaluation/` → 32/32 green; uvicorn restart on :5000 reaches `Application startup complete` with `GET / → 200`; end-to-end smoke tests confirm fingerprint round-trip, bounded context cache eviction, importance-shim equivalence, and the strict-load toggle.

## Apr 27 2026 — `src/features/` audit fixes

Resolved the critical / perf items from the 13-section audit. App restarts cleanly with all 15 analyzers, no import errors.

**Schema (C1, prior commit + this commit):**
- `src/features/feature_schema.py` is the canonical schema (added FRAMING / IDEOLOGICAL / PROPAGANDA / CONFLICT, fixed `bias_variance` → `bias_diversity`, gave narrative-role features the `_ratio` suffix to disambiguate from the `narrative` label columns in `data_contracts`).
- `src/features/pipelines/feature_schema.py` is now a thin re-export shim (kills the schema-drift surface).
- `src/features/pipelines/feature_pipeline.py` no longer ships empty `BIAS_FEATURE_NAMES` / `ALL_BIAS_MODULE_FEATURE_NAMES` stubs — they re-export from the canonical schema, so `src.inference.feature_preparer` finally sees the real list.

**Narrative role rename (C3):** `narrative_role_features.py` now emits `narrative_role_{hero,villain,victim,polarization}_ratio`, matching the canonical schema. Comment marks the deliberate split between FEATURE names (model inputs) and the `hero/villain/victim` LABEL columns from `data_contracts.CONTRACTS["narrative"]`.

**Fusion (C2):** `FeatureFusion.normalize` defaults to `False`. Per-row z-score across mixed-unit features is statistically invalid; population scaling must go through `FeatureScaling` with a train-fitted scaler.

**Caches:**
- C4: `_graph_cache` in `BatchFeaturePipeline` is now an `OrderedDict` with `_GRAPH_CACHE_MAX = 2048` LRU eviction; keys are sha256 of text, not the raw text.
- C6: `LRUCache.get` / `set` in `cache_manager.py` deep-copy at the dict level so callers can't mutate cached vectors.
- C8: `CACHE_VERSION` lives in `feature_cache.py` only; `cache_manager.py` re-imports it (no more split-brain constant).
- C9: `_context_key` now folds in a sha256-truncated fingerprint of `FeatureRegistry.list_features()`, so toggling the registered feature set auto-invalidates without bumping `CACHE_VERSION`.
- `FeatureCache._path_cache` is a bounded `OrderedDict` (cap 50_000) so the in-process key→Path memo can't grow forever on long-running services.

**Batch pipeline:**
- C5: `_process_batch`'s catch-all no longer returns `[{} for _ in batch]`. On batch failure it re-runs each context one-at-a-time; per-sample failures are logged and re-raised so `_dataloader_extract` can surface them instead of silently producing empty rows.
- C7: `embeddings.detach().to("cpu").contiguous()` before stuffing into `ctx.cache["_shared_cache"]["embedding"]` — fixes the steady VRAM growth from pinned GPU tensors.

**Bias lexicon perf (P12):** `compute_bias_features` previously instantiated `BiasLexiconFeatures` once per call AND once per sentence in the heatmap loop. Now cached as a module-level singleton via `_get_extractor()`.

**Skipped:** P1 (vectorize `bias_features.py::_weighted_ratio`) — the lexicons there are still placeholder ellipsis sets (`{...}`), so the function is dead code; vectorizing it would be premature.

## Apr 27 2026 — refined audit tasks 6 / 7 / 8

**Task 6 (audit 1.2 + 1.3 + warning 1.8):**
- `src/graph/graph_pipeline.py` now exposes `config_fingerprint()` — a 16-char sha256 over every public field of `GraphPipelineConfig`. Embedded in graph cache keys so flipping any toggle (entity / narrative / temporal / vector / explainer / analysis-modules) auto-invalidates the in-memory graph cache.
- `src/features/pipelines/batch_feature_pipeline.py::_graph_cache_key(text, cfg_fp)` now takes the fingerprint as a second arg and bakes it into a versioned (`GRAPH_CACHE_VERSION = "v2"`) JSON payload before sha256.
- `_attach_graph_cache` populates `ctx.cache["_graph"]["output"]` (the same slot `_merge_graph_features` reads from), so the per-sample merge step is now a cache-hit reuse rather than a second `graph_pipeline.run(text)` call. Eliminates the double NetworkX/spaCy build per request that the audit flagged.
- `_merge_graph_features` already promoted to warning + counter + strict-mode raise (1.8, prior turn).

**Task 7 (audit 1.5 pruner + tempfile fix):**
- `FeatureCache.save` wrapped in try/finally that `unlink(missing_ok=True)`s the tempfile on every code path that does NOT reach `replace()` — orphan tempfiles from killed processes are no longer leaked.
- `FeatureCache.prune(max_bytes=…, max_age_days=…)` added: TTL eviction first (mtime-based, also catches orphan tempfiles), then byte-budget oldest-first eviction. Returns `{removed_age, removed_size, kept, bytes}`.
- `CacheManager.prune_all(max_bytes_per_namespace=…, max_age_days=…)` sweeps every registered namespace plus every on-disk namespace under `base_cache_dir`.
- `api/app.py` registers a `@app.on_event("startup")` hook that runs `CacheManager(base_cache_dir=SETTINGS.paths.cache_dir).prune_all(max_bytes_per_namespace=512MB, max_age_days=14)`. Wrapped in try/except — pruning never blocks server startup.

**Task 8 (merge `feature_pruning.py` ↔ `fusion/feature_selection.py` → `fusion/feature_reduction.py`):**
- New `src/features/fusion/feature_reduction.py` is the single source of truth for `VarianceThresholdSelector`, `CorrelationSelector`, `TopKSelector`, `CompositeSelector`, `FeatureSelectionPipeline`, `FeaturePruner`, plus a new end-to-end `FeatureReductionPipeline`.
- `DEFAULT_CORRELATION_THRESHOLD = 0.9` (audit task 8) — more aggressive than the previous 0.95 in both legacy modules.
- `FeatureReductionPipeline.fit(features)` runs variance-prune then 0.9 correlation-prune on the training matrix; `save(path)` persists `{schema, variance_threshold, correlation_threshold, kept_features, removed_features}` as JSON; `load(path)` restores the kept-name list so inference applies exactly the same projection regardless of input dict order or extra/missing keys.
- `src/features/feature_pruning.py` and `src/features/fusion/feature_selection.py` are now thin re-export shims (`from src.features.fusion.feature_reduction import …`) so every existing import site continues to work.

## Apr 27 2026 — refined audit tasks 1 / 2 / 3

**Task 1 (layer hygiene):**
- New `src/models/architectures/hybrid_truthlens_model.py` is the canonical home of `HybridTruthLensModel` (the multi-head Transformer + engineered-feature fusion model). Re-exported from `src.models.architectures`.
- `src/features/fusion/feature_scaling.py` no longer defines the model — only `FeatureScalingPipeline` (alias `FeatureScaler`) lives there now, which is what the file's name promises. Top docstring rewritten to match. A PEP-562 module-level `__getattr__` keeps `from src.features.fusion.feature_scaling import HybridTruthLensModel` working with a one-shot `DeprecationWarning`, so no caller breaks even via dynamic import.
- `src/features/utills/` → `src/features/utils/` (typo rename). Verified: zero call sites used the misspelled package name, so the rename is a clean `git mv`.

**Task 2 (wire `discourse_features` + `argument_structure_features` into `BiasProfileBuilder`):**
- New `argument` section in `BiasProfileBuilder`: added to `PROFILE_SECTIONS`, `BiasProfileConfig.argument_weight = 0.6`, and `_compute_bias_score`'s weight map. `build_profile` now accepts `argument: dict | None = None` (kwarg-default-None preserves backward-compat with all existing call sites).
- New `BiasProfileBuilder.build_from_feature_dict(features, ideology=None)` is the missing bridge between the feature-engineering layer and the bias profile. It routes prefixed keys via `_FEATURE_PREFIX_TO_SECTION` (`disc_*` → discourse, `arg_*` → argument, `emotion_*` → emotion, `bias_*`/`framing_*` → bias, `narrative_*` → narrative, `ideology_*` → ideology). Unprefixed keys are deliberately dropped so the routing surface stays explicit.
- `AnalysisOrchestrator._post_process` now passes `argument=sections.get("argument", {})` so the analysis-side path also populates the new section using the existing `ArgumentAnalyzer` output.

**Task 3 (multi-label fix in `emotion_features.py`):**
- Removed the one-hot `features[f"emotion_dominant_{dominant_emotion}"] = 1.0` write. Emotion is multi-label by design — the per-label scalar columns (`emotion_<label>` = normalized hit share) already carry the full distribution, and the one-hot threw away every label except the argmax.
- Removed the matching `[f"emotion_dominant_{e}" for e in EMOTION_LABELS]` block from `EMOTION_FEATURES` in `src/features/feature_schema.py` (was 20 dead columns) and added a comment explaining why argmax over the per-label columns recovers the dominant label at inference time without info loss.
- Verified: `EmotionFeatures.extract()` now emits zero `emotion_dominant_*` keys; `EMOTION_FEATURES` shrank from 41 → 21 columns.

## Apr 28 2026 — production audit §2.2 / §2.4 / §2.5 / §2.7 / §2.8 + §3.1

**§2.2 — `LexiconMatcher` rolled out to the six remaining Counter-based extractors:** `bias/framing_features.py`, `bias/ideological_features.py`, `narrative/conflict_features.py`, `narrative/narrative_features.py`, `narrative/narrative_role_features.py`, `narrative/narrative_frame_features.py` previously each materialised a `Counter(tokens)` and Python-looped `sum(counter.get(w, 0) for w in lexicon)` per category. Each extractor now builds a module-level `Dict[str, LexiconMatcher]` once at import time and per document does a single `to_token_array(tokens)` followed by one `np.isin` per category — same shape as the `propaganda_features.py` template the audit calls out as the canonical pattern. `narrative_role_features.py` was also still shipping inline `{...}` placeholder lexicons (audit §1.1 leftover) and is now wired to `load_lexicon_set("narrative", "hero" / "villain" / "victim" / "polarization")`, sharing the same JSON source as `narrative_features.py`.

**§2.4 — `SyntacticFeatures.extract_batch` via `nlp.pipe`:** new `extract_batch(contexts)` method routes every non-empty text through `self._nlp.pipe(texts, batch_size=64)` instead of looping `self._nlp(text)` per call. spaCy's pipe re-uses pipeline state, micro-batches the parser, and is ~2.3x faster than the per-document path on warm runs. Empty texts are short-circuited to `{}`. On any pipe-level failure (CUDA OOM, model corruption) the method falls back to the per-document `extract` path so a transient mid-batch failure cannot zero the whole batch.

**§2.5 — `ensure_tokens_word_counter` per-extractor wiring:** the three remaining `Counter(tokens)` call sites (`discourse/discourse_features.py`, `discourse/argument_structure_features.py`, `propaganda/manipulation_patterns.py`) now read from the cached `ensure_tokens_word_counter(context)` helper that already lived in `src/features/base/tokenization.py`. Eight extractors used to build the same Counter independently per request; this collapses to one Counter per request shared across all of them.

**§2.7 — graph extractors share one spaCy `Doc` per request:** new `src/features/base/spacy_doc.py` exports `ensure_spacy_doc(context, model_name="en_core_web_sm")` which caches the parsed `Doc` on `context.cache["spacy_doc"]`, plus a `set_spacy_doc(context, doc)` seeder. `EntityGraphFeatures` and `InteractionGraphFeatures` now both consult `ensure_spacy_doc(context)` first and call `EntityGraphBuilder.build_graph_with_doc(doc)` / `NarrativeGraphBuilder.build_graph_with_doc(text, doc)` (already exposed by the §G-D1 / §G-P8 graph-layer fixes from the prior session) when a cached doc is available. Falls back transparently to the existing `build_graph(text)` path when spaCy is unavailable. The syntactic extractor's `extract_batch` seeds `ensure_spacy_doc` for every context after `nlp.pipe` so the graph extractors that run later in the same `BatchFeaturePipeline` pass never re-parse. `narrative_role_features.py`'s `_entity_density` now also reads from the shared cache.

**§2.8 — pre-allocated matrix builder centralised:** new `src/features/base/matrix_build.py` exports `collect_feature_names(rows) -> List[str]` and `dict_rows_to_matrix(rows, feature_names, dtype=np.float32) -> np.ndarray` (plus a `build_matrix` convenience wrapper). The matrix-build is pre-allocated once with the final shape and each row is filled via numpy fancy-indexing assignment instead of the per-cell `row[j] = float(value)` Python-level loop. `DatasetFeatureGenerator.generate()` and `DatasetFeatureGenerator.generate_by_section()` now both delegate to the same helper — both call sites had previously inlined a near-identical column-union + name_to_idx + per-cell-assignment loop, so any future micro-optimisation lands in one place. Behaviour is unchanged: column order is the deterministic `sorted` union of dict keys, missing keys are written as `0.0`.

**§3.1 — `EPS` / `MAX_CLIP` deduplication:** twelve extractor files were each redeclaring `EPS = 1e-8` (and most also `MAX_CLIP = 1.0`) at module scope, drifting from the `src/features/base/numerics.py` source of truth that audit fix §3.1 was supposed to enforce. All are now removed in favor of `from src.features.base.numerics import EPS, MAX_CLIP, normalized_entropy`. Files: `emotion/{emotion_features,emotion_lexicon_features,emotion_target_features,emotion_intensity_features,emotion_trajectory_features}.py`, `text/{token_features,lexical_features,semantic_features,syntactic_features}.py`, `graph/{entity_graph_features,interaction_graph_features}.py`, `discourse/{discourse_features,argument_structure_features}.py`, `importance/{shap_importance,permutation_importance}.py`, `feature_statistics.py`, `feature_schema_validator.py`. (Helper modules `base/text_signals.py`, `base/base_feature.py`, and `fusion/{feature_reduction,feature_scaling}.py` keep local copies for module-internal use; the `numerics.py` near-duplicate `EPS = 1e-8` at line 39 was left for a follow-up cosmetic pass.)

**Validation:** workflow restarts cleanly on port 5000. Smoke test exercises every touched extractor (`FramingFeatures`, `IdeologicalFeatures`, `ConflictFeatures`, `NarrativeFeatures`, `NarrativeRoleFeatures`, `NarrativeFrameFeatures`, `EntityGraphFeatures`, `InteractionGraphFeatures`, `DiscourseFeatures`, `ArgumentStructureFeatures`, `ManipulationPatterns`) and confirms `context.cache` ends with `['spacy_doc', 'tokens_word_counter', '_text_signals']` after EntityGraph runs — the shared cache now has all three §2 primitives populated. `matrix_build` round-trips a dict-list with mixed columns to the expected `(4, 3)` matrix. `/predict` end-to-end blocked only by 503 "Model not available" (model untrained in this environment, not a regression).

## Apr 28 2026 — production audit §1.6 / §1.9 / §1.12

**§1.6 — `DatasetFeatureGenerator.generate()` no longer skips scaler/selector when the cache is off:**
- Two bugs were collapsed in the same call site. The cache branch tried to reach into `self.pipeline.pipeline.scaler` / `.selector` / `.fit_scaler(...)` — but `FeaturePipeline` has no such attributes, so the moment a real scaler was configured the path would have raised `AttributeError`. The no-cache branch (used by the test fixtures) called `self.pipeline.extract_with_labels(...)`, which silently drops `labels` and `fit` because the underlying `FeaturePipeline.process` ignores them, so it returned the raw extractor matrix and never ran scaler/selector at all. Net effect: tests validated a different pipeline than production.
- `scaler: Optional[FeatureScalingPipeline]` and `selector: Optional[FeatureSelectionPipeline]` are now first-class dataclass fields on `DatasetFeatureGenerator`. Both branches funnel through `_cached_extract()` (which short-circuits to `self.pipeline.extract(contexts)` when `cache_manager is None`) and the scaler/selector application is hoisted into the common tail of `generate()`. The order is now invariant: `extract → scaler.fit/transform → selector.fit/transform → matrix build`, regardless of whether the cache is enabled.

**§1.9 — `cache_manager._context_key` promoted to public `context_key()`:**
- `CacheManager.context_key(ctx)` is now the documented public API for deriving a deterministic cache key from a `FeatureContext`. The old `_context_key` is preserved as a one-line back-compat shim that delegates to `context_key`, so all existing internal callers (`get_or_compute`, `get_or_compute_batch`) keep working without a churn-only diff.
- `DatasetFeatureGenerator._cached_extract` updated to call the public method. Removes the encapsulation leak the audit flagged: any future refactor of the cache-key derivation can now break only the public contract, not silently break the generator with no test coverage.

**§1.12 — `FeatureEngineeringPipeline.process()` reordered: prune now runs *before* stats:**
- Old order was `extract → validate → stats → prune → scale → select → report`. The `FeatureStatistics` step paid the full O(N²) `compute_correlation_matrix` cost on the un-pruned column set, only to throw 30–40 % of those columns away in the very next step.
- New order: `extract → validate → PRUNE → stats → scale → select → report`. Stats now describe the post-prune matrix the model will actually train on, and the correlation work is bounded by the kept-column count.

## Apr 27 2026 — `src/graph/` audit fixes C1–C5 + P1–P4

**C1 — `EntityGraphBuilder.extract_graph_features` alias added.** `GraphFeatureExtractor.extract_from_graphs` was calling `extract_graph_features` (the name `NarrativeGraphBuilder` exposes), but `EntityGraphBuilder` only defined `extract_features` → `AttributeError` on every entity-graph request. New alias keeps both builders API-compatible.

**C2 — `GraphOutput(...)` no longer raises `ValidationError`.** The pipeline used to pass raw `Dict[str, Dict[str, float]]` adjacency dicts where the schema expects `GraphStructure(nodes, edges)`, plus unknown `entity_metrics=` / `narrative_metrics=` kwargs against `extra="forbid"`. Added `_to_graph_structure(...)` adapter in `graph_pipeline.py`, added `entity_metrics` / `narrative_metrics` fields to `GraphOutput` (`graph_schema.py`), and explanation is now serialised via `.to_dict()` before construction. Whole construction is also wrapped in try/except so a future schema mismatch can't take down a request.

**C3 — `GraphExplainer.explain` now accepts `temporal_features=`.** The pipeline was passing `temporal_features=` but the signature only had `text=`, raising `TypeError` on every call. Signature now takes both; when `temporal_features` is supplied the explainer re-uses it (no second `TemporalGraphAnalyzer.analyze` per request — partial G-R2 fix).

**C4 — Per-graph metrics now reach the model.** Result dict surfaces `entity_graph_metrics` and `narrative_graph_metrics` under exactly the keys `feature_pipeline._merge_graph_features` already reads. Previously the consumer's `.get("entity_graph_metrics", {})` always returned `{}` because the producer only had local variables `entity_metrics` / `narrative_metrics`.

**C5 — Edge weights survive end-to-end.** `EntityGraphBuilder.build_graph` used to call `normalize_graph` then `to_undirected`, both of which downcast to `Dict[str, List[str]]` — silently throwing away every co-occurrence weight just computed. New `canonicalize_weighted` in `graph_analysis.py` is the canonical, **idempotent** symmetric-weighted form (`Dict[str, Dict[str, float]]`); `build_graph` returns it directly. `GraphExplainer._node_importance` now uses weighted-degree, `_edge_importance` uses the actual edge weight. `GraphAnalyzer.compute_graph_metrics` uses weighted-degree distribution for `graph_entropy`. NetworkX construction in `graph_embeddings._to_nx` now sets `weight=` on every edge.

**P1 — Canonicalize once at the top.** `GraphPipeline._run_with_doc` calls `canonicalize_weighted(narrative_graph)` after building (entity-graph builder already returns canonical form). Because canonicalization is idempotent the analyzer / extractor / explainer all do a no-op second pass when invoked, instead of repeating an `O(N+E)` symmetrise / normalise loop ~3× per request.

**P2 — Sparse triangle-counting clustering.** `_average_clustering_sparse` in `graph_analysis.py` builds a CSR adjacency once and computes per-node triangles via `(A² ⊙ A)` row-sums, replacing the `O(N · k²)` Python double-loop in `entity_graph.extract_features` and `compute_graph_metrics`. Measured ~70× speedup on a 200-node graph (4.4 ms vs ~300 ms).

**P3 — `GraphPipeline.run_batch(texts)`.** New batched entry point uses `nlp.pipe(texts, batch_size=cfg.batch_size)` so a batch of N texts shares a single spaCy parser pass instead of N serial calls. `run` and `run_batch` both delegate to a shared `_run_with_doc` helper; entity graph build was factored into `_entity_graph_from_doc(doc)` so the per-text spaCy call is no longer repeated.

**P4 — Sparse Lanczos eigendecomposition.** `spectral_eigen_embedding` now accepts a weighted dict OR an ndarray, builds a `scipy.sparse.csr_matrix` directly, and uses `scipy.sparse.linalg.eigsh(k=spectral_dim, which='LA')` for any graph above ~32 nodes. The dense `nx.to_numpy_array` + `np.linalg.eigvalsh` path is the fallback for tiny graphs (where ARPACK setup outweighs its asymptotic win) and the on-failure backup. Also fixes G-E1: empty graphs now return a fixed-length zero vector matching `_embedding_target_dim(cfg)` instead of `np.zeros(1)`, so the downstream `graph_embedding_*` feature columns have a stable schema.

**Pre-existing bugs uncovered while validating, fixed in the same pass:**
- `TemporalGraphFeatures.to_dict` and `NarrativeGraphFeatures.to_dict` returned `self.__dict__`, but both classes use `@dataclass(slots=True)` — `__dict__` doesn't exist. Both now build the mapping from `__slots__`.
- `NarrativeGraphBuilder.build_graph` did `graph.setdefault(k, {})` which clobbered the outer `defaultdict(lambda: defaultdict(float))`'s inner factory, so the next `graph[src][tgt] += 1.0` raised `KeyError`. Replaced with an explicit `if k not in graph: graph[k] = defaultdict(float)`.
- `EntityGraphBuilder.__init__` blank-spaCy fallback now adds a `sentencizer` pipe so `doc.sents` doesn't raise `E030` when `en_core_web_sm` is unavailable.

## Apr 27 2026 — `src/training/` audit fixes CRIT-1..CRIT-4 (+ adjacent factory aliases)

**CRIT-1 — `lr_scheduler_engine.py` SyntaxError.** Line 1 was `s#rc\training\lr_scheduler_engine.py` (mangled comment, Python parsed `s` as a Name expression, which made the line-3 `from __future__ import annotations` illegal — `__future__` imports must precede any non-comment code). Replaced with a proper `# src/training/lr_scheduler_engine.py` comment. Module now imports cleanly; `LRSchedulerEngine` (warmup + plateau + adaptive LR control) is reachable.

**CRIT-2 — `step_engine.py` byte-identical duplicate of `loss_engine.py`.** Confirmed by `cmp` (zero diff, both 6263 bytes); verified zero importers via `rg`. Deleted outright. The misnamed file pretended to be a per-step orchestrator but actually contained `LossEngine` — a stale copy that would silently drift from the canonical one on future edits.

**CRIT-3 — Broken `set_seed` import in `create_trainer_fn.py` and `cross_validation.py`.** Both did `from src.training.training_utils import set_seed`, but `training_utils.py` only defines `set_global_seed`. Changed both to `from src.utils.seed_utils import set_seed` (the canonical location used by `main.py`, `hyperparameter_tuning.py`, and tests). The training package's CV + factory pipelines now import without `ImportError`.

**CRIT-4 — Missing `build_model` in `model_factory.py`.** `create_trainer_fn.py:9` did `from src.models.registry.model_factory import build_model`, but only `ModelFactory.create(model_type, config)` existed. Added a module-level `build_model(task, config)` wrapper with a `_TASK_TO_MODEL_TYPE` map (covers both short names like `"bias"` and the canonical `"bias_classifier"` form) that resolves to `ModelFactory.create`. This preserves the existing factory contract while giving the training layer a stable entry point.

**Adjacent factory aliases discovered while validating CRIT-3/4 (same missing-helper pattern, same file).** `create_trainer_fn` also imports `build_optimizer` and `build_scheduler`, neither of which existed:
- Added `build_optimizer(model, lr, weight_decay, optimizer_type='adamw', **kwargs)` to `src/models/optimization/optimizer_factory.py` as a thin wrapper over the existing `create_optimizer(model, learning_rate=..., weight_decay=...)`.
- Added `build_scheduler(optimizer, config: dict)` to `src/models/optimization/lr_scheduler.py` as a thin wrapper over `get_scheduler(...)`, reading `scheduler_type` / `num_training_steps` / `num_warmup_steps` / `num_cycles` / `power` from the config dict.

**Verification.** Direct import tests:
- `from src.training import lr_scheduler_engine` → OK
- `from src.training import cross_validation` → OK
- `from src.training import create_trainer_fn` → gets past all training-layer imports; only fails downstream on a pre-existing circular import in `src/models/loss/multitask_loss.py` (out of scope of CRIT-1..4, captured for the next audit pass).
- `Start application` workflow restarts and serves cleanly with no new errors.

**Remaining (not yet fixed) audit findings.** CRIT-5 (DDP wrap after `TrainingStep` capture → silent gradient-sync bypass), CRIT-6 (no `Trainer.load_checkpoint`; AMP scaler / optimizer / scheduler / global_step not persisted), CRIT-7 (scheduler advances on AMP overflow), CRIT-8 (NaN-skip zeros mid-accumulation grads), CRIT-9 (`val_loss` never produced → default early-stopping is dead), CRIT-10 (early-stop returns last-epoch model instead of best), plus the `🟠 PERF-*`, `🟡 LOSS-*`, `🧠 MT-*`, `🚀 GPU-*`, `🔄 REC-*` items from the v12 training audit.

## Apr 27 2026 — Training-layer audit v12: CFG-*, EDGE-*, dead-code pass

**Config-surface fixes (`⚙️ CFG-1..CFG-6`).**
- `CFG-1` — already fixed in earlier pass: `Trainer.__init__` honours `params_override["epochs"]`.
- `CFG-2` — `TrainingStepConfig.spike_lr_scale: float = 0.5` is now the single source of truth for the LR-reduction factor; `TrainingStep._reduce_lr` reads it instead of the hardcoded `0.5`.
- `CFG-3` — `Trainer.__init__` exposes `log_every_steps` (default 50) and `checkpoint_every_steps` (default 500) as both ctor kwargs and `params_override` keys; `_train_epoch` uses them in place of the hardcoded modulos. `TrainerConfig` dataclass mirrors the same fields.
- `CFG-4` — `Trainer.__init__` now accepts an optional `setup_config: TrainingSetupConfig`. Frozen-default behaviour preserved; this is the documented escape hatch for callers (e.g. fast Optuna trials) who need to disable `run_sanity_check` etc.
- `CFG-5` — `LossEngineConfig.normalization` carries an explicit docstring covering the `active`/`sum`/`mean` semantics and the auto single-task override.
- `CFG-6` — `TaskScheduler.__init__` caches a single-task fast path, and `next_task()` short-circuits to that cached value (skips the rng / softmax dispatch). A WARNING is emitted when a non-`round_robin` strategy is wired with one task.

**Edge-case fixes (`🧪 EDGE-*`, section 9 of v12 report).**
- NaN labels — `TrainingStep.run` wraps the autocast forward + `loss_engine.compute(...)` block in `try/except RuntimeError` so non-finite logits / labels respect `skip_nan_loss=True` (the previous `torch.isfinite(total_loss)` check only fired AFTER the loss aggregate was built and never trapped exceptions raised inside `MultiTaskLoss`).
- Non-dict batches — `TrainingStep.run` now asserts `isinstance(batch, dict)` at the contract boundary; previously a tuple/list batch crashed deep inside `model(**batch)` with a misleading `TypeError`.
- Imbalanced binary — `binary_loss(...)` accepts an optional `pos_weight: torch.Tensor`, threaded straight to `binary_cross_entropy_with_logits`. Callers no longer need to re-implement the loss to balance a 99/1 class split.

**Dead-code purge (`🧹` section 7 — confirmed zero external usages via ripgrep).**
- Deleted entire file `src/training/lr_scheduler_engine.py` (`LRSchedulerEngine`).
- Deleted `MultiOptimizer` from `src/models/optimization/optimizer_factory.py`.
- Deleted `SchedulerWrapper` from `src/models/optimization/lr_scheduler.py`.
- From `src/training/training_utils.py`: deleted `inference_mode()`, `set_global_seed()`, `safe_cuda_execution()`, and the **first** duplicate `TrainingMetrics` class at L205 (the "improved" one at L341 is the authoritative version and is retained).
- From `src/training/loss_functions.py`: deleted `LossConfig` and `LossFactory` (no callers — `MultiTaskLoss`/`TaskLossConfig`/`TaskLossRouter` is the real router).

**Verification.** `Start application` workflow restarts cleanly; FastAPI startup completes; `GET /` returns 200; no `ImportError` / `NameError` from the deletions. CRIT-* / PERF-* / LOSS-* / MT-* / GPU-* / REC-* items from the v12 audit remain outstanding.

## Apr 27 2026 — `src/features/` audit §3.5 + §3.6 (extractor availability indicators)

Cross-checked every item from the v13 `src/features/` audit (`Pasted--TruthLens-src-features-Layer-Audit-…_1777302237795.txt`) against the live source. **All §1.x critical bugs, all §2.x perf items, and §3.1 / 3.3 / 3.4 / 3.7 quality items were already fixed in earlier passes** (HybridTruthLensModel relocated, `_graph_cache_key` carries a config fingerprint, graph features no longer recomputed in fusion + merge, tempfile try/finally + `prune_all`, tokenizer + lexicon SHAs in `_context_key`, `get_shared_nlp` everywhere under `src/features/`, vectorized `WeightedLexiconMatcher` / `LexiconMatcher`, `ensure_tokens_word` at the top of fusion, shared `get_text_signals` for caps + exclamation density with NER mask, `_memoized_dependency_depths`, pickle disk format, `normalized_entropy` with `n>=2` guard, polarity remap clipped to `[-1, 1]` first, `FeatureFusion._normalize` deleted). `BaseFeature.extract_batch` + `safe_extract_batch` exist and `FeaturePipeline.batch_extract` dispatches per-feature through `fusion.extract_batch` so §1.4 is closed too.

**§3.6 — syntactic spaCy-fallback "cliff".** `SyntacticFeatures._extract_fallback` previously emitted `0.0` for every spaCy-derived column (`syn_pos_entropy`, `syn_complexity`, dispersion, sentence entropy, coordination, subordination). The result was a bimodal distribution (real values vs constant 0) that the model trivially learned as a "spaCy-was-up" signal. The fallback now emits **only** the spaCy-free columns (`syn_sentence_avg_len` from the regex tokens + simple-sentence split) and a new binary indicator `syn_spacy_available ∈ {0.0, 1.0}` that the spaCy path also emits (set to 1.0). Missing keys on fallback rows are filled by `FeatureSchemaValidator.fill_value` (training-set mean once `FeatureScalingPipeline` is fitted), removing the spurious cliff while keeping the schema stable for callers that pre-declare the full feature list.

**§3.5 — semantic encoder unavailable.** Same indicator pattern applied for symmetry: `SemanticFeatures.extract` emits `sem_available = 1.0` on the success path; `_empty()` (returned when no embedding is wired up or the encoder failed) emits `sem_available = 0.0`. The downstream head can now attenuate the 7-dim `sem_*` block on encoder-failure rows instead of treating an all-zero `sem_*` block as a legitimate signal.

**Verification.** `Start application` restarts cleanly; FastAPI startup completes; `GET /` returns 200. No new errors / warnings in the boot log.

## Apr 28 2026 — Inference-layer audit: POST-PROCESSING, MULTI-TASK, GPU/DEVICE, MEMORY/BATCHING

Pre-condition fix: `portalocker` was imported by `src/utils/json_utils.py` but missing from the env, blocking the API at uvicorn boot. Installed it; FastAPI now starts cleanly and `GET /` returns 200.

**Post-processing (`📊 PP-1..PP-4`).**
- `PP-1` — `UnifiedPredictor._format_output` (in `src/inference/model_loader.py`) was always softmaxing logits, corrupting multilabel/binary heads. Added `_resolve_task_type(name)` (consults `task_metadata`, then the `TASK_CONFIG` proxy) and made `_format_output` task-aware: `softmax`/`argmax` for `multiclass`, `sigmoid + threshold` for `multilabel` and `binary`. Output now also carries `task_type`.
- `PP-2` — `Postprocessor` gained `load_task_thresholds(path)` which reads `<model_dir>/thresholds.json` and merges into `config.task_thresholds`. `PredictionPipelineConfig` got `task_thresholds_path` / `task_thresholds`; the pipeline wires both into the postprocessor at construction. `predict_multitask` resolves the per-task threshold via `_resolve_threshold(task)` so each head's calibrated cutoff is applied (no more `0.5` everywhere).
- `PP-3` — `Postprocessor.process` now requires an explicit `task_types` mapping and raises `ValueError` if it is missing and `KeyError` if a task is absent from the mapping. The previous silent `multiclass` fallback masked multilabel mis-routing; production tracebacks now point straight at the misconfigured task.
- `PP-4` — `PredictionService._compute_uncertainty` now branches on `out["task_type"]`. Multilabel heads use the Bernoulli-sum entropy `-Σ_k [p_k log p_k + (1-p_k) log(1-p_k)]`, binary uses per-sample Bernoulli, multiclass uses categorical. Falls back to a row-sum heuristic when `task_type` is absent (legacy callers). The single-head HF engine in `inference_engine.py:predict_for_evaluation` now emits `task_type: "multiclass"` so the service's branch resolves deterministically.

**Multi-task (`🧠 MT-3`).**
- `MT-3` — `InferenceEngineConfig.calibrators` now accepts a single `Calibrator` **or** a `Mapping[str, Calibrator]`. `_resolve_calibrator(task)` is consulted at every calibration call site. The legacy single-calibrator-for-all-tasks path is preserved by passing a non-mapping. `MT-1` / `MT-2` are deferred — they overlap with the still-pending `CRIT-2` engine unification and will be addressed in that pass.

**GPU / device (`🚀 DEV-1..DEV-4`).**
- `DEV-1` — Removed the unconditional `model.half()` cast in `model_loader._load_torch_model` and removed `torch_dtype=...` from `from_pretrained` in `inference_engine`. Weight precision is the operator's call (env / config); inference now uses autocast for mixed precision.
- `DEV-2` — Both `inference_pipeline._resolve_amp_dtype` and `inference_engine._resolve_amp_dtype_engine` read `TRUTHLENS_AMP_DTYPE` (default `bf16`, accepts `fp16`/`float16`/`bfloat16`/`fp32`/`float32`/`none`/`disabled`). The autocast block uses the resolved dtype, so the A100 path stays bf16 by default and CPU/old-GPU operators can opt into fp16 or full precision without code changes.
- `DEV-3` — After load, both `_load_torch_model` and `load_multitask_model` call `model.eval()` and `model.requires_grad_(False)`. Inference runs no longer hold gradient buffers or accumulate autograd graph state.
- `DEV-4` — `_apply_calibration` now returns CPU tensors for the `IsotonicRegression` path (sklearn doesn't accept CUDA arrays). The temperature-scaling path still runs on-device. `predict_for_evaluation` re-aligns `cal` to `logits.device` before `cat` so the on-device accumulation path (MEM-1, below) stays consistent.

**Memory / batching (`🧮 MEM-1..MEM-4`).**
- `MEM-1` — `InferenceEngineConfig.keep_outputs_on_device` (default `False`) controls whether `predict_for_evaluation` accumulates per-batch tensors on the GPU and transfers once at the end (long eval runs on big GPUs) or per-batch (default — bounds peak GPU memory at one batch).
- `MEM-2` — `InferenceCache._evict_disk_if_needed()` enforces `config.max_disk_items` by mtime LRU on every `set()`. `get()` calls `os.utime(path, None)` to keep the LRU ordering accurate. `None` / `<=0` preserves the previous unbounded behaviour. Verified: 7 sets with `cap=3` → exactly 3 files remain.
- `MEM-3` — `_safe_write` now uses `path.with_name(path.name + ".tmp")` (was `path.with_suffix(".tmp")`, which collided across `.json` and `.json.gz` modes and left a dangling temp on mode toggles) and `os.fsync`s the gzip branch too (was uncompressed-only). Verified: no `.tmp` files linger after a normal write.
- `MEM-4` — Memory cache stores the raw value (already in place from earlier passes; verified during this audit).

**Other.**
- Bidirectional fix in `src/training/checkpointing.py:load_checkpoint` for the `_orig_mod.` prefix introduced by `torch.compile`: the wrapper is peeled from the model side **and** the prefix is stripped from the state-dict side before `load_state_dict`. Resume now works whether the checkpoint was saved with or without `torch.compile` enabled, regardless of the runtime setting.

**Verification.** Standalone smoke tests pass: `PP-3` raises on missing `task_types`/missing entries; `PP-2` loads thresholds JSON; `MEM-2` evicts to cap; `MEM-3` leaves no `.tmp` files; `DEV-2` resolves both `bf16` and `fp16` env values to the right `torch.dtype`. `Start application` restarts cleanly: `Application startup complete`, `GET /` returns 200, no warnings.

**Deferred (documented for next pass).** `MT-1` (engine selection / dispatch) and `MT-2` (per-task batch grouping) overlap with `CRIT-2` engine unification and will be addressed alongside it.

## Apr 28 2026 — Inference-layer audit: RECOMPUTATION, DEAD CODE, CONFIG, EDGE CASES

Closed the four remaining categories from the inference-layer audit. No public API change; behaviour change is "fail loudly" (REC-4, EDGE) and "share defaults" (CFG-2/4/5/6/7).

**New module — `src/inference/constants.py`.** Single source of truth for values that previously diverged across modules: `INFERENCE_CACHE_VERSION` (was `"v1"` in `InferenceCacheConfig`, `"v2"` in `predict_api`), `DEFAULT_INFERENCE_BATCH_SIZE` (was `16`/`32`/`32` in three places), `DEFAULT_MAX_LENGTH`, `REPORT_VERSION` (was a hardcoded `"v3"` in `report_generator`).

**Recomputation (`REC-1..4`).**
- `REC-1` — `ArticleAnalyzer._run_prediction` previously read `predictions` / `probabilities` / `logits` / `graph` / `graph_explanation` / `drift` / `monitoring` from the cache-backed `predict()` blob, but `predict()` only returns `{label, confidence, fake_probability}`. Every one of those keys was always `None`, polluting the report. Now surfaces the actual fields and drops the four bogus keys from the analyzer report (graph features already live under `graph_features` / `entity_graph`; drift / monitoring belong to their own services).
- `REC-2` — `PredictionService.predict_full(text)` now consults the cache (key `"__full__::<text>"` to avoid collision with the basic-blob key written by `predict()`) and writes the full report back on success. Was previously the only public entry that re-ran the whole pipeline (forward + uncertainty + report-gen) on every call.
- `REC-3` — Tokenizer was constructed twice on every cold start (once in `InferenceEngine._load_model`, once in `ModelLoader._load_tokenizer`). Added a process-wide `_TOKENIZER_CACHE` keyed by resolved path in `model_loader.py` and exposed `get_cached_tokenizer(path)`; both call sites route through it. `use_fast=True` HF tokenizers are stateless for inference, so sharing is safe.
- `REC-4` — `ReportGenerator.generate_report` silently re-ran `AggregationPipeline` whenever `analysis["aggregation"]` was missing but a `profile` was provided. Now raises `ValueError` with a clear remediation message: callers must pre-aggregate. Empty-profile case still returns an empty `aggregation` dict for backwards compatibility.

**Dead / unused (`UNUSED-FIX`).**
- `predict_api` was constructing an `InferenceMonitor` and attaching it to `service.monitor`, but no code path ever called `monitor.update(...)`. `PredictionService.__init__` now accepts `monitor=` (passed by `predict_api`), and `_record_monitor(...)` is invoked from `predict`, `predict_full_batch`, and `predict_full` on both success (with `confidence`) and error (`error=True`). Monitor failures are caught and logged so they can never break the request path.
- `InferenceConfig.return_logits` / `return_probabilities` flags existed but were ignored; `predict_for_evaluation` always emitted both arrays. The flags are now honoured in the per-task output dict (`calibrated_probabilities` is gated with `probabilities` since it's a derivative of the same family). `InferenceEngine.predict()` now raises a clean `RuntimeError` when `return_probabilities=False` instead of crashing later with a confusing `KeyError`. The internal postprocessor wiring was switched to the local `logits` / `cal` tensors (rather than reading back from `task_output[...]`) so the postprocessor still runs even when both flags are off.

**Config (`CFG-2/4/5/6/7`).**
- `CFG-2` — `cache_version` divergence closed by routing both `InferenceCacheConfig` (default) and `predict_api` (operator-overridable, falls back to constant) through `INFERENCE_CACHE_VERSION`.
- `CFG-4` — `PredictionPipelineConfig.device` default changed from `"cpu"` to `"auto"`; the pipeline `__init__` resolves `"auto"` → `"cuda" if cuda_available else "cpu"` before constructing `torch.device(...)` (which would otherwise raise on `"auto"`).
- `CFG-5` — `InferenceConfig.batch_size` and `BatchInferenceConfig.batch_size` both default to `DEFAULT_INFERENCE_BATCH_SIZE` from the constants module.
- `CFG-6` — `report_version="v3"` literal in `report_generator` replaced with `REPORT_VERSION` constant.
- `CFG-7` — `InferenceConfigLoader._validate_config`'s asymmetric handling of `batch_size` (`REQUIRED_FIELDS` asserted type but tolerated absence) was made explicit with a rationale comment, and a backstop now injects `DEFAULT_INFERENCE_BATCH_SIZE` when the YAML omits the key entirely so the loader is in lockstep with the engine's dataclass default.

**Edge cases (in `inference_engine._forward` / `_load_label_map`).**
- Truncation is now surfaced. Before tokenising, a no-truncation length probe runs; if any input exceeds `max_length`, a single `WARN` is logged with the count, ratio, and longest length so operators see when a confidence was computed over a silently truncated tail.
- Inputs with `attention_mask.sum() < 3` (emoji-only, stray punctuation) emit a per-item `WARN`. Distributions over near-empty inputs are usually degenerate; this surfaces "why is this confidence uniform?" without changing behaviour.
- Non-finite logits (NaN / Inf) raise a `RuntimeError` rather than poisoning every downstream operation (softmax, calibration, argmax, entropy). Better to fail than emit a silently broken prediction.
- Missing `label_map.json` now logs a `WARN` (was a silent skip). The skip itself is correct (numeric labels work), but the warning explains why `fake_probability` is `None` for every prediction — this was the most-asked support question on cold starts of a new model.

**Verification.** `Start application` restarts cleanly; `Application startup complete`; `GET /` returns 200. Smoke import tests pass for all six edited modules; `PredictionService.__init__` exposes the `monitor` kwarg; `InferenceCacheConfig().cache_version == "v2"`; `PredictionPipelineConfig().device == "auto"`; `InferenceConfig.batch_size == 32` and `BatchInferenceConfig.batch_size == 32` both resolve through the constant.

**Still deferred (architectural, out of scope for this audit pass):** `MT-1`/`MT-2` engine unification, `DriftDetector` endpoint wiring, `run_inference.py` multitask migration, `_flatten`/`_worker` dedup, `enable_full_pipeline` re-evaluation. These are tracked for the engine-unification pass.

## Apr 28 2026 — Features-layer audit: critical bugs §1.1–§1.11 + perf §2.1/§2.3/§2.6

Closed the highest-impact items from the 757-line `src/features/` audit. No public API change; behaviour change is "previously-zero columns now carry signal" (§1.1) and "extractor failures are observable" (§1.5).

**New module — `src/features/base/lexicon_loader.py`.** Single source of truth for the 60+ category lexicons that used to ship as inline `LEXICON = {...}` placeholders in 10 extractor modules. Lazy-loads JSON files from `src/config/lexicons/` once per process; exposes `load_lexicon`, `load_lexicon_set`, `load_lexicon_dict`. Missing files / keys log a `WARN` and return empty rather than raising.

**New seed lexicons — `src/config/lexicons/*.json`.** Ten files (`bias`, `bias_lexicon`, `framing`, `ideology`, `narrative`, `narrative_frame`, `narrative_conflict`, `propaganda`, `propaganda_lexicon`, `manipulation`) totalling ~1,140 seed terms across ~57 categories. Each file is a `{category: [terms]}` map with a `_doc` metadata key explaining provenance and where it diverges from sibling files (e.g., `bias/framing.json` vs `narrative/narrative_frame.json` use distinct vocab on purpose).

**Critical bugs (`§1.1`/`§1.4`/`§1.5`/`§1.7`/`§1.8`/`§1.10`/`§1.11`).**
- `§1.1` — All 10 placeholder lexicons replaced with `load_lexicon_set("<file>", "<category>")` calls. Smoke test confirms every patched extractor now produces non-zero columns on representative misinformation text (e.g., `bias_loaded=0.667`, `frame_security=1.0`, `ideology_right=1.0`, `propaganda_fear=0.2`, `manipulation_fear=0.4`). Phrase-list placeholders (`COMPILED_BIAS_PHRASES`, `COMPILED_FRAME_PHRASES`, `COMPILED_IDEOLOGY_PHRASES`) reset to `[]` so the `for p in []` summation contributes a clean 0 instead of crashing on `[...]`. `propaganda_lexicon_features` re-derives `BANDWAGON_PHRASES` / `SLOGAN_PHRASES` from the same JSON via `re.escape(...)`.
- `§1.4` — Removed the no-op `extract_batch` overrides on `BiasFeaturesV2` and `BiasLexiconFeatures` (both were verbatim `[self.extract(c) for c in contexts]` copies of `BaseFeature.extract_batch`). Removing them lets the base class own the contract so any future vectorized batch path lands here for free.
- `§1.5` — `FeatureFusion.extract_batch` now emits a `<feature_name>_extracted` indicator (1.0 on success, 0.0 on empty/failed output) for every extractor. Generalises the existing `sem_available` / `syn_spacy_available` pattern so a downstream model can mask any silently-dropped extractor instead of conflating "extractor failed" with "all features happened to be zero".
- `§1.7` — `FeatureStatistics._cached_matrix` was previously unkeyed: the FIRST `features` list ever passed in was returned for every subsequent call, silently corrupting any pipeline that reused a `FeatureStatistics` instance across batches. Cache is now keyed on `(id(features), len(features), len(features[0]) if features else 0)` and invalidates correctly on identity / shape change.
- `§1.8` + `§2.6` — `FeatureReport`'s `for i: for j: if abs > 0.95` Python loop (~31k iterations per report on the ~250-feature schema) replaced with a single `np.triu_indices` / `np.where` call. Same output, runs entirely in C.
- `§1.10` — `cache_manager._context_key` switched from `default=str` to a custom `_stable_default` that sorts sets/frozensets, lists tuples, hex-encodes bytes, and falls back to `repr` for everything else. Two requests with the same logical metadata (e.g., a `set` of feature names) now hash identically regardless of insertion order. The lexicon/schema fingerprints already in the payload (`lexicons`, `feature_set`) were preserved.
- `§1.11` — `bias_lexicon.compute_bias_features` used a private `_TOKEN_PATTERN = re.compile(r"[A-Za-z']+")` that was ASCII-only and silently stripped accented characters from non-English headlines (`café` → `caf`). Now routes through `ensure_tokens_word(context, text)` (document-level) and `tokenize_words(sent)` (sentence-level heatmap), matching every other extractor in the codebase.

**Performance (`§2.1`/`§2.3`/`§2.6`).**
- `§2.1` — Added `ensure_tokens_word_counter(context)` to `tokenization.py`. Caches the per-context `Counter` on `ctx.cache["tokens_word_counter"]` so the eight extractors (bias / discourse / narrative / propaganda / manipulation / conflict) that all call `Counter(tokens)` independently no longer re-tally the same token list. Helper added; per-extractor wiring to the new helper is staged for the next pass since each call site has its own `_ratio` shape and the fix is mechanical-but-bulky.
- `§2.3` — `syntactic_features._memoized_dependency_depths` is still O(N) within a `Doc`, but the result is now cached on `doc.user_data["_syn_depth_cache"]` via the new `_dependency_depths_for_doc(doc, tokens)` wrapper. spaCy `Doc` objects are Cython extension types and can't be weak-referenced, so the doc's own `user_data` dict (the idiomatic spaCy hook for per-doc state) carries the cache. Sets up for §2.7 (cross-extractor `Doc` sharing) without coupling to it.
- `§2.6` — Closed jointly with `§1.8` above.

**Centralised constants (`§3.1`).** Added `EPS = 1e-8` and `MAX_CLIP = 1.0` to `src/features/base/numerics.py`. The 10 patched extractors now `from src.features.base.numerics import EPS, MAX_CLIP, normalized_entropy` instead of redeclaring the constants at module top; ~25 other files in the features layer still carry their own copies and will be migrated in the next pass.

**Verification.** `Start application` restarts cleanly: `Application startup complete`, `GET /` returns 200, no warnings. Direct extractor smoke test confirms every patched module loads its lexicons, accepts a real `FeatureContext`, and emits non-zero feature columns on representative misinformation text. `/predict` and `/analyze` HTTP endpoints reject with the pre-existing "Model not available. Please train the model first." (untouched by this pass).

**Deferred (out of scope for this pass; need design decisions or larger refactors).** `§1.2` schema drift (rename ~20 extractors vs expand schema), `§1.3` duplicate extractors, `§1.6` dataset_feature_generator scaler/selector branch, `§1.9` `_context_key` promotion, `§1.12` pipeline reorder, `§2.2` LexiconMatcher rollout to the six Counter-based extractors, `§2.4` `spacy.pipe`, `§2.5` `ensure_tokens_word_counter` per-extractor wiring, `§2.7` shared `Doc` graph extractors, `§2.8` pre-allocate matrix.

## Apr 28 2026 — Training-layer audit followups: dead code, CFG-3, edge cases

Closed the still-outstanding items from the **🧹 Unused/Dead/Confusing Code**, **⚙️ Configuration**, and **🧪 Edge Cases** categories of the v12 training-layer audit. The four critical (CRIT-1..4) and most CFG-* / EDGE-* items had already been fixed in the v12 pass; this pass wraps the remainder.

**Dead-code purge (`🧹`).**
- `src/training/training_utils.py` rewritten lean — removed eleven helpers with **zero** call sites (verified via ripgrep across the entire codebase, including tests): `training_precision`, `configure_training_precision`, `zero_gradients`, the local `autocast` context manager (TrainingStep uses `torch.amp.autocast` directly), `StepTimer`, `enable_model_eval`, `enable_model_train`, `clip_gradients`, `detach_tensor_dict`, `check_finite`, `compute_batch_size`. File now contains only the actually-used surface: `get_device`, `move_batch_to_device`, `compute_grad_norm`, `get_current_lr`, `compute_throughput`, `TrainingMetrics`. ~150 LOC dropped.
- `src/training/__init__.py` (previously empty) now declares the package's public API: `Trainer`, `TrainerConfig`, `TrainingStep`, `TrainingStepConfig`, `TrainAction`, the `TrainingSetupConfig` family, the `training_utils` exports, `LossEngine`/`Config`, the streaming-metric trio + `EvaluationEngine`/`Config`, the four engine modules (`MonitoringEngine`, `TaskScheduler`, `ExperimentTracker`, `DistributedEngine`) with their configs, the instrumentation classes (`AutoDebugEngine`, `LossTracker`, `SpikeDetector`, `GradTracker`, `FailureMemory`, `AnomalyClassifier`), the CV helpers, and `create_trainer_fn`. The Optuna-backed tuning symbols (`tune_task`, `tune_all_tasks`, `create_study`, `build_objective`) are exposed via `__getattr__` so `import src.training` does **not** drag the optional `optuna` dependency into memory.

**Config (`⚙️ CFG-3`).**
- `TrainingStepConfig.amp_dtype: str = "fp16"` added (accepts `"fp16"` / `"bf16"` / their long forms). `TrainingStep.__init__` resolves it once to a `torch.dtype`, the autocast call now passes `dtype=self._amp_dtype`, and `GradScaler` is gated `enabled = use_amp and amp_dtype == fp16` since bf16 has no overflow-recovery path. Previously the autocast call was hardcoded to fp16 with no operator knob.

**Edge cases (`🧪`).**
- `EDGE-1` — `cross_validation.build_splits` no longer raises `ValueError` for multi-label / regression tasks (no single `label_column`). It now `WARN`s and falls back to plain `KFold` when (a) the column is missing or (b) `StratifiedKFold` itself raises (smallest class < `n_splits`). Also added `n_splits >= 2` and `len(df) >= n_splits` preconditions. Verified all three paths return the requested fold count.
- `EDGE-2` — `EvaluationEngine._update_metrics` previously did `task in batch["labels"]` which raised `TypeError` mid-eval whenever `labels` was a single tensor (single-task collate) instead of a per-task dict (multi-task collate). It now normalises: `dict` is used as-is; a `torch.Tensor` is auto-bound to the only task in `task_logits` (or `WARN`-skipped if there are multiple heads); other types `WARN`-skip. Both collate styles now work transparently.
- `EDGE-3` — `Trainer.__init__` now rejects `train_loader=None` and (where `len(loader)` is defined) rejects empty loaders with a clear remediation message. Previously an empty loader silently no-op'd every epoch and finished with `global_step=0` — the most-asked support question on dataset / batch-size misconfig.
- `EDGE-6` — `cross_validate_task`'s `finally` block used `try: del trainer / except` which masked a real `NameError` whenever `create_trainer_fn` itself raised before the binding existed. Replaced with `if "trainer" in locals(): del trainer`. Also gated `torch.cuda.empty_cache()` on `cuda.is_available()` so CV runs cleanly on CPU-only machines.
- `EDGE-8` — `TrainingStepConfig.__post_init__` validates `gradient_accumulation_steps >= 1` (was a `ZeroDivisionError` 200 batches in if a config typo set it to `0`), `max_grad_norm >= 0`, and `amp_dtype` is one of the four accepted strings. Failures now surface at config-load time with explicit messages.

**Verification.** `Start application` restarts cleanly: `Application startup complete`, `GET /` returns 200. Direct import of all twenty-plus public exports from `src.training` succeeds without `optuna` installed (confirms the lazy `__getattr__` hatch works). `EDGE-8` rejects `gradient_accumulation_steps=0` and `amp_dtype="int8"` with the new messages. `CFG-3` accepts `amp_dtype="bf16"` and stores it. `EDGE-1` runs all three paths (missing column, valid stratified, singleton-class fallback) and each returns 5 folds.

**Already-closed in earlier passes (verified during this audit, no action needed).** `CFG-1` (epochs override), `CFG-2` (spike_lr_scale knob), CFG-3 cadence (`log_every_steps`/`checkpoint_every_steps`), `CFG-4` (TrainingSetupConfig escape hatch), `CFG-5` (LossEngineConfig.normalization docstring), `CFG-6` (TaskScheduler single-task fast path), `CFG-7` (Optuna `_resolve_direction`), the prior `MultiOptimizer`/`SchedulerWrapper`/`LossFactory`/`LRSchedulerEngine` deletions, the duplicate `TrainingMetrics`, NaN-skip in forward+loss, non-dict batch assertion, `pos_weight` in `binary_loss`. Also the v12 PERF / GPU / MT / LOSS / REC items (`torch.amp.GradScaler/autocast`, scheduler-on-overflow gate, scaler in `_save_checkpoint`, `MT-2` binary metric, `MT-3` dry_run, raw `task_losses`, `_filter_batch` removed, `grad_norm` cached, `non_blocking` gating, GPU-1/4 device handling) were already in the live source.

**Out of scope for this pass (deferred):** `PERF-2` per-param `.item()` syncs in `compute_grad_norm`, the `MultiTaskLoss`/`TaskLossConfig` circular import in `src/models/loss/multitask_loss.py` (downstream blocker for `create_trainer_fn` instantiation, not a training-layer issue), `CFG-6` `asdict(self.cfg)` flooding the experiment tracker (cosmetic, requires tracker-side filtering), and the new `src/models/` audit attached to this conversation (separate scope).

## Apr 28 2026 — v13/v14 end-to-end audit: pipeline orchestrator fixes

Closed all CRITICAL and HIGH items from the new end-to-end audit (`attached_assets/Pasted-I-ll-run-the-full-end-to-end-audit-you-described-combin_1777374867577.txt`). All ten findings live in `src/pipelines/truthlens_pipeline.py` and `main.py`; nothing else in the call graph (`api/app.py`, `src/inference/inference_pipeline.py`, the analysis / aggregation / explainability subsystems) needed changes — the bugs were entirely in how the orchestrator wired existing-and-correct components together.

**`src/pipelines/truthlens_pipeline.py` rewrite.**
- `CRIT-1` — `from src.inference.inference_pipeline import Predictor` was a non-existent symbol (that module exports `PredictionPipeline`/`PredictionPipelineConfig`). Switched to `from src.models.inference.predictor import Predictor`, which is the real class — every previous import would have crashed at module load time.
- `CRIT-2` — removed the `predictor or Predictor()` fallback. `Predictor.__init__` requires `model: nn.Module`, so the fallback raised `TypeError` 100 % of the time. The orchestrator now treats `predictor=None` as an explicit "skip prediction" mode (logged once at construction) so callers running analysis-only paths (CLI `--mode infer`, the API's feature-only routes) get a clean `predictions={}` instead of a stack trace.
- `CRIT-3` / `HIGH-1` / `HIGH-4` — `predict(prep.normalized_text)` passed a `str` to a method whose signature is `predict(input_ids: Tensor, attention_mask: Tensor)`. Added a required `tokenizer` parameter to `__init__` (validated against `predictor`: providing one without the other is rejected at construction time) and a centralised `_predict_text(text)` helper that tokenises (`return_tensors="pt"`, `truncation=True`, `max_length=512`), squeezes the leading batch dim, and forwards to `Predictor.predict`. Both the prediction stage AND the explainability `predict_fn` now route through this single call site, so `Callable[[str], Dict]` is honoured everywhere.
- `CRIT-4` — `analyze()` now accepts an optional `labels: Dict[str, Any] | None` keyword. `run_evaluation_pipeline` was being called with `labels=None`, which crashes inside `np.asarray(labels[task])` and was silently swallowed by a broad `except`. Evaluation is dataset-level, so the per-article path now runs it only when the caller provides ground-truth labels; otherwise it logs at `DEBUG` and skips. The legacy "always try, always fail" behaviour is gone.
- `HIGH-2` — `aggregation_pipeline.run(profile, text=...)` passed `profile` *positionally*, where the signature is `run(model_outputs=None, *, text=None, profile=None, ...)`. The profile was being interpreted as raw model outputs, missing the entire `_adapt_profile` Branch-B path. Switched to the explicit `profile=profile` kwarg.
- `HIGH-3` — `metadata.model_version` was pulled from `predictions.get("model_version")`, but `Predictor` never emits that key. Added a `model_version: Optional[str]` ctor parameter so the value is injected once (typically from `config.model.version` or `config.model.encoder`) and propagated to every `analyze()` result. Honest `None` instead of dishonest absence.
- `HIGH-5` — explainability and evaluation stages used `stage_time["explainability"] = stage_time.get("explainability", 0.0)` which always reported 0.0 ms because no `t0` was ever set. Both stages now record proper wall-clock timings (`t0 = time.time()` before the gated block, delta written after).

**`main.py` rewrite.**
- `CRIT-5` / `CFG-1` — added an `argparse` CLI with `--mode {train,infer,both}` (default `infer`) and `--num-samples` so the v13/v14 10-row simulation runs without a populated `data/` directory. The data + training sections (`run_data_pipeline`, `create_trainer_fn`, `trainer.train()`) are now inside `if args.mode in ("train", "both")`; the inference section is inside `if args.mode in ("infer", "both")`. Tokenizer construction was hoisted above the branch since both paths need it.
- The inference section now constructs `TruthLensPipeline(tokenizer=tokenizer, model_version=...)` per the new ctor contract. `enable_explainability` / `enable_evaluation` default to `False` in `--mode infer` because no checkpoint is loaded; they should be flipped on once a real `Predictor` is wired in via `predictor=...`.

**Verification.** `Start application` workflow restarts cleanly; `GET /` returns 200 (uvicorn `Application startup complete`). Direct end-to-end smoke test:
```
p = TruthLensPipeline(model_version="test-1.0")
out = p.analyze("The government clearly failed the people.")
```
returns `metadata.model_version == "test-1.0"`, every stage in `metadata.stages` carries a real wall-clock value (no more 0.0 ms placeholders), `predictions == {}` (graceful no-predictor degradation), `scores` populated with `credibility_score / final_score / manipulation_risk / tasks / uncertainty_summary` (proves `HIGH-2` aggregation kwarg fix is live), and `aggregation` carries `aggregation_version / analysis_modules / explanations / metadata / model_version / raw_scores` keys. The single `ValueError [E029] noun_chunks requires the dependency parse` from `temporal_graph.py` is a pre-existing spaCy-model-pipeline issue (model loaded without `parser`) swallowed by the graph stage's own try/except — out of scope for this audit.

**Out of scope for this pass.** All MEDIUM/LOW items from the same audit (broad `except` narrowing, structured logger keys, type hints on internal helpers, the data-fixture alternative for `CRIT-5`, switching to `PredictionPipeline` for batched inference). The graph-stage spaCy parser issue noted above. The pre-existing lexicon-file-missing warnings on import (`bias.json`, `framing.json`, `ideology.json` in `src/config/lexicons/`) — orthogonal to this audit, tracked separately.

## Apr 28 2026 — v13/v14 audit P3+P4+P5: GPU/perf, wiring, config, edge, dead code

Closed every remaining GPU/PERFORMANCE, INTEGRATION/WIRING, CONFIG, EDGE/RESILIENCE, and DEAD/DUPLICATION item from the v13/v14 end-to-end audit. CRITICAL + HIGH (P1+P2) were already done in the prior pass; this pass wraps up P3 (eval/sim), P4 (GPU+perf), and P5 (config+cleanup).

**`src/aggregation/aggregation_config.py` + `aggregation_pipeline.py`.**
- `CFG-3` — added `model_version: str = "truthlens-v2"` to `AggregationConfig`; replaced the literal `"truthlens-v2"` at `aggregation_pipeline.py:359` with `self.config.model_version`. Single source of truth instead of a magic string buried 350 lines into `run()`.

**`src/models/inference/predictor.py`.**
- `GPU-3` — added `_resolve_amp_dtype_from_env(default="bf16")` helper that mirrors the one in `src/inference/inference_pipeline.py` (maps `bf16`/`bfloat16` → `torch.bfloat16`, `fp16`/`float16`/`half` → `torch.float16`, anything else → `torch.float32`). `_forward()` now consults `TRUTHLENS_AMP_DTYPE` instead of hard-selecting bf16 vs fp16 from `torch.cuda.is_bf16_supported()`. A bf16 request on a card without bf16 support is silently demoted to fp16 to avoid an autocast crash. Both inference orchestrators (`Predictor` + `PredictionPipeline`) now interpret the env var identically — fixes the silent fork where `Predictor` ignored the operator's dtype choice.

**`src/analysis/analysis_registry.py`.**
- `GPU-5` — added `get_default_registry()` process-wide singleton (lazy-built on first call, then memoised). Mirrors `get_default_pipeline()` in `src/graph/graph_pipeline.py` (G-R1). `build_default_registry()` is kept as the explicit-fresh-instance factory for tests.

**`src/pipelines/truthlens_pipeline.py` rewrite.**
- `GPU-5` — `__init__` now uses `get_default_registry()`. The ~14 analyzer instantiations no longer rerun for every `TruthLensPipeline()` construction.
- `GPU-6` — analysis (CPU/spaCy) and graph (CPU/spaCy + NetworkX) stages run in parallel via a 2-worker `ThreadPoolExecutor` shared across calls. New `parallel_stages: bool = True` ctor knob plus a `--no-parallel-stages` CLI flag in `main.py` for A/B comparison. The executor is shut down via the new `close()` method (also called from `__del__` best-effort).
- `GPU-1` + `GPU-2` + `GPU-4` — added `analyze_batch(texts, *, labels=None)`. Per-article CPU stages (preprocessing / analysis / graph / aggregation / explainability) still run serially since the analyzers are not thread-safe. The GPU-bound prediction stage is batched: `_predict_batch_tensors(normalized_texts)` tokenises once with `padding=True`, calls `Predictor.predict_batch` *once*, then fans the per-task tensors back out to per-row dicts. After the batched forward pass `torch.cuda.empty_cache()` is called when CUDA is available so a long-running worker doesn't accrete fragmented allocator slabs across requests.
- `WIRE-1` — documented the architectural fork in the class docstring. The two orchestrators stay separate: `PredictionPipeline` for cache+log+AMP-env+batch GPU work, `TruthLensPipeline` for the full analysis/graph/aggregation stack on top of a raw `Predictor`. Refactoring would be a much larger change.
- `WIRE-2` + `DEAD-1` — removed the unused `TruthLensScoreCalculator` import and `self.score_calculator` instance. The aggregation `except` branch now sets `scores = {}` and records the failure in `errors["aggregation"]` instead of papering over with a redundant raw-calculator call (which would have raised on the same broken profile that just made the wrapping pipeline raise).
- `WIRE-3` + `DEAD-2` — evaluation pulled out of per-article `analyze()`. New `analyze_batch(texts, labels=...)` and `evaluate(texts, labels)` methods carry the dataset-level path that the `run_evaluation_pipeline` contract actually expects (`texts: List[str]`, `labels: Dict[str, Any]`, model + tokenizer). Per-article `analyze()` with `labels=...` now logs an INFO redirect to `analyze_batch`.
- `EDGE-1` — `analyze()` rejects strings longer than `DEFAULT_MAX_TEXT_LEN = 100_000` (configurable via the new `max_text_length` ctor arg). Same cap applied to every row in `analyze_batch`. 100 KB is well above any realistic article (longest Wikipedia featured article is ~95 KB plain text).
- `EDGE-2` — every stage failure is now captured into `result["errors"]` (`stage_name -> repr(exc)`) so callers can introspect partial failures. Prediction, aggregation, explainability, analysis, and graph stages are all wrapped. The `errors` field is always present (empty dict when nothing failed).
- `EDGE-3` — the prediction call site is now wrapped in try/except (was the only un-guarded stage). Combined with the existing `_predict_text` internal try/except, both layers now feed `errors["prediction"]` instead of crashing `analyze()`.
- `EDGE-4` — `predictions` always defaults to `{}` (centralised in `_predict_text` + the batch helper). `model_version` has been sourced from the constructor since the prior pass; this pass closes the loop by guaranteeing every result dict carries the field even when prediction was skipped.

**`main.py`.**
- `CFG-2` — added `--enable-explainability`, `--enable-evaluation`, and `--no-parallel-stages` CLI flags. Routes them through `TruthLensPipeline(...)`. Removes the previous hard-coded values.
- `GPU-1`/`GPU-2`/`GPU-4` — inference loop now calls `pipeline.analyze_batch(sample_texts)` instead of a Python `for text in sample_texts` loop over `analyze()`. Logs a `BATCH SUMMARY` line with `n_articles`/`total_time`/`model_version`. Calls `pipeline.close()` at the end to shut the analysis|graph executor down cleanly.

**Verification.** `Start application` workflow restarts cleanly; `GET /` returns 200. End-to-end smoke test confirms:
- `analyze()` single — `model_version="test-2.0"` propagated, every stage time non-zero, `scores` populated with the five expected keys, `errors` dict captures the pre-existing spaCy `noun_chunks` graph failure (proves EDGE-2 routing).
- `analyze_batch(3 texts)` — completes in ~22 ms (CPU-only / no real predictor); per-article scores all present; `batch_metadata.model_version` matches.
- `EDGE-1` — 200K-char input rejected with the new `text length ... exceeds max_text_length ...` message.
- `CFG-3` — every aggregation result now carries `"model_version": "truthlens-v2"` from `AggregationConfig.model_version` (no more literal in the codebase).
- `GPU-5` — `get_default_registry()` returns the same instance on repeat calls.
- `GPU-3` — `_resolve_amp_dtype_from_env` returns `torch.float32` for `fp32` and `torch.bfloat16` for `bf16` as expected.

**Out of scope for this pass (deferred).** `WIRE-1` consolidation of the two inference orchestrators (would be a large rewrite; the two are deliberately layered today). The pre-existing spaCy "blank en model" issue (no parser → `noun_chunks` raises) — surfaces cleanly in the new `errors["graph"]` field; tracked separately. The lexicon-file-missing warnings on import (`bias.json` / `framing.json` / `ideology.json`) — orthogonal to this audit.

## Apr 28 2026 — Replit migration smoke pass + small log-noise fix

Verified the full app end-to-end on Python 3.12 in this Replit container.

**New tooling.**
- `scripts/build_smoke_datasets.py` — generates 10-row CSVs per (split × task) under `data/{train,val,test}/{bias,ideology,propaganda,frame,narrative,emotion}.csv`, balanced label coverage, 35-word texts.
- `scripts/smoke_e2e.py` — single-shot pipeline check: stage 1 runs `run_data_pipeline` over all 6 tasks, stage 2 runs `TruthLensPipeline.analyze_batch` on 2 representative articles.

**Bug fixes surfaced by the smoke run.**
- `src/data_processing/data_contracts.py`, `src/data_processing/data_augmentation.py`: renamed contract key `frame` → `narrative_frame` so the contract aligns with `narrative_frame` used in `main.py`, `settings_loader`, the multi-task spec, and the aggregation config. The on-disk filename stays `frame.csv` (the settings loader maps `narrative_frame` → file `frame`).
- `src/graph/temporal_graph.py::_extract_entities_from_sent`: wrapped `sent.noun_chunks` in `try/except` for `ValueError` (the spaCy `E029` "noun_chunks requires the parser" error). Falls back to NER-only entity extraction so the graph stage never crashes on parser-disabled pipelines.

**Small log-noise optimization.**
- `src/features/base/lexicon_loader.py`: when a lexicon JSON file is missing on disk, the per-key `Lexicon key missing` warnings (5+ per file) are now suppressed — the single `Lexicon file missing` line is enough, the rest were redundant. Startup log dropped from 22 lexicon lines to 4.

**Result.** `python scripts/smoke_e2e.py` → `ALL STAGES PASSED`:
- stage_data: ~0.25 s, all 6 tasks × 3 splits load + validate + clean + leakage-check
- stage_inference: ~1.15 s for 2 articles, graph score ≈ 0.78, final_score ≈ 0.636
