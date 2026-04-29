# TruthLens AI

## Project Overview
TruthLens AI is a multi-layer AI platform for misinformation detection and news credibility analysis. It combines deep linguistic analysis, narrative extraction, propaganda detection, and graph-based reasoning to provide an interpretable "Credibility Score."

## Architecture
- **Backend**: FastAPI REST API (`api/app.py`) served via Uvicorn
- **Language**: Python 3.12
- **ML/NLP**: PyTorch, Hugging Face Transformers, spaCy, NLTK, LIME, SHAP
- **Port**: 5000

## Recent Refactors (audit fixes applied)
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

## Apr 29 2026 — True multi-task trainer factory (`create_multitask_trainer_fn`)

Built the multi-task counterpart to `src.training.create_trainer_fn`. The single-task factory builds one Trainer per task, which trains six independent encoders for the six TruthLens heads — wasting compute (each encoder forward pass is reused exactly once) and generalisation (no shared representation). The new factory wires up the topology that `MultiTaskTruthLensModel` was designed for: ONE shared encoder, six per-task heads, joint training over weighted task sampling + weighted per-task loss.

**New module — `src/data_processing/multitask_loader.py`.** `MultiTaskLoader` mixes a dict of per-task `DataLoader`s into a single batch stream. Per emitted batch:
- The whole batch comes from exactly ONE underlying per-task loader (the `collate.py`-enforced single-task batch is preserved — no mixed-task forward passes).
- `batch["task"]` is set to the sampled task name; `batch["labels"]` is rewrapped from `Tensor` into `{task: Tensor}`. The label rewrap is the critical glue that lets the existing `MultiTaskLoss.forward` (which `isinstance(labels, dict)` asserts and iterates `task_names`) consume what `ClassificationDataset` / `MultiLabelDataset` produce. It is idempotent (already-dict labels pass through, but the key must match the sampled task — silent mismatches would cause `MultiTaskLoss` to skip the head).
- Sampling: `"weighted"` for training (probability ∝ `task_weights`, per-task iterators wrap on exhaustion, epoch length = sum of per-task lengths); `"round_robin"` for validation (deterministic; epoch length = `num_tasks × min(per-task lengths)` so every task contributes the same number of eval batches and the per-task metrics are comparable).

**New module — `src/training/create_multitask_trainer_fn.py`.** Factory entry: `create_multitask_trainer_fn(settings, data_bundle, *, tokenizer, enabled_tasks=None, config_path=None) -> Trainer`. Wiring:
1. `set_seed(settings.project.seed)` and resolve device (`settings.training.device` → `"cuda"` if available else `"cpu"`).
2. Resolve task list (`enabled_tasks` ?? `get_all_tasks()`), validate `data_bundle` has both `"train"` and `"val"` for every task.
3. Build per-task `Dataset` + `DataLoader` (using existing `dataset_factory` / `dataloader_factory`).
4. Wrap into `MultiTaskLoader` (`weighted` for train, `round_robin` for val).
5. Build `MultiTaskTruthLensModel(config=...)` with the head set pinned to the resolved tasks (so model `task_logits` keys match what the loader emits — mismatches would silently skip heads in `MultiTaskLoss`). Move to device BEFORE the optimizer (GPU-1 invariant carried over from the single-task factory).
6. `LossEngine` with the FULL `task_types` map (NOT a single-entry dict — that would trigger the `LossEngine.__init__` single-task fast-path which force-disables the EMA normalizer / coverage tracker / `normalization="active"`). `gradient_accumulation_steps` is forwarded so static task weights survive the LOSS-3 pre-scaling.
7. `TaskScheduler(strategy="weighted", task_weights=...)` for the loss-EMA / instrumentation path. Batch-level task selection happens inside `MultiTaskLoader`; the scheduler tracks per-task loss EMAs for adaptive monitoring.
8. `build_optimizer` with resolved `lr` / `weight_decay`.
9. `MonitoringEngine(settings.monitoring)`.
10. `TrainingStep(model, optimizer, scheduler=None, loss_engine, monitor, task_scheduler, config=TrainingStepConfig(grad_accum, max_grad_norm, use_amp))`.
11. `EvaluationEngine(EvaluationConfig(task_types, device))` — uses the full task_types map so every head's per-task metric is reported.
12. `Trainer(...)` with `params_override={"epochs": ...}` so the YAML `training.epochs` always reaches the loop.

**Settings contract.** Reads from the existing `AttrDict` produced by `src.utils.settings.load_settings()`. Every knob has fallbacks across both the spec layout (`settings.training.lr`, `settings.training.use_amp`, `settings.training.batch_size`) AND the live `config/config.yaml` layout (`settings.optimizer.lr`, `settings.precision.use_amp`, `settings.data.batch_size`) so callers don't have to migrate the YAML before adopting the factory. LR has no default — missing both `training.lr` and `optimizer.lr` raises `ValueError` (a silent default would be catastrophic).

**Why the `normalization` choice diverges from the spec.** The spec recommends `normalization="mean"` "for stability", but `"mean"` divides by `len(task_types)` regardless of which heads fired this step. With single-task batches (as `MultiTaskLoader` produces) exactly one head fires per step → the gradient gets shrunk by `1/N` (here `1/6`) for no reason. The factory uses `"active"` instead, which divides by the number of heads that actually contributed → effectively `1/1` per step, equivalent in magnitude to `"sum"` but advertising the multi-task intent. Tested under simulation; matches the existing `MultiTaskLoss` contract.

**Verification.** `Start application` restarts cleanly; `GET /` returns 200; both new modules import without error. `MultiTaskLoader` smoke test passes:
- weighted strategy with `{bias: 3.0, narrative: 1.0}` over loaders of length (2, 1) yields 3 batches with the heavy task seen more often;
- round-robin yields exactly `min(2,1) × 2 = 2` batches in deterministic order;
- pre-wrapped dict labels pass through unchanged;
- pre-wrapped dict labels under the wrong task key raise `KeyError` instead of silently no-op'ing the loss.

**Out of scope (deferred).** LR scheduler wiring (needs explicit `num_training_steps` / `num_warmup_steps` from settings — the existing single-task factory derives them from `len(train_loader) // grad_accum × epochs`, but `MultiTaskLoader.__len__` follows the same contract so this can be added when the YAML grows the keys); ExperimentTracker / CheckpointEngine wiring (factory exposes the slots — caller can attach by patching the returned `Trainer`); GradNorm / Uncertainty balancer (`LossEngine.attach_balancer` is supported, but the choice of balancer is research-level so not hardcoded here).
