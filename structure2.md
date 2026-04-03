# TruthLens AI Repository Structure (Updated)

Snapshot date: 2026-04-03
Scope: code-focused structure, updated for recent refactors and compatibility layers.
Excluded from tree: `.git`, `.pytest_cache`, `__pycache__`, virtualenv folders, generated runtime artifacts.

```text
Truthlens Ai/
|-- .github/
|   `-- workflows/
|       `-- ci.yml
|-- api/
|   `-- app.py
|-- config/
|   |-- config.yaml
|   `-- data_config.yaml
|-- data/
|   |-- __init__.py
|   |-- interim/
|   |-- processed/
|   |-- raw/
|   |-- splits/
|   |-- unified_dataset_test.csv
|   |-- unified_dataset_train.csv
|   `-- unified_dataset_validation.csv
|-- features/                         # compatibility package
|   |-- __init__.py
|   `-- pipelines/
|       |-- __init__.py
|       `-- feature_pipeline.py
|-- logs/
|-- models/
|   |-- __init__.py                  # compatibility package
|   |-- inference/
|   |   |-- __init__.py
|   |   `-- predictor.py
|   |-- roberta_model/
|   |   |-- config.json
|   |   |-- model.safetensors
|   |   |-- tokenizer.json
|   |   |-- tokenizer_config.json
|   |   `-- training_args.bin
|   `-- tfidf_vectorizer.joblib
|-- pipelines/                        # compatibility package
|   `-- __init__.py
|-- reports/
|-- src/
|   |-- aggregation/
|   |   |-- aggregation_pipeline.py
|   |   |-- risk_assessment.py
|   |   |-- score_explainer.py
|   |   |-- score_normalizer.py
|   |   |-- score_schema.py
|   |   |-- truthlens_score_calculator.py
|   |   `-- weight_manager.py
|   |-- analysis/
|   |   |-- argument_mining.py
|   |   |-- bias_profile_builder.py
|   |   |-- context_omission_detector.py
|   |   |-- discourse_coherence_analyzer.py
|   |   |-- emotion_target_analysis.py
|   |   |-- framing_analysis.py
|   |   |-- ideological_language_detector.py
|   |   |-- information_density_analyzer.py
|   |   |-- information_omission_detector.py
|   |   |-- narrative_conflict.py
|   |   |-- narrative_propagation.py
|   |   |-- narrative_role_extractor.py
|   |   |-- narrative_temporal_analyzer.py
|   |   |-- propaganda_pattern_detector.py
|   |   |-- rhetorical_device_detector.py
|   |   `-- source_attribution_analyzer.py
|   |-- data/
|   |   |-- class_balance.py
|   |   |-- clean_data.py
|   |   |-- clean_data2.py
|   |   |-- data_augmentation.py
|   |   |-- data_pipeline.py
|   |   |-- data_profiler.py
|   |   |-- data_split.py
|   |   |-- eda.py
|   |   |-- load_data.py
|   |   |-- merge_datasets.py
|   |   |-- unified_label_schema.py
|   |   `-- validate_data.py
|   |-- evaluation/
|   |   |-- advanced_analysis.py
|   |   |-- calibration.py
|   |   |-- evaluate_model.py
|   |   |-- evaluate_saved_model.py
|   |   |-- evaluation_dashboard.py
|   |   |-- evaluator.py
|   |   |-- metrics.py
|   |   |-- mlflow_tracker.py
|   |   |-- pdf_report.py
|   |   |-- report_writer.py
|   |   |-- task_correlation.py
|   |   `-- uncertainty.py
|   |-- explainability/
|   |   |-- attention_rollout.py
|   |   |-- attention_visualizer.py
|   |   |-- bias_explainer.py
|   |   |-- emotion_explainer.py
|   |   |-- explanation_aggregator.py
|   |   |-- explanation_cache.py
|   |   |-- explanation_consistency.py
|   |   |-- explanation_metrics.py
|   |   |-- explanation_report_generator.py
|   |   |-- explanation_visualizer.py
|   |   |-- lime_explainer.py
|   |   |-- model_explainer.py
|   |   |-- propaganda_explainer.py
|   |   |-- shap_explainer.py
|   |   `-- token_alignment.py
|   |-- features/
|   |   |-- base/
|   |   |-- bias/
|   |   |-- cache/
|   |   |-- discourse/
|   |   |-- emotion/
|   |   |-- fusion/
|   |   |-- graph/
|   |   |-- importance/
|   |   |-- narrative/
|   |   |-- pipelines/
|   |   |-- propaganda/
|   |   |-- text/
|   |   |-- dataset_feature_generator.py
|   |   |-- feature_schema_validator.py
|   |   `-- feature_statistics.py
|   |-- graph/
|   |   |-- entity_graph.py
|   |   |-- graph_analysis.py
|   |   |-- graph_config.py
|   |   |-- graph_embeddings.py
|   |   |-- graph_features.py
|   |   |-- graph_pipeline.py
|   |   |-- graph_utils.py
|   |   |-- graph_visualization.py
|   |   |-- narrative_graph_builder.py
|   |   `-- temporal_graph.py
|   |-- inference/
|   |   |-- analyze_article.py
|   |   |-- batch_inference.py
|   |   |-- feature_preparer.py
|   |   |-- inference_cache.py
|   |   |-- inference_config.py
|   |   |-- inference_engine.py
|   |   |-- inference_logger.py
|   |   |-- model_loader.py
|   |   |-- prediction_pipeline.py
|   |   |-- report_generator.py
|   |   |-- result_formatter.py
|   |   `-- run_inference.py
|   |-- models/
|   |   |-- base/
|   |   |-- calibration/
|   |   |-- checkpointing/
|   |   |-- config/
|   |   |-- encoder/
|   |   |-- ensemble/
|   |   |-- export/
|   |   |-- heads/
|   |   |-- inference/
|   |   |-- metadata/
|   |   |-- multitask/
|   |   |-- registry/
|   |   `-- tasks/
|   |-- pipelines/
|   |   |-- feature_pipeline.py
|   |   |-- prediction_pipeline.py
|   |   |-- preprocessing_pipeline.py
|   |   `-- truthlens_pipeline.py
|   |-- training/
|   |   |-- checkpointing.py
|   |   |-- cross_validation.py
|   |   |-- hyperparameter_tuning.py
|   |   |-- optimizer_factory.py
|   |   |-- scheduler_factory.py
|   |   `-- train_transformer_model.py
|   |-- utils/
|   |   |-- config_loader.py
|   |   |-- device_utils.py
|   |   |-- experiment_utils.py
|   |   |-- helper_functions.py
|   |   |-- input_validation.py
|   |   |-- json_utils.py
|   |   |-- logging_utils.py
|   |   |-- seed_utils.py
|   |   |-- settings.py
|   |   `-- time_utils.py
|   `-- visualization/
|       `-- visualize.py
|-- tests/
|   |-- conftest.py
|   |-- TEST.md
|   |-- test_api.py
|   |-- test_api_error_paths.py
|   |-- test_checkpoint_manager.py
|   |-- test_config_integrity.py
|   |-- test_config_loading.py
|   |-- test_data_augmentation.py
|   |-- test_data_leakage.py
|   |-- test_data_pipeline_module.py
|   |-- test_data_processing.py
|   |-- test_data_validation.py
|   |-- test_dataset_schema.py
|   |-- test_dataset_split_integrity.py
|   |-- test_evaluation.py
|   |-- test_explainability.py
|   |-- test_feature_pipeline.py
|   |-- test_inference_speed.py
|   |-- test_input_validation.py
|   |-- test_logging.py
|   |-- test_model_registry.py
|   |-- test_model_subpackage_imports.py
|   |-- test_model_training.py
|   |-- test_model_utils.py
|   |-- test_multitask_label_helpers.py
|   |-- test_prediction_pipeline.py
|   |-- test_prediction_pipeline_module.py
|   |-- test_prediction_stability.py
|   |-- test_project_structure.py
|   |-- test_reproducibility.py
|   |-- test_settings_compatibility.py
|   |-- test_shap_explainer.py
|   |-- test_tokenization.py
|   |-- test_training_pipeline.py
|   |-- test_unified_label_schema.py
|   `-- test_utils.py
|-- .env
|-- .gitignore
|-- architecture.md
|-- CONTRIBUTING.md
|-- docker-compose.yml
|-- Dockerfile
|-- KNOWLEDGE.md
|-- LICENSE
|-- main.py
|-- PROJECT_REVIEW.md
|-- QUICKSTART.md
|-- README.md
|-- requirements.txt
|-- run_eda.py
|-- save.txt
|-- setup.py
|-- source_scores.json
|-- structure.md
|-- structure2 copy.md
|-- structure2.md
|-- test.py
`-- truthlens_learning_log.xlsx
```

## Recent Structural Changes Reflected

- Added compatibility packages at root:
  - `features/` (with `pipelines/feature_pipeline.py`)
  - `pipelines/` (compat import for `prediction_pipeline`)
  - `models/inference/predictor.py`
  - `data/__init__.py`
- Added `src/evaluation/evaluate_model.py`.
- `src/pipelines/` now contains:
  - `feature_pipeline.py`, `prediction_pipeline.py`, `preprocessing_pipeline.py`, `truthlens_pipeline.py`
- Removed old pipeline files from `src/pipelines`:
  - `data_pipeline.py`
  - `emotion_pipeline.py`
