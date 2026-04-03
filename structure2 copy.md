# TruthLens AI Repository Structure (Current)

Snapshot date: 2026-04-02 20:24:12
Excluded folders: .git, .pytest_cache, __pycache__, venv/.venv/env, Lib/lib/libs/libraries, site-packages, .mypy_cache, .ruff_cache, .tox, .idea, .vscode

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
|   |-- interim/
|   |   |-- bias/
|   |   |   |-- babe_dataset.csv
|   |   |   |-- basil_dataset.csv
|   |   |   `-- mbic_dataset.csv
|   |   |-- emotion/
|   |   |   |-- goemotions.csv
|   |   |   `-- semeval_emotions.csv
|   |   |-- idealogy/
|   |   |   `-- allsides_bias_dataset.csv
|   |   |-- narrative/
|   |   |   |-- framenet_narrative.csv
|   |   |   `-- narrative_frames.csv
|   |   |-- prapoganda/
|   |   |   `-- ptc_propaganda.csv
|   |   |-- framenet_full.csv
|   |   `-- merged_dataset.csv
|   |-- processed/
|   |   |-- bias/
|   |   |   |-- babe_processed.csv
|   |   |   |-- basil_processed.csv
|   |   |   `-- mbic_processed.csv
|   |   |-- emotion/
|   |   |   |-- goemotions_processed2.csv
|   |   |   `-- semeval_emotions_processed2.csv
|   |   |-- ideology/
|   |   |   `-- allsides_processed2.csv
|   |   |-- narrative/
|   |   |   |-- framenet_processed2.csv
|   |   |   `-- narrative_frames_processed2.csv
|   |   |-- propaganda/
|   |   |   |-- propaganda_processed.csv
|   |   |   `-- ptc_propoganda.csv
|   |   |-- cleaned_dataset.csv
|   |   `-- test_set.csv
|   |-- raw/
|   |   `-- liar_dataset/
|   |       `-- analise_sentimento_ok.ipynb
|   |-- splits/
|   |   |-- test/
|   |   |   |-- allsides_test.csv
|   |   |   |-- babe_test.csv
|   |   |   |-- basil_test.csv
|   |   |   |-- framenet_test.csv
|   |   |   |-- goemotion_test.csv
|   |   |   |-- mbic_test.csv
|   |   |   |-- narrative_test.csv
|   |   |   |-- propaganda_test.csv
|   |   |   `-- semeval_emotions_test.csv
|   |   |-- train/
|   |   |   |-- allsides_train.csv
|   |   |   |-- babe_train.csv
|   |   |   |-- basil_train.csv
|   |   |   |-- framenet_train.csv
|   |   |   |-- goemotion_train.csv
|   |   |   |-- mbic_train.csv
|   |   |   |-- narrative_train.csv
|   |   |   |-- propaganda_train.csv
|   |   |   `-- semeval_emotions_train.csv
|   |   `-- validation/
|   |       |-- allsides_validation.csv
|   |       |-- babe_validation.csv
|   |       |-- basil_validation.csv
|   |       |-- framenet_validation.csv
|   |       |-- goemotion_validation.csv
|   |       |-- mbic_validation.csv
|   |       |-- narrative_validation.csv
|   |       |-- propaganda_validation.csv
|   |       `-- semeval_emotions_validation.csv
|   |-- unified_dataset_test.csv
|   |-- unified_dataset_train.csv
|   `-- unified_dataset_validation.csv
|-- experiments/
|-- logs/
|   |-- training.log
|   |-- uvicorn_test.err
|   `-- uvicorn_test.out
|-- models/
|   |-- roberta_model/
|   |   |-- config.json
|   |   |-- model.safetensors
|   |   |-- tokenizer.json
|   |   |-- tokenizer_config.json
|   |   `-- training_args.bin
|   `-- tfidf_vectorizer.joblib
|-- notebooks/
|-- reports/
|   |-- _test_tmp/
|   |   |-- cm_test.png
|   |   `-- cm_test_runtime_check.png
|   |-- Article-Bias-Prediction-(Allsides)/
|   |   |-- 2gram_frequency.png
|   |   |-- 3gram_frequency.png
|   |   |-- dataset_profile.json
|   |   |-- dataset_quality_report.md
|   |   |-- doc_length_by_label.png
|   |   |-- eda_summary.json
|   |   |-- fake_wordcloud.png
|   |   |-- label_distribution.png
|   |   |-- real_wordcloud.png
|   |   |-- text_length_distribution.png
|   |   |-- text_length_outliers.png
|   |   `-- word_frequency.png
|   |-- bias/
|   |   |-- BABE/
|   |   |   |-- 2gram_frequency.png
|   |   |   |-- 3gram_frequency.png
|   |   |   |-- dataset_profile.json
|   |   |   |-- dataset_quality_report.md
|   |   |   |-- doc_length_by_label.png
|   |   |   |-- eda_summary.json
|   |   |   |-- label_distribution.png
|   |   |   |-- text_length_distribution.png
|   |   |   |-- text_length_outliers.png
|   |   |   `-- word_frequency.png
|   |   |-- BASIL/
|   |   |   |-- 2gram_frequency.png
|   |   |   |-- 3gram_frequency.png
|   |   |   |-- dataset_profile.json
|   |   |   |-- dataset_quality_report.md
|   |   |   |-- doc_length_by_label.png
|   |   |   |-- eda_summary.json
|   |   |   |-- label_distribution.png
|   |   |   |-- text_length_distribution.png
|   |   |   |-- text_length_outliers.png
|   |   |   `-- word_frequency.png
|   |   `-- MBIC/
|   |       |-- 2gram_frequency.png
|   |       |-- 3gram_frequency.png
|   |       |-- dataset_profile.json
|   |       |-- dataset_quality_report.md
|   |       |-- doc_length_by_label.png
|   |       |-- eda_summary.json
|   |       |-- label_distribution.png
|   |       |-- text_length_distribution.png
|   |       |-- text_length_outliers.png
|   |       `-- word_frequency.png
|   |-- Framenet/
|   |   |-- 2gram_frequency.png
|   |   |-- 3gram_frequency.png
|   |   |-- dataset_profile.json
|   |   |-- dataset_quality_report.md
|   |   |-- eda_summary.json
|   |   |-- text_length_distribution.png
|   |   |-- text_length_outliers.png
|   |   `-- word_frequency.png
|   |-- goemotion/
|   |   |-- 2gram_frequency.png
|   |   |-- 3gram_frequency.png
|   |   |-- dataset_profile.json
|   |   |-- dataset_quality_report.md
|   |   |-- eda_summary.json
|   |   |-- text_length_distribution.png
|   |   |-- text_length_outliers.png
|   |   `-- word_frequency.png
|   |-- isot+liar+fakenews/
|   |   |-- 2gram_frequency.png
|   |   |-- 3gram_frequency.png
|   |   |-- confusion_matrix copy.png
|   |   |-- confusion_matrix.png
|   |   |-- correlation_matrix copy.png
|   |   |-- correlation_matrix.png
|   |   |-- data_cleaning_report copy.json
|   |   |-- data_cleaning_report.json
|   |   |-- doc_length_by_label.png
|   |   |-- eda_report copy.json
|   |   |-- eda_report.json
|   |   |-- eda_summary.json
|   |   |-- evaluation_results copy.json
|   |   |-- evaluation_results.json
|   |   |-- fake_wordcloud copy.png
|   |   |-- fake_wordcloud.png
|   |   |-- label_distribution.png
|   |   |-- real_wordcloud copy.png
|   |   |-- real_wordcloud.png
|   |   |-- text_length_distribution.png
|   |   |-- text_length_outliers.png
|   |   |-- text_length_vs_label copy.png
|   |   |-- text_length_vs_label.png
|   |   |-- word_frequency.png
|   |   |-- wordcloud copy.png
|   |   `-- wordcloud.png
|   |-- narrative_frames_dataset/
|   |   |-- 2gram_frequency.png
|   |   |-- 3gram_frequency.png
|   |   |-- dataset_profile.json
|   |   |-- dataset_quality_report.md
|   |   |-- eda_summary.json
|   |   |-- text_length_distribution.png
|   |   |-- text_length_outliers.png
|   |   `-- word_frequency.png
|   |-- propaganda2/
|   |   |-- 2gram_frequency.png
|   |   |-- 3gram_frequency.png
|   |   |-- dataset_profile.json
|   |   |-- dataset_quality_report.md
|   |   |-- doc_length_by_label.png
|   |   |-- eda_summary.json
|   |   |-- fake_wordcloud.png
|   |   |-- label_distribution.png
|   |   |-- real_wordcloud.png
|   |   |-- text_length_distribution.png
|   |   |-- text_length_outliers.png
|   |   `-- word_frequency.png
|   |-- ptc/
|   |   |-- 2gram_frequency.png
|   |   |-- 3gram_frequency.png
|   |   |-- dataset_profile.json
|   |   |-- dataset_quality_report.md
|   |   |-- eda_summary.json
|   |   |-- text_length_distribution.png
|   |   |-- text_length_outliers.png
|   |   `-- word_frequency.png
|   `-- semeval/
|       |-- 2gram_frequency.png
|       |-- 3gram_frequency.png
|       |-- dataset_profile.json
|       |-- dataset_quality_report.md
|       |-- eda_summary.json
|       |-- text_length_distribution.png
|       |-- text_length_outliers.png
|       `-- word_frequency.png
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
|   |   |   |-- base_feature.py
|   |   |   |-- feature_config.py
|   |   |   `-- feature_registry.py
|   |   |-- bias/
|   |   |   |-- bias_features.py
|   |   |   |-- bias_lexicon_features.py
|   |   |   |-- framing_features.py
|   |   |   `-- ideological_features.py
|   |   |-- cache/
|   |   |   |-- cache_manager.py
|   |   |   `-- feature_cache.py
|   |   |-- discourse/
|   |   |   |-- argument_structure_features.py
|   |   |   `-- discourse_features.py
|   |   |-- emotion/
|   |   |   |-- emotion_features.py
|   |   |   |-- emotion_intensity_features.py
|   |   |   |-- emotion_lexicon_features.py
|   |   |   |-- emotion_target_features.py
|   |   |   `-- emotion_trajectory_features.py
|   |   |-- fusion/
|   |   |   |-- feature_fusion.py
|   |   |   |-- feature_scaling.py
|   |   |   `-- feature_selection.py
|   |   |-- graph/
|   |   |   |-- entity_graph_features.py
|   |   |   `-- interaction_graph_features.py
|   |   |-- importance/
|   |   |   |-- feature_ablation.py
|   |   |   |-- permutation_importance.py
|   |   |   `-- shap_importance.py
|   |   |-- narrative/
|   |   |   |-- conflict_features.py
|   |   |   |-- narrative_features.py
|   |   |   |-- narrative_frame_features.py
|   |   |   `-- narrative_role_features.py
|   |   |-- pipelines/
|   |   |   |-- batch_feature_pipeline.py
|   |   |   `-- feature_pipeline.py
|   |   |-- propaganda/
|   |   |   |-- manipulation_patterns.py
|   |   |   |-- propaganda_features.py
|   |   |   `-- propaganda_lexicon_features.py
|   |   |-- text/
|   |   |   |-- lexical_features.py
|   |   |   |-- semantic_features.py
|   |   |   |-- syntactic_features.py
|   |   |   `-- token_features.py
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
|   |   `-- result_formatter.py
|   |-- models/
|   |   |-- emotion/
|   |   |   |-- __init__.py
|   |   |   |-- load_emotion_model.py
|   |   |   `-- train_emotion_model.py
|   |   |-- encoder/
|   |   |   |-- __init__.py
|   |   |   `-- transformer_encoder.py
|   |   |-- ideology/
|   |   |   |-- __init__.py
|   |   |   `-- ideology_classifier.py
|   |   |-- multitask/
|   |   |   |-- __init__.py
|   |   |   `-- multitask_truthlens_model.py
|   |   |-- narrative/
|   |   |   |-- __init__.py
|   |   |   `-- narrative_detector.py
|   |   |-- propaganda/
|   |   |   |-- __init__.py
|   |   |   `-- propaganda_detector.py
|   |   |-- __init__.py
|   |   |-- checkpoint_manager.py
|   |   |-- feature_cache.py
|   |   |-- inference.py
|   |   |-- model_config.py
|   |   |-- model_registry.py
|   |   |-- model_utils.py
|   |   |-- predict.py
|   |   |-- prediction_pipeline.py
|   |   |-- train_roberta.py
|   |   `-- truthlens_model.py
|   |-- pipelines/
|   |   |-- data_pipeline.py
|   |   |-- emotion_pipeline.py
|   |   |-- feature_pipeline.py
|   |   |-- preprocessing_pipeline.py
|   |   `-- truthlens_pipeline.py
|   |-- training/
|   |   |-- checkpointing.py
|   |   |-- cross_validation.py
|   |   |-- hyperparameter_tuning.py
|   |   |-- optimizer_factory.py
|   |   `-- scheduler_factory.py
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
|-- .env.example
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
