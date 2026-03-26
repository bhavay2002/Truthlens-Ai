# TruthLens AI Repository Structure (Full Snapshot)

This file includes all folders and files recursively, with repeated numeric-suffix variants combined, excluding runtime cache folders and selected Git internals.

Snapshot date: 2026-03-26
Folder count: 62
File count: 223

```text
Truthlens Ai/
|-- .git/
|   |-- hooks/
|   |   |-- applypatch-msg.sample
|   |   |-- commit-msg.sample
|   |   |-- fsmonitor-watchman.sample
|   |   |-- post-update.sample
|   |   |-- pre-applypatch.sample
|   |   |-- pre-commit.sample
|   |   |-- pre-merge-commit.sample
|   |   |-- prepare-commit-msg.sample
|   |   |-- pre-push.sample
|   |   |-- pre-rebase.sample
|   |   |-- pre-receive.sample
|   |   |-- push-to-checkout.sample
|   |   |-- sendemail-validate.sample
|   |   `-- update.sample
|   |-- info/
|   |   `-- exclude
|   |-- logs/
|   |   |-- refs/
|   |   |   |-- heads/
|   |   |   |   |-- Bias-Detection-+-Emotion-/
|   |   |   |   |   `-- -Manipulation-Detection
|   |   |   |   `-- main
|   |   |   `-- remotes/
|   |   |       `-- origin/
|   |   |           |-- Bias-Detection-+-Emotion-/
|   |   |           |   `-- -Manipulation-Detection
|   |   |           `-- main
|   |   `-- HEAD
|   |-- refs/
|   |   |-- heads/
|   |   |   |-- Bias-Detection-+-Emotion-/
|   |   |   |   `-- -Manipulation-Detection
|   |   |   `-- main
|   |   |-- remotes/
|   |   |   `-- origin/
|   |   |       |-- Bias-Detection-+-Emotion-/
|   |   |       |   `-- -Manipulation-Detection
|   |   |       `-- main
|   |   `-- tags/
|   |-- COMMIT_EDITMSG
|   |-- config
|   |-- description
|   |-- FETCH_HEAD
|   |-- HEAD
|   |-- index
|   `-- ORIG_HEAD
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
|   |   `-- merged_dataset.csv
|   |-- processed/
|   |   |-- cleaned_dataset.csv
|   |   `-- test_set.csv
|   |-- raw/
|   |   |-- FakeNewsNet/
|   |   |   |-- gossipcop_fake.csv
|   |   |   |-- gossipcop_real.csv
|   |   |   |-- politifact_fake.csv
|   |   |   |-- politifact_real.csv
|   |   |   `-- README.md
|   |   |-- isot/
|   |   |   |-- dataset1info.txt
|   |   |   |-- Fake.csv
|   |   |   `-- True.csv
|   |   |-- liar_dataset/
|   |   |   |-- analise_sentimento_ok.ipynb
|   |   |   |-- README
|   |   |   |-- test.tsv
|   |   |   |-- test_pos.csv
|   |   |   |-- train.tsv
|   |   |   |-- train_pos.csv
|   |   |   |-- valid.tsv
|   |   |   `-- valid_pos.csv
|   |   `-- dataset.py
|   `-- splits/
|-- experiments/
|-- logs/
|   |-- training.log
|   |-- uvicorn_test.err
|   `-- uvicorn_test.out
|-- models/
|   `-- roberta_model/
|       |-- config.json
|       |-- model.safetensors
|       |-- tokenizer.json
|       |-- tokenizer_config.json
|       `-- training_args.bin
|-- notebooks/
|-- reports/
|   |-- _test_tmp/
|   |   |-- cm_test.png
|   |   `-- cm_test_runtime_check.png
|   |-- figures/
|   |   |-- 2gram_frequency.png
|   |   |-- 3gram_frequency.png
|   |   |-- correlation_matrix.png
|   |   |-- doc_length_by_label.png
|   |   |-- fake_wordcloud.png
|   |   |-- label_distribution.png
|   |   |-- real_wordcloud.png
|   |   |-- text_length_distribution.png
|   |   |-- text_length_outliers.png
|   |   |-- text_length_vs_label.png
|   |   |-- word_frequency.png
|   |   `-- wordcloud.png
|   |-- confusion_matrix.png
|   |-- data_cleaning_report.json
|   |-- eda_report.json
|   |-- eda_summary.json
|   `-- evaluation_results.json
|-- src/
|   |-- aggregation/
|   |   `-- truthlens_score_calculator.py
|   |-- analysis/
|   |   |-- argument_mining.py
|   |   |-- bias_profile_builder.py
|   |   |-- context_omission_detector.py
|   |   |-- emotion_target_analysis.py
|   |   |-- narrative_conflict.py
|   |   `-- narrative_propagation.py
|   |-- data/
|   |   |-- class_balance.py
|   |   |-- clean_data.py
|   |   |-- data_augmentation.py
|   |   |-- data_profiler.py
|   |   |-- data_split.py
|   |   |-- eda.py
|   |   |-- load_data.py
|   |   |-- merge_datasets.py
|   |   `-- validate_data.py
|   |-- evaluation/
|   |   |-- evaluate_model.py
|   |   `-- visualize_metrics.py
|   |-- explainability/
|   |   |-- attention_visualizer.py
|   |   |-- bias_explainer.py
|   |   |-- emotion_explainer.py
|   |   |-- lime_explainer.py
|   |   |-- model_explainer.py
|   |   |-- propaganda_explainer.py
|   |   `-- shap_explainer.py
|   |-- features/
|   |   |-- bias/
|   |   |   |-- bias_detector.py
|   |   |   |-- bias_features.py
|   |   |   |-- bias_lexicon.py
|   |   |   |-- framing_detector.py
|   |   |   |-- ideology_detector.py
|   |   |   |-- narrative_patterns.py
|   |   |   `-- propaganda_detector.py
|   |   |-- discourse/
|   |   |   `-- discourse_features.py
|   |   |-- emotion/
|   |   |   |-- emotion_classifier.py
|   |   |   |-- emotion_detector.py
|   |   |   |-- emotion_feature_extractor.py
|   |   |   |-- emotion_intensity.py
|   |   |   |-- emotion_lexicon.py
|   |   |   |-- emotion_patterns.py
|   |   |   |-- emotion_polarization.py
|   |   |   |-- emotion_score.py
|   |   |   |-- emotion_target.py
|   |   |   |-- emotion_trajectory.py
|   |   |   `-- manipulation_patterns.py
|   |   |-- narrative/
|   |   |   `-- narrative_features.py
|   |   |-- feature_fusion.py
|   |   |-- feature_importance.py
|   |   |-- feature_pipeline.py
|   |   |-- metadata_features.py
|   |   |-- source_features.py
|   |   `-- text_features.py
|   |-- graph/
|   |   |-- entity_graph.py
|   |   |-- graph_analysis.py
|   |   `-- narrative_graph_builder.py
|   |-- inference/
|   |   `-- analyze_article.py
|   |-- models/
|   |   |-- emotion/
|   |   |   |-- load_emotion_model.py
|   |   |   `-- train_emotion_model.py
|   |   |-- encoder/
|   |   |   `-- transformer_encoder.py
|   |   |-- ideology/
|   |   |   `-- ideology_classifier.py
|   |   |-- multitask/
|   |   |   `-- multitask_truthlens_model.py
|   |   |-- narrative/
|   |   |   `-- narrative_detector.py
|   |   |-- propaganda/
|   |   |   `-- propaganda_detector.py
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
|   |   |-- cross_validation.py
|   |   `-- hyperparameter_tuning.py
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
|   |-- test_model_training.py
|   |-- test_model_utils.py
|   |-- test_prediction_pipeline.py
|   |-- test_prediction_stability.py
|   |-- test_project_structure.py
|   |-- test_reproducibility.py
|   |-- test_settings_compatibility.py
|   |-- test_shap_explainer.py
|   |-- test_tokenization.py
|   |-- test_training_pipeline.py
|   `-- test_utils.py
|-- .env
|-- .env.example
|-- .gitignore
|-- architecture.md
|-- CONTRIBUTING.md
|-- docker-compose.yml
|-- Dockerfile
|-- evaluate.py
|-- KNOWLEDGE.md
|-- LICENSE
|-- main.py
|-- PROJECT_REVIEW.md
|-- QUICKSTART.md
|-- README.md
|-- requirements.txt
|-- run_eda.py
|-- setup.py
|-- source_scores.json
|-- structure.md
|-- structure2.md
|-- test.py
`-- truthlens_learning_log.xlsx
```

