# TruthLens AI

TruthLens AI is a modular NLP platform for misinformation analysis. The current app supports:

- binary fake-news classification (RoBERTa-based)
- unified multi-task dataset workflows for 7 NLP tasks
- feature engineering, evaluation, explainability, graph analysis, and API serving

## Current Capabilities

- End-to-end training and evaluation for transformer classifiers.
- Data validation and schema normalization for unified datasets.
- Multi-task model components for:
  - `bias_label`
  - `ideology_label`
  - `propaganda_label`
  - `frame`
  - `CO`, `EC`, `HI`, `MO`, `RE`
  - `hero`, `villain`, `victim`
  - `emotion_0` ... `emotion_19`
- FastAPI inference endpoint for deployed binary model usage.
- Cross-validation and hyperparameter tuning utilities that now support configurable label columns.

## Canonical Unified Dataset Column Groups

Text input:
- `title`
- `text`

Media bias task:
- `bias_label`

Ideology classification:
- `ideology_label`

Propaganda detection:
- `propaganda_label`

Narrative framing:
- single-label: `frame`
- multi-label: `CO`, `EC`, `HI`, `MO`, `RE`

Narrative role extraction:
- `hero`
- `villain`
- `victim`
- `hero_entities`
- `villain_entities`
- `victim_entities`

Emotion classification:
- `emotion_0` ... `emotion_19`

Metadata:
- `dataset`

## Key Modules

- Data schema/normalization: `src/data/unified_label_schema.py`
- Data pipeline orchestration: `src/pipelines/data_pipeline.py`
- Feature pipeline: `src/features/feature_pipeline.py`
- Binary model training: `src/models/train_roberta.py`
- Multi-task model: `src/models/multitask/multitask_truthlens_model.py`
- Inference helpers: `src/models/predict.py`, `src/models/inference.py`
- Evaluation and plotting: `src/evaluation/evaluate_model.py`, `src/evaluation/visualize_metrics.py`
- API service: `api/app.py`

## Repository Layout

```text
Truthlens Ai/
  api/
  config/
  data/
  models/
  reports/
  src/
  tests/
  README.md
  architecture.md
  KNOWLEDGE.md
  PROJECT_REVIEW.md
  structure.md
```

## Typical Workflows

1. Train/evaluate binary classifier:
- `python main.py`
- `python evaluate.py`

2. Build unified dataset splits:
- `python "ztest3 copy.py" --split train`
- `python "ztest3 copy.py" --split validation`
- `python "ztest3 copy.py" --split test`

3. Run API:
- `uvicorn api.app:app --reload`

4. Run tests:
- `pytest -q`

## Quality Snapshot

- Latest local test run: `78 passed`.
- Core training/evaluation paths are compatible with configurable label columns and unified schema columns.

## Documentation

- Architecture details: `architecture.md`
- Structure snapshot: `structure.md`
- Deep knowledge map: `KNOWLEDGE.md`
- Current status and gaps: `PROJECT_REVIEW.md`

## License

MIT (see `LICENSE`).
