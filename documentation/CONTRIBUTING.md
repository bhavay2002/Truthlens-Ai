# Contributing

Thank you for your interest in contributing to TruthLens AI. This document describes the development workflow, coding standards, and how to add new components.

---

## Development Setup

1. Clone the repository and open it in your Replit workspace
2. All Python dependencies are installed automatically via `requirements.txt`
3. Start the development server via the Run button or:
   ```bash
   python -m uvicorn api.app:app --host 0.0.0.0 --port 5000 --reload \
     --reload-dir api --reload-dir src --reload-dir config --reload-dir models
   ```
4. Run the test suite to confirm your setup is working:
   ```bash
   pytest
   ```

---

## Project Structure Overview

```
api/               REST API — only modify app.py for endpoint changes
config/            YAML configuration — no Python logic here
src/               All core logic
  aggregation/     Credibility scoring engine (FeatureMapper → WeightManager → ScoreCalculator)
  analysis/        14 linguistic analysis modules (AnalyzerRegistry + AnalysisPipeline)
  data_processing/ Data loading, cleaning, validation, splitting (8-stage pipeline)
  evaluation/      Metrics and evaluation tools
  explainability/  SHAP, LIME, attention methods
  features/        Feature extractors (most common place to contribute)
  graph/           Entity and narrative graph reasoning
  models/          Model architecture and task heads
  pipelines/       End-to-end pipeline orchestration
  training/        Training utilities
  utils/           Shared configuration, logging, helpers
models/            Trained artifacts and inference helpers
tests/             Test suite — add tests for every change
```

**Important path note:** Data processing code lives in `src/data_processing/` (not `src/data/`). The 8-stage pipeline is in `src/data_processing/data_pipeline.py`. Task schemas are defined in `src/data_processing/data_contracts.py`.

---

## Coding Standards

### Python Style

- Python 3.12+ syntax throughout
- Follow PEP 8 formatting — use `black` for auto-formatting:
  ```bash
  black src/ api/ models/ tests/
  ```
- Use `flake8` for linting:
  ```bash
  flake8 src/ api/ models/
  ```
- Use type hints on all function signatures:
  ```python
  def compute_bias_features(text: str) -> BiasResult:
      ...
  ```
- Prefer `from __future__ import annotations` at the top of each file for forward reference support
- Use `dataclasses` for structured return types — avoid returning bare dicts from public APIs

### Docstrings

All public functions and classes must have docstrings:

```python
def compute_bias_features(text: str) -> BiasResult:
    """
    Compute bias features for a news article.

    Parameters
    ----------
    text : str
        Raw article text to analyze.

    Returns
    -------
    BiasResult
        Structured bias analysis result with score, media bias level,
        biased tokens, and per-sentence heatmap.
    """
```

### Imports

- Absolute imports only: `from src.features.bias.bias_lexicon import BiasResult`
- No relative imports (`.` or `..` style)
- Group imports: standard library → third-party → local, each group separated by a blank line

### No Silent Failures

- Do not silently swallow exceptions — log and re-raise or return a structured error
- Use the existing `logger = logging.getLogger(__name__)` pattern in every module
- FastAPI endpoints should raise `HTTPException` with descriptive `detail` messages

---

## Adding a New Feature Extractor

Feature extractors are the most common type of contribution. Follow this pattern:

**1. Create the module:**

```python
# src/features/bias/my_new_feature.py

from __future__ import annotations
import logging
from src.features.base.base_feature import BaseFeature
from src.analysis.feature_context import FeatureContext

logger = logging.getLogger(__name__)

class MyNewFeature(BaseFeature):
    """Detects [describe what this feature captures]."""

    def extract(self, context: FeatureContext) -> dict[str, float]:
        """
        Extract [description] features from text.

        Returns
        -------
        dict[str, float]
            Flat dictionary of feature_name → numeric value.
        """
        text = context.text
        # ... your logic ...
        return {
            "my_feature_name": 0.0,
        }
```

**2. Add tests:**

```python
# tests/test_features/test_my_new_feature.py

import pytest
from src.analysis.feature_context import FeatureContext
from src.features.bias.my_new_feature import MyNewFeature

def test_basic_extraction():
    extractor = MyNewFeature()
    context = FeatureContext(text="This is a sample news article with some content.")
    features = extractor.extract(context)
    assert "my_feature_name" in features
    assert isinstance(features["my_feature_name"], float)
    assert 0.0 <= features["my_feature_name"] <= 1.0

def test_empty_text_does_not_crash():
    extractor = MyNewFeature()
    context = FeatureContext(text="short words")
    features = extractor.extract(context)
    assert isinstance(features, dict)
```

**3. Register in the feature pipeline** (`src/features/pipelines/feature_pipeline.py`)

**4. Document in `documentation/FEATURE_ENGINEERING.md`**

---

## Adding a New Analysis Analyzer

The `AnalyzerRegistry` (in `src/analysis/analysis_registry.py`) holds all 14 active analyzers, constructed by `build_default_registry()`.

**1. Create the module in `src/analysis/`:**

```python
# src/analysis/my_analyzer.py

from __future__ import annotations
import logging
from src.analysis.base_analyzer import BaseAnalyzer
from src.analysis.feature_context import FeatureContext
from src.analysis.output_models import AnalysisResult

logger = logging.getLogger(__name__)

class MyAnalyzer(BaseAnalyzer):
    """Analyzes [describe the signal]."""

    def analyze(self, context: FeatureContext) -> AnalysisResult:
        ...
```

**2. Register in `build_default_registry()`:**

```python
# src/analysis/analysis_registry.py
reg.register("my_analyzer", MyAnalyzer(), order=15)
```

**Critical:** The registry key must match exactly wherever it is referenced. Any downstream config or code referencing this analyzer must use the exact string `"my_analyzer"`.

**3. Connect output to aggregation** (`src/aggregation/`) if the signal contributes to scoring.

**4. Add tests in `tests/`.**

---

## Adding a New API Endpoint

1. Add the endpoint handler to `api/app.py`
2. Define request/response Pydantic models using `BaseModel`
3. Handle all error cases with appropriate `HTTPException` status codes
4. Update `documentation/API_REFERENCE.md`

Example:

```python
class MyRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=10000)

class MyResponse(BaseModel):
    result: str
    score: float

@app.post("/my-endpoint", response_model=MyResponse)
def my_endpoint(request: MyRequest):
    try:
        # your logic
        return MyResponse(result="ok", score=0.5)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Endpoint error: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error")
```

---

## Modifying the Aggregation Engine

The credibility scoring engine is in `src/aggregation/`. The weight grouping structure is defined in `src/aggregation/aggregation_config.py` as `WEIGHT_GROUPS` — this is the **single source of truth** for which signals belong to which group. Both `weight_manager.py` and `truthlens_score_calculator.py` import from here.

To change how signals are weighted:

1. Edit `WEIGHT_GROUPS` or `TASK_TO_GROUP` in `src/aggregation/aggregation_config.py`
2. Ensure all weight changes are reflected in `documentation/SYSTEM_DESIGN.md`
3. Write a test that verifies the score direction is correct (higher manipulation signals → lower credibility)

Do not define group mappings in `weight_manager.py` directly — any drift from `aggregation_config.py` silently breaks per-group renormalization.

---

## Modifying the Data Pipeline

The 8-stage data pipeline is orchestrated by `src/data_processing/data_pipeline.py`. Task schemas are defined in `src/data_processing/data_contracts.py` — the **single source of truth** for column names and task types.

When adding a new task or changing column naming:
1. Update `data_contracts.py` first
2. Update the relevant dataset source (e.g. `data/raw/emotion/`)
3. Update `config/data_config.yaml` dataset section
4. Run the full pipeline to verify end-to-end compatibility

**Emotion column naming:** Do not introduce new string-named emotion columns. The EMOTION-11 schema uses positional integer columns (`emotion_0` through `emotion_10`). See `src/features/emotion/emotion_schema.py` for the canonical label list.

---

## Writing Tests

All contributions must include tests. The test suite uses `pytest`.

**Test file naming:** `tests/test_{module_name}.py`

**Test function naming:** `test_{what_it_does}_{expected_result}`

**Test categories and location:**

| Category               | Location                      |
|------------------------|-------------------------------|
| Feature extractors     | `tests/test_features/`        |
| Model components       | `tests/test_models/`          |
| Inference / predictor  | `tests/test_inference/`       |
| API endpoints          | `tests/test_api/`             |
| Data processing        | `tests/test_data/`            |
| Explainability         | `tests/test_explainability/`  |
| Utilities              | `tests/test_utils/`           |

**Run tests:**
```bash
pytest                          # All tests
pytest tests/test_features/     # Feature tests only
pytest -k "test_bias"           # Tests matching a pattern
pytest -v                       # Verbose output
pytest --cov=src --cov=api      # With coverage report
```

**Test guidelines:**
- Every public function must have at least one test
- Test edge cases: empty strings, very long strings, non-English text, all-zeros inputs
- Do not write tests that make real HTTP requests to external services
- Mock `ModelRegistry.load_model()` in tests that would otherwise require trained model weights

---

## Configuration Changes

If your contribution requires a new configuration key:

1. Add the key with a sensible default to `config/config.yaml`
2. If it's a data pipeline setting, add it to `config/data_config.yaml`
3. Add the key to `src/utils/settings.py` in the appropriate `@dataclass`
4. Document the new key in `documentation/CONFIGURATION.md`
5. Write a test that verifies the key is loaded correctly

---

## Pull Request Checklist

Before submitting changes:

- [ ] Code passes `black` formatting check
- [ ] Code passes `flake8` linting
- [ ] All new functions have type hints and docstrings
- [ ] Tests are added for all new functionality
- [ ] `pytest` passes with no failures
- [ ] Relevant documentation files are updated
- [ ] No secrets, API keys, or local paths are hardcoded

---

## Documentation Updates

When making changes that affect system behavior, update the relevant docs:

| Change type               | Update these docs                         |
|---------------------------|-------------------------------------------|
| New API endpoint          | `API_REFERENCE.md`                        |
| New feature extractor     | `FEATURE_ENGINEERING.md`                 |
| Architecture change       | `ARCHITECTURE.md`, `SYSTEM_DESIGN.md`    |
| New configuration key     | `CONFIGURATION.md`                        |
| Training pipeline change  | `TRAINING_GUIDE.md`                       |
| Deployment change         | `DEPLOYMENT.md`                           |
| Directory structure change| `PROJECT_STRUCTURE.md`                    |
| Model architecture change | `MODEL_CARD.md`                           |
| Data schema change        | `TRAINING_GUIDE.md`, `CONFIGURATION.md`   |

Also update `replit.md` for any significant architectural changes — this file is always loaded into the agent's memory.
