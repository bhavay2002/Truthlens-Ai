# Troubleshooting

This document covers **common issues and their solutions** when running, training, or deploying TruthLens AI.

---

## API and Server Issues

### Server won't start — "No module named uvicorn"

**Cause:** Python packages are not installed.

**Fix:**
```bash
pip install -r requirements.txt
```

If the issue persists, ensure the Python environment is the same one that installed the packages:
```bash
python -c "import uvicorn; print(uvicorn.__version__)"
```

---

### Server keeps restarting repeatedly

**Cause:** The `--reload` flag is watching `.pythonlibs/` and triggering restarts when packages are installed.

**Fix:** Use the scoped reload command (already configured in the Replit workflow):
```bash
python -m uvicorn api.app:app --host 0.0.0.0 --port 5000 --reload \
  --reload-dir api --reload-dir src --reload-dir config --reload-dir models
```

---

### `GET /health` returns `"status": "degraded"`

**Cause:** The model has not been trained yet. Model files are missing from `models/truthlens_model/`.

**Fix:** Train the model:
```bash
python main.py
```

After training, verify the checkpoint is present:
```bash
ls models/checkpoints/
# checkpoint.pt  checkpoint.meta.json
```

---

### `POST /predict` returns `503 Service Unavailable`

**Cause:** Same as above — model checkpoint is missing.

**Detail message:** `"Model not available. Please train the model first."`

**Fix:** Run `python main.py` to train the model.

---

### `POST /predict` returns `400 Bad Request`

**Cause:** The request text is too short (< 10 characters) or too long (> 10,000 characters), or the JSON is malformed.

**Fix:** Ensure the request body is valid JSON with a `text` field between 10 and 10,000 characters:
```json
{"text": "Your news article text here, at least 10 characters."}
```

---

### `POST /analyze` is very slow

**Cause:** LIME explanation generation runs 256 perturbation passes. This is expected, especially on CPU.

**Expected times:**
- GPU: 2–8 seconds
- CPU: 10–60 seconds depending on article length

**Fix (reduce LIME samples):** Edit `api/app.py`:
```python
LIME_NUM_SAMPLES = 64   # Reduce from 256 for faster responses at lower quality
```

---

### LIME returns `"error": "lime_unavailable"` in the response

**Cause:** LIME failed internally, usually because the model is not trained or an unexpected output format was encountered.

**Note:** This is a handled error — the `/analyze` endpoint still returns a complete response with `"error": "lime_unavailable"` in the LIME field. The endpoint does not fail.

**Fix:** Ensure the model is trained and the `/predict` endpoint works first. LIME calls `predict_batch()` internally.

---

### Configuration file not found

**Error:** `FileNotFoundError: Configuration file not found: .../config/config.yaml`

**Cause:** The working directory is not the project root.

**Fix:** Always run commands from the project root:
```bash
cd /home/runner/workspace
python main.py
```

---

### `load_app_config()` raises `ValueError: Missing required config sections`

**Cause:** The `config.yaml` is missing required top-level keys (`model`, `data`, `training`).

**Fix:** Ensure `config/config.yaml` has all three sections. Compare with the reference in `documentation/CONFIGURATION.md`.

---

## Training Issues

### Training fails with `FileNotFoundError: data/splits/train.csv not found`

**Cause:** The data pipeline hasn't been run yet, or raw data is missing.

**Fix:** Ensure raw datasets exist in `data/raw/` and run the full pipeline:
```bash
python main.py
```

The pipeline auto-generates split files before the training loop starts.

---

### Data contracts validator rejects emotion columns

**Error:** `ValueError: Column 'emotion_joy' not in schema` or similar

**Cause:** Emotion dataset has legacy column names (`emotion_joy`, `emotion_fear`, etc.) from the 20-label GoEmotions schema. The live EMOTION-11 schema uses positional integer column names.

**Fix:** Rename emotion columns to the positional format:

| Old name (reject)   | New name (required) | Label        |
|---------------------|---------------------|--------------|
| `emotion_0`         | `emotion_0`         | `neutral`    |
| `emotion_joy` ✗     | `emotion_1`         | `admiration` |
| `emotion_fear` ✗    | `emotion_4`         | `annoyance`  |
| ...                 | `emotion_10`        | `anger`      |

See `src/data_processing/data_contracts.py` and `src/features/emotion/emotion_schema.py` for the authoritative column-to-label mapping.

---

### CUDA out of memory during training

**Cause:** Batch size is too large for available GPU memory.

**Fix:** Reduce batch size and increase gradient accumulation to maintain effective batch size:
```yaml
data:
  batch_size: 16                   # Reduce from 32
training:
  gradient_accumulation_steps: 4   # Effective batch = 16 × 4 = 64 (unchanged)
```

Or disable gradient checkpointing (trades memory efficiency for speed — only if memory is not the constraint):
```yaml
model:
  gradient_checkpointing: false
```

---

### Training on CPU is extremely slow

**Cause:** RoBERTa model training requires significant compute. CPU training can take hours per epoch.

**Fix options:**
1. Reduce model size: `encoder: distilroberta-base` (lighter, faster)
2. Reduce batch size and accumulation for more frequent gradient updates per epoch
3. Disable `torch_compile` on CPU if it causes overhead:
   ```yaml
   model:
     torch_compile: false
   ```
4. Use a GPU environment if available

---

### Training converges poorly / loss doesn't decrease

**Possible causes and fixes:**

| Symptom                     | Likely cause                    | Fix                                    |
|-----------------------------|----------------------------------|----------------------------------------|
| Loss doesn't decrease at all | Learning rate too high          | Reduce learning rate by 5–10×          |
| Loss oscillates wildly      | Gradient exploding              | Ensure `max_grad_norm: 1.0`            |
| Fast convergence then plateau| Schedule issue                  | Increase warmup ratio                  |
| Validation loss goes up     | Overfitting                     | Add more data or increase dropout      |
| NaN losses under BF16       | BF16 overflow (rare on bf16)    | Switch `amp_dtype: "fp16"` and lower `grad_scaler_init_scale` |

---

### Early stopping triggers immediately (before epoch 4)

**Cause:** `min_epochs` guards against premature stopping. If early stopping fires before epoch 4, this indicates a configuration issue.

**Expected behavior:** Early stopping is disabled until `min_epochs: 4` is reached. After that it requires `patience: 2` consecutive epochs without improvement exceeding `min_delta: 0.003`.

**Fix if still triggering too early:**
```yaml
training:
  min_epochs: 6              # Extend the minimum run
  early_stopping_patience: 3  # Increase grace period
```

---

### Data augmentation is slow

**Cause:** Synonym replacement requires NLTK WordNet lookups.

**Fix:** Disable augmentation while developing:
```yaml
augmentation:
  enabled: false
```

Re-enable for final training runs.

---

### `data_config.yaml` validation fails — class imbalance

**Error:** Class ratio below `min_class_ratio`

**Cause:** One label class is underrepresented in your dataset.

**Fix options:**
1. Add more samples of the minority class to `data/raw/`
2. Enable oversampling:
   ```yaml
   balancing:
     enabled: true
     method: oversample
   ```
3. Lower the threshold (not recommended for production):
   ```yaml
   validation:
     label_checks:
       min_class_ratio: 0.05
   ```

---

### Leakage check fails

**Error:** `DataLeakageError: N samples from train found in test`

**Cause:** Raw data contains duplicate texts that span splits.

**Fix:** Review and deduplicate your raw datasets in `data/raw/`. The leakage checker compares by text hash before augmentation, so removing source duplicates is the only reliable fix. Do not disable the leakage check in production.

---

## Feature Engineering Issues

### `BiasLexiconFeatures` returns all zeros

**Cause:** The article uses bias terms not in the internal lexicon, or the text doesn't contain obvious bias markers.

**Note:** This is expected behavior for neutral, factual text. A bias score near zero is a correct output.

---

### `EmotionLexiconAnalyzer` returns `"neutral"` for all articles

**Cause:** The lexicon-based emotion extractor relies on a predefined lexicon mapped to the EMOTION-11 labels. If the input uses uncommon emotional vocabulary, scores will be low.

**Note:** `"neutral"` is the correct output when no EMOTION-11 lexicon terms are matched. The emotion model head (transformer-based) may still detect emotion signals at inference time even when lexicon scores are zero.

---

### Feature extraction is slow on large datasets

**Fix:** Enable feature caching:

```python
# Use batch_feature_pipeline.py for dataset-scale extraction
# src/features/cache/cache_manager.py handles lifecycle
```

Or run feature extraction once and save to CSV before training.

---

## Import and Module Errors

### `ModuleNotFoundError: No module named 'src'`

**Cause:** Running Python from a subdirectory instead of the project root.

**Fix:**
```bash
cd /home/runner/workspace
python -c "from src.utils.settings import load_settings; print('OK')"
```

---

### `ModuleNotFoundError: No module named 'src.data'`

**Cause:** Outdated import path. The data processing module was reorganized.

**Fix:** Replace `from src.data` with `from src.data_processing`:
```python
# Wrong:
from src.data.data_loader import load_dataframe
# Correct:
from src.data_processing.data_loader import load_dataframe
```

---

### `ModuleNotFoundError: No module named 'models.inference'`

**Cause:** Project root is not in the Python path.

**Fix:** Always run from `/home/runner/workspace`. The Replit workflow does this automatically.

---

### Circular import errors at startup

**Cause:** Two modules are importing each other at the module level.

**Fix:** Move one of the imports inside the function that uses it:

```python
# Instead of top-level:
from src.models.registry.model_registry import ModelRegistry

# Inside the function:
def load_model_and_tokenizer():
    from src.models.registry.model_registry import ModelRegistry
    ...
```

This pattern is already used in `models/inference/predictor.py`.

---

### Analysis registry key mismatch

**Error:** `KeyError: 'bias_profiler'` or analyzer not running as expected

**Cause:** A config or downstream reference uses a registry key that doesn't match the key used in `build_default_registry()`.

**Fix:** Use the exact keys from `src/analysis/analysis_registry.py`:
`rhetorical`, `argument`, `context`, `discourse`, `emotion`, `framing`, `information`, `information_omission`, `ideology`, `narrative_role`, `narrative_conflict`, `narrative_propagation`, `narrative_temporal`, `source`

---

## Test Failures

### Tests fail with `FileNotFoundError` for model artifacts

**Cause:** Tests that require trained model weights are running without a trained model.

**Fix:** Mock the model loading in tests:

```python
from unittest.mock import patch, MagicMock

def test_predict_with_mock_model():
    mock_model = MagicMock()
    mock_model.config.label2id = {"REAL": 0, "FAKE": 1}
    mock_tokenizer = MagicMock()

    with patch("models.inference.predictor.load_model_and_tokenizer",
               return_value=(mock_tokenizer, mock_model)):
        from models.inference.predictor import predict
        # test here
```

---

### Tests are slow

**Cause:** Feature extractors or model loading is happening in tests.

**Fix:** Use fixtures and mocks for expensive operations. Only load real models in integration tests.

---

### `pytest` not found

**Fix:**
```bash
pip install pytest
```

Or run with the full path:
```bash
python -m pytest
```

---

## Getting Help

1. Check the other documentation files:
   - [CONFIGURATION.md](CONFIGURATION.md) — for config-related issues
   - [TRAINING_GUIDE.md](TRAINING_GUIDE.md) — for training pipeline issues
   - [API_REFERENCE.md](API_REFERENCE.md) — for endpoint behavior questions
   - [ARCHITECTURE.md](ARCHITECTURE.md) — for understanding how components connect

2. Check the application logs:
   - Development: workflow console output in Replit
   - Training: `logs/training.log`
   - Production: Replit deployment logs panel

3. Run the test suite to identify broken components:
   ```bash
   pytest -v 2>&1 | head -50
   ```

4. Check which config is being loaded:
   ```bash
   python -c "from src.utils.config_loader import load_config; import json; print(json.dumps(load_config(), indent=2, default=str))"
   ```
