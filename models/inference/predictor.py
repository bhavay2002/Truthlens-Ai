from src.models.inference.predictor import Predictor
from src.models.registry.model_registry import ModelRegistry
from src.utils.input_validation import ensure_non_empty_text
from src.utils.settings import load_settings

_SETTINGS = load_settings()

_predictor = None


def _get_predictor():
    global _predictor

    if _predictor is None:
        assets = ModelRegistry.load_model()
        model = assets["model"]

        _predictor = Predictor(model=model)

    return _predictor


def predict(text: str) -> dict:
    ensure_non_empty_text(text)

    predictor = _get_predictor()
    tokenizer = ModelRegistry.load_model()["tokenizer"]

    inputs = tokenizer(
        [text],
        truncation=True,
        padding="max_length",
        max_length=_SETTINGS.model.max_length,
        return_tensors="pt",
    )

    outputs = predictor.predict_batch(inputs)

    return predictor.build_fake_real_output(outputs)


def predict_batch(texts: list) -> list:
    """Run batch inference and return one [real_prob, fake_prob] pair per text."""
    if not isinstance(texts, list) or not texts:
        raise ValueError("texts must be a non-empty list of strings")
    for t in texts:
        ensure_non_empty_text(t)

    predictor = _get_predictor()
    tokenizer = ModelRegistry.load_model()["tokenizer"]

    inputs = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=_SETTINGS.model.max_length,
        return_tensors="pt",
    )

    # outputs contains tensors of shape (N, ...) — one row per text.
    outputs = predictor.predict_batch(inputs)

    import torch as _torch

    n = len(texts)
    results: list = []
    for i in range(n):
        # Slice each tensor to (1, ...) so build_fake_real_output sees batch=1.
        sample: dict = {}
        for k, v in outputs.items():
            if isinstance(v, _torch.Tensor) and v.dim() > 0 and v.size(0) == n:
                sample[k] = v[i : i + 1]
            else:
                sample[k] = v
        r = predictor.build_fake_real_output(sample)
        fake_prob = float(r["fake_probability"])
        results.append([round(1.0 - fake_prob, 6), round(fake_prob, 6)])

    return results