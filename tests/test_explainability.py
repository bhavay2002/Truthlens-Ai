from __future__ import annotations

import pytest

from src.explainability import bias_explainer


def test_explain_bias_returns_expected_sections(monkeypatch: pytest.MonkeyPatch) -> None:
    text = "This shocking news will destroy the country!"

    monkeypatch.setattr(
        bias_explainer,
        "compute_shap_importance",
        lambda *_args, **_kwargs: [{"token": "shocking", "importance": 0.8}],
    )
    monkeypatch.setattr(
        bias_explainer,
        "compute_integrated_gradients",
        lambda *_args, **_kwargs: [{"token": "shocking", "importance": 0.3}],
    )
    monkeypatch.setattr(
        bias_explainer,
        "compute_attention_scores",
        lambda *_args, **_kwargs: [{"token": "shocking", "attention": 0.4}],
    )
    monkeypatch.setattr(
        bias_explainer,
        "compute_sentence_bias",
        lambda _text: [
            {
                "sentence": _text,
                "bias_score": 0.7,
                "biased_tokens": ["shocking"],
            }
        ],
    )

    result = bias_explainer.explain_bias(model=object(), tokenizer=object(), text=text)

    assert set(result.keys()) == {
        "token_importance",
        "integrated_gradients",
        "biased_token_highlights",
        "sentence_bias_scores",
        "attention_scores",
        "bias_heatmap",
    }
    assert result["biased_token_highlights"] == ["shocking"]
