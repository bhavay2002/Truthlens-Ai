from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import api.app as api_app


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """FastAPI test client with a stubbed predictor to avoid model dependency."""

    def _fake_predict(_text: str) -> dict[str, float | str]:
        return {
            "label": "FAKE",
            "fake_probability": 0.81,
            "confidence": 0.81,
        }

    def _fake_predict_batch(texts: list[str]) -> list[dict[str, float | str]]:
        return [_fake_predict(text) for text in texts]

    monkeypatch.setattr(api_app, "predict", _fake_predict)
    monkeypatch.setattr(api_app, "predict_batch", _fake_predict_batch)
    return TestClient(api_app.app)


def test_health_endpoint_returns_status(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert "status" in response.json()


def test_predict_endpoint_returns_response_schema(client: TestClient) -> None:
    response = client.post(
        "/predict",
        json={"text": "Breaking news: stock market rises after policy change."},
    )

    assert response.status_code == 200

    data = response.json()

    assert set(data.keys()) == {"text", "fake_probability", "prediction", "confidence"}
    assert data["prediction"] in {"FAKE", "REAL"}
    assert 0.0 <= data["fake_probability"] <= 1.0
    assert 0.0 <= data["confidence"] <= 1.0


def test_predict_endpoint_rejects_short_text(client: TestClient) -> None:
    response = client.post("/predict", json={"text": "too short"})

    assert response.status_code == 422


def test_analyze_endpoint_returns_response_schema(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        api_app,
        "compute_bias_features",
        lambda _text: SimpleNamespace(
            bias_score=0.1,
            media_bias="Neutral",
            biased_tokens=[],
            sentence_heatmap=[],
        ),
    )
    monkeypatch.setattr(
        api_app.EMOTION_ANALYZER,
        "analyze",
        lambda _text: SimpleNamespace(
            dominant_emotion="neutral",
            emotion_scores={},
            emotion_distribution={},
        ),
    )
    monkeypatch.setattr(api_app, "explain_emotion", lambda _text: {})
    monkeypatch.setattr(
        api_app,
        "explain_prediction",
        lambda *_args, **_kwargs: {"text": "x", "important_features": []},
    )

    response = client.post(
        "/analyze",
        json={"text": "Breaking news: stock market rises after policy change."},
    )

    assert response.status_code == 200
    data = response.json()
    assert set(data.keys()) == {
        "text",
        "prediction",
        "fake_probability",
        "confidence",
        "bias",
        "emotion",
        "explainability",
    }


def test_health_reports_degraded_when_vectorizer_missing(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    for filename in ("config.json", "tokenizer.json", "model.safetensors"):
        (model_dir / filename).write_text("{}", encoding="utf-8")

    monkeypatch.setattr(api_app, "MODEL_PATH", model_dir)
    monkeypatch.setattr(api_app, "TRAINING_TEXT_COLUMN", "engineered_text")
    monkeypatch.setattr(api_app, "VECTORIZER_PATH", tmp_path / "missing.joblib")

    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "degraded"
    assert data["vectorizer_required"] is True
    assert data["vectorizer_exists"] is False
