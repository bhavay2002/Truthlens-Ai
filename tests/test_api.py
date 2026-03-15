from __future__ import annotations

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

    monkeypatch.setattr(api_app, "predict", _fake_predict)
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
