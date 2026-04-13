from __future__ import annotations

import pytest

from app.api import create_app
from models.train import FEATURE_NAMES, train_and_save_model


@pytest.fixture()
def client(tmp_path):
    artifacts_path = tmp_path / "artifacts"
    train_and_save_model(output_dir=str(artifacts_path))

    app = create_app(model_path=str(artifacts_path / "model.pkl"))
    app.config.update({"TESTING": True})
    return app.test_client()


def _valid_payload() -> dict[str, float]:
    return {
        "credit_limit": 6000,
        "age": 34,
        "bill_amount": 2400,
        "payment_amount": 900,
        "late_payments_6m": 2,
    }


def test_health_endpoint(client):
    response = client.get("/health")
    body = response.get_json()

    assert response.status_code == 200
    assert body["healthy"] is True
    assert body["model_version"].startswith("v1-")


def test_predict_success(client):
    response = client.post("/predict", json=_valid_payload())
    body = response.get_json()

    assert response.status_code == 200
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0
    assert body["model_version"].startswith("v1-")


def test_predict_missing_feature(client):
    payload = _valid_payload()
    payload.pop(FEATURE_NAMES[0])

    response = client.post("/predict", json=payload)
    body = response.get_json()

    assert response.status_code == 400
    assert "Missing required features" in body["error"]


def test_predict_invalid_json(client):
    response = client.post("/predict", data="not-json", content_type="text/plain")
    body = response.get_json()

    assert response.status_code == 400
    assert body["error"] == "Request body must be a JSON object."

