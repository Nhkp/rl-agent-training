from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from projet_11 import api


@pytest.fixture
def client():
    return TestClient(api.app)


def test_health_reports_model_availability(client, tmp_path, monkeypatch):
    model_path = tmp_path / "model.zip"
    model_path.write_bytes(b"model")
    monkeypatch.setattr(api, "MODEL_PATH", model_path)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_available": True,
        "model_path": str(model_path),
    }


def test_play_returns_predicted_action(client, monkeypatch):
    monkeypatch.setattr(api, "predict_action", lambda observation, deterministic=True: 3)

    response = client.post(
        "/play",
        json={"observation": [0, 0, 0, 0, 0, 0, 0, 0], "deterministic": False},
    )

    assert response.status_code == 200
    assert response.json() == {"action": 3}


def test_play_returns_503_when_model_is_missing(client, monkeypatch):
    def raise_missing_model(observation, deterministic=True):
        raise FileNotFoundError("missing model")

    monkeypatch.setattr(api, "predict_action", raise_missing_model)

    response = client.post(
        "/play",
        json={"observation": [0, 0, 0, 0, 0, 0, 0, 0]},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "missing model"


def test_play_returns_422_for_invalid_observation_from_agent(client, monkeypatch):
    def raise_invalid_observation(observation, deterministic=True):
        raise ValueError("invalid observation")

    monkeypatch.setattr(api, "predict_action", raise_invalid_observation)

    response = client.post(
        "/play",
        json={"observation": [0, 0, 0, 0, 0, 0, 0, 0]},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "invalid observation"


def test_play_request_schema_rejects_short_observation(client):
    response = client.post("/play", json={"observation": [0, 0, 0]})

    assert response.status_code == 422


def test_play_response_schema_accepts_integer_action():
    assert api.PlayResponse(action=1).action == 1


def test_play_request_defaults_to_deterministic():
    request = api.PlayRequest(observation=[0, 0, 0, 0, 0, 0, 0, 0])

    assert request.deterministic is True
    assert isinstance(api.MODEL_PATH, Path)
