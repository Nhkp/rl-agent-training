import numpy as np
import pytest

from projet_11 import agent


class FakeModel:
    def __init__(self):
        self.received_observation = None
        self.received_deterministic = None

    def predict(self, observation, deterministic=True):
        self.received_observation = observation
        self.received_deterministic = deterministic
        return np.array(2), None


def test_predict_action_validates_observation_and_returns_int(monkeypatch):
    fake_model = FakeModel()
    monkeypatch.setattr(agent, "load_model", lambda: fake_model)

    action = agent.predict_action([0, 0, 0, 0, 0, 0, 0, 0], deterministic=False)

    assert action == 2
    assert fake_model.received_deterministic is False
    assert fake_model.received_observation.dtype == np.float32
    assert fake_model.received_observation.shape == (agent.OBSERVATION_SIZE,)


def test_predict_action_rejects_wrong_observation_shape(monkeypatch):
    monkeypatch.setattr(agent, "load_model", lambda: FakeModel())

    with pytest.raises(ValueError, match="Expected observation shape"):
        agent.predict_action([0, 0, 0])


def test_load_model_raises_when_model_file_is_missing(tmp_path):
    missing_model = tmp_path / "missing.zip"

    with pytest.raises(FileNotFoundError, match="Model not found"):
        agent.load_model(missing_model)
