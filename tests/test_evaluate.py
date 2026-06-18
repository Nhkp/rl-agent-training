import numpy as np

from projet_11 import evaluate


class FakeEnv:
    def __init__(self):
        self.closed = False
        self.step_count = 0

    def reset(self, seed=None):
        self.seed = seed
        return np.zeros(8, dtype=np.float32), {}

    def step(self, action):
        self.step_count += 1
        terminated = self.step_count == 2
        reward = 10.0 if action == 1 else 5.0
        return np.ones(8, dtype=np.float32), reward, terminated, False, {}

    def close(self):
        self.closed = True


class FakeModel:
    def predict(self, observation, deterministic=True):
        return 1, None


def test_make_env_wraps_configured_environment(monkeypatch):
    calls = {}

    def fake_make(env_id, render_mode=None):
        calls["env_id"] = env_id
        calls["render_mode"] = render_mode
        return FakeEnv()

    monkeypatch.setattr(evaluate.gym, "make", fake_make)
    monkeypatch.setattr(evaluate, "Monitor", lambda env: ("monitor", env))

    wrapped = evaluate.make_env(render_mode="rgb_array")

    assert wrapped[0] == "monitor"
    assert calls == {"env_id": evaluate.ENV_ID, "render_mode": "rgb_array"}


def test_play_episode_returns_reward_steps_and_actions(monkeypatch):
    fake_env = FakeEnv()
    monkeypatch.setattr(evaluate.gym, "make", lambda env_id: fake_env)

    episode = evaluate.play_episode(FakeModel(), seed=123)

    assert episode == {
        "total_reward": 20.0,
        "steps": 2,
        "actions": [1, 1],
    }
    assert fake_env.seed == 123
    assert fake_env.closed is True


def test_evaluate_model_loads_model_and_closes_env(monkeypatch, tmp_path):
    fake_env = FakeEnv()
    loaded = {}

    class FakePPO:
        @staticmethod
        def load(path, env=None):
            loaded["path"] = path
            loaded["env"] = env
            return FakeModel()

    monkeypatch.setattr(evaluate, "make_env", lambda: fake_env)
    monkeypatch.setattr(evaluate, "PPO", FakePPO)
    monkeypatch.setattr(evaluate, "evaluate_policy", lambda *args, **kwargs: (123.4, 5.6))

    mean_reward, std_reward = evaluate.evaluate_model(tmp_path / "model.zip", n_eval_episodes=3)

    assert mean_reward == 123.4
    assert std_reward == 5.6
    assert loaded["env"] is fake_env
    assert fake_env.closed is True
