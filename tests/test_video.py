import numpy as np
import pytest

from projet_11 import video


class FakeRenderEnv:
    def __init__(self, frames_to_capture=2):
        self.frames_to_capture = frames_to_capture
        self.step_count = 0
        self.closed = False

    def reset(self, seed=None):
        self.seed = seed
        return np.zeros(8, dtype=np.float32), {}

    def step(self, action):
        self.step_count += 1
        terminated = self.step_count >= self.frames_to_capture
        return np.ones(8, dtype=np.float32), 1.0, terminated, False, {}

    def render(self):
        return np.zeros((4, 4, 3), dtype=np.uint8)

    def close(self):
        self.closed = True


class FakeModel:
    def predict(self, observation, deterministic=True):
        return 0, None


def test_record_episode_video_writes_frames(monkeypatch, tmp_path):
    fake_env = FakeRenderEnv(frames_to_capture=2)
    saved = {}

    monkeypatch.setattr(video, "ensure_artifacts_dir", lambda: tmp_path)
    monkeypatch.setattr(video.PPO, "load", lambda path: FakeModel())
    monkeypatch.setattr(video.gym, "make", lambda env_id, render_mode=None: fake_env)
    monkeypatch.setattr(
        video.imageio,
        "mimsave",
        lambda output, frames, fps: saved.update(
            {"output": output, "frame_count": len(frames), "fps": fps}
        ),
    )

    output = video.record_episode_video(
        model_path=tmp_path / "model.zip",
        output_path=tmp_path / "landing.mp4",
        fps=2,
        min_seconds=0,
        max_seconds=2,
        seed=99,
    )

    assert output == tmp_path / "landing.mp4"
    assert saved == {"output": output, "frame_count": 2, "fps": 2}
    assert fake_env.seed == 99
    assert fake_env.closed is True


def test_record_episode_video_rejects_empty_capture(monkeypatch, tmp_path):
    class EmptyEnv(FakeRenderEnv):
        def step(self, action):
            return np.ones(8, dtype=np.float32), 0.0, True, False, {}

    monkeypatch.setattr(video, "ensure_artifacts_dir", lambda: tmp_path)
    monkeypatch.setattr(video.PPO, "load", lambda path: FakeModel())
    monkeypatch.setattr(video.gym, "make", lambda env_id, render_mode=None: EmptyEnv(frames_to_capture=0))

    with pytest.raises(RuntimeError, match="No frames"):
        video.record_episode_video(
            model_path=tmp_path / "model.zip",
            output_path=tmp_path / "landing.mp4",
            fps=2,
            min_seconds=0,
            max_seconds=0,
        )
