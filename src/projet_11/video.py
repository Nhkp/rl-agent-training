from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
from stable_baselines3 import PPO

from projet_11.settings import ENV_ID, MODEL_PATH, SEED, VIDEO_PATH, ensure_artifacts_dir


def record_episode_video(
    model_path: str | Path = MODEL_PATH,
    output_path: str | Path = VIDEO_PATH,
    seed: int = SEED,
    fps: int = 30,
    min_seconds: int = 20,
    max_seconds: int = 30,
) -> Path:
    ensure_artifacts_dir()
    output = Path(output_path)
    model = PPO.load(Path(model_path))
    env = gym.make(ENV_ID, render_mode="rgb_array")

    frames = []
    observation, _info = env.reset(seed=seed)
    done = False

    while not done and len(frames) < max_seconds * fps:
        action, _state = model.predict(observation, deterministic=True)
        observation, _reward, terminated, truncated, _info = env.step(int(action))
        frames.append(env.render())
        done = terminated or truncated

    env.close()

    if not frames:
        raise RuntimeError("No frames were captured from the environment.")

    while len(frames) < min_seconds * fps:
        frames.append(frames[-1])

    imageio.mimsave(output, frames[: max_seconds * fps], fps=fps)
    return output
