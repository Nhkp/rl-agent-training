from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from projet_11.settings import ENV_ID, MODEL_PATH, SEED


def make_env(render_mode: str | None = None):
    return Monitor(gym.make(ENV_ID, render_mode=render_mode))


def evaluate_model(
    model_path: str | Path = MODEL_PATH,
    n_eval_episodes: int = 100,
    deterministic: bool = True,
    seed: int = SEED,
) -> tuple[float, float]:
    env = make_env()
    env.reset(seed=seed)
    model = PPO.load(Path(model_path), env=env)
    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=n_eval_episodes,
        deterministic=deterministic,
    )
    env.close()
    return float(mean_reward), float(std_reward)


def play_episode(model: PPO, deterministic: bool = True, seed: int = SEED) -> dict:
    env = gym.make(ENV_ID)
    observation, _info = env.reset(seed=seed)
    rewards = []
    actions = []
    done = False

    while not done:
        action, _state = model.predict(observation, deterministic=deterministic)
        observation, reward, terminated, truncated, _info = env.step(int(action))
        rewards.append(float(reward))
        actions.append(int(action))
        done = terminated or truncated

    env.close()
    return {
        "total_reward": float(np.sum(rewards)),
        "steps": len(rewards),
        "actions": actions,
    }
