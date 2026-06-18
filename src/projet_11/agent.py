from functools import lru_cache
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from projet_11.settings import MODEL_PATH, OBSERVATION_SIZE


@lru_cache(maxsize=1)
def load_model(model_path: str | Path = MODEL_PATH) -> PPO:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Train the mission notebook first."
        )
    return PPO.load(path)


def predict_action(observation: list[float] | np.ndarray, deterministic: bool = True) -> int:
    observation_array = np.asarray(observation, dtype=np.float32)
    if observation_array.shape != (OBSERVATION_SIZE,):
        raise ValueError(
            f"Expected observation shape ({OBSERVATION_SIZE},), got {observation_array.shape}."
        )

    model = load_model()
    action, _state = model.predict(observation_array, deterministic=deterministic)
    return int(action)
