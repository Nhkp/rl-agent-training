from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "mission"
MODEL_PATH = ARTIFACTS_DIR / "best_lunarlander_ppo.zip"
RESULTS_PATH = ARTIFACTS_DIR / "results.csv"
VIDEO_PATH = ARTIFACTS_DIR / "eagle1_landing.mp4"
LOGS_DIR = PROJECT_ROOT / "logs" / "mission"

ENV_ID = "LunarLander-v3"
SEED = 42
OBSERVATION_SIZE = 8


def ensure_artifacts_dir() -> Path:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    return ARTIFACTS_DIR
