from pathlib import Path

from projet_11 import settings


def test_project_paths_point_to_expected_locations():
    assert settings.PROJECT_ROOT == Path(__file__).resolve().parents[1]
    assert settings.ARTIFACTS_DIR == settings.PROJECT_ROOT / "artifacts" / "mission"
    assert settings.MODEL_PATH.name == "best_lunarlander_ppo.zip"
    assert settings.RESULTS_PATH.name == "results.csv"
    assert settings.VIDEO_PATH.name == "eagle1_landing.mp4"


def test_environment_constants_are_lunar_lander_defaults():
    assert settings.ENV_ID == "LunarLander-v3"
    assert settings.SEED == 42
    assert settings.OBSERVATION_SIZE == 8


def test_ensure_artifacts_dir_creates_directory(tmp_path, monkeypatch):
    target = tmp_path / "artifacts" / "mission"
    monkeypatch.setattr(settings, "ARTIFACTS_DIR", target)

    created = settings.ensure_artifacts_dir()

    assert created == target
    assert target.is_dir()
