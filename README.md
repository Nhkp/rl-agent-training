# Eagle-1 LunarLander RL Agent

![CI](https://github.com/Nhkp/rl-agent-training/actions/workflows/ci.yml/badge.svg)
![Coverage](coverage.svg)

Eagle-1 is a reinforcement learning project built around the `LunarLander-v3`
environment from Gymnasium. The goal is simple to state and harder to solve:
train an autonomous agent that can control a lunar lander, slow its descent, and
touch down between the landing flags with a strong reward.

The project includes the training notebooks, trained PPO models, experiment
results, a FastAPI inference endpoint, a Streamlit dashboard, and a replay video
showing a successful landing.

## Project Overview

This work explores how an agent can learn control behavior from interaction
rather than from manually written rules. At each step, the lander observes the
state of the environment and chooses one discrete action. The learning objective
is to maximize the cumulative reward over an episode.

The final agent is trained with Proximal Policy Optimization (PPO), a policy
gradient algorithm provided by Stable-Baselines3. Several hyperparameters were
tested to compare their impact on learning stability and final performance.

## What the Agent Learns

The agent receives an 8-dimensional observation describing the lander's state,
including position, velocity, angle, angular velocity, and leg contact signals.
It learns to choose between the available engine actions in order to:

- reduce horizontal and vertical speed before landing;
- keep the lander stable and upright;
- avoid crashing or drifting away from the landing zone;
- maximize the final episode reward.

The main reinforcement learning notions covered in this project are reward
optimization, policy evaluation, exploration versus exploitation, and
hyperparameter comparison.

## Technical Approach

The project uses:

- Python 3.12 for the codebase;
- Gymnasium with Box2D for the `LunarLander-v3` environment;
- Stable-Baselines3 for PPO training and inference;
- TensorBoard logs for experiment tracking;
- FastAPI for serving the trained policy through an HTTP API;
- Streamlit for visualizing results and replaying the trained agent;
- pytest and coverage for automated quality checks.

The trained model is loaded from `artifacts/mission/best_lunarlander_ppo.zip`.
The experiment summary is stored in `artifacts/mission/results.csv`.

## Experiments and Results

Several PPO configurations were trained and evaluated over 50 episodes. The best
result in the current experiment table is:

| Experiment | Mean reward | Std reward | Total timesteps |
| --- | ---: | ---: | ---: |
| `gae_lambda_098` | 248.54 | 42.26 | 350,000 |

Full experiment summary:

| Experiment | Mean reward | Std reward | Main parameter change |
| --- | ---: | ---: | --- |
| `baseline` | 51.95 | 124.30 | default configuration |
| `learning_rate_1e_4` | 6.46 | 101.93 | `learning_rate=0.0001` |
| `gamma_0995` | 232.54 | 52.13 | `gamma=0.995` |
| `n_steps_2048` | 2.41 | 144.17 | `n_steps=2048` |
| `batch_size_128` | 141.13 | 113.35 | `batch_size=128` |
| `gae_lambda_098` | 248.54 | 42.26 | `gae_lambda=0.98` |
| `clip_range_01` | 84.61 | 124.25 | `clip_range=0.1` |

The strongest configuration improves both average reward and consistency. In
practice, this means the trained policy is much more likely to complete a clean
landing rather than relying on occasional lucky episodes.

## Successful Landing Demo

The repository includes a recorded landing produced by the trained agent:

[Watch the successful landing replay](artifacts/mission/eagle1_landing.mp4)

<video src="artifacts/mission/eagle1_landing.mp4" controls width="720">
  Your browser does not support embedded videos. Open
  artifacts/mission/eagle1_landing.mp4 instead.
</video>

## How to Run the Project

Install the project and its dependencies with uv:

```bash
uv sync
```

Run the Streamlit dashboard:

```bash
uv run streamlit run src/projet_11/app.py
```

Run the FastAPI service:

```bash
uv run uvicorn projet_11.api:app --reload
```

Check the API health endpoint:

```bash
curl http://127.0.0.1:8000/health
```

Ask the trained agent for an action:

```bash
curl -X POST http://127.0.0.1:8000/play \
  -H "Content-Type: application/json" \
  -d '{"observation":[0,0,0,0,0,0,0,0],"deterministic":true}'
```

## API and Streamlit Dashboard

The FastAPI app exposes two endpoints:

- `GET /health`: verifies that the service is running and reports whether the
  trained model artifact is available.
- `POST /play`: receives an 8-value observation and returns the action selected
  by the PPO policy.

The Streamlit dashboard displays experiment results, reward charts, model
availability, and the successful landing replay. It can also generate a fresh
episode video from the saved model.

## Tests and Continuous Integration

The project includes unit tests for the core Python modules:

- model loading and action prediction validation;
- FastAPI health and prediction routes;
- episode evaluation logic;
- replay video generation logic;
- project settings and artifact paths.

Run the test suite locally:

```bash
uv sync --group dev
uv run pytest --cov=src/projet_11 --cov-report=term-missing --cov-report=xml --cov-fail-under=80
uv run coverage-badge -o coverage.svg -f
```

GitHub Actions runs the same checks on every push and pull request. The CI fails
if tests fail, if coverage drops below 80%, or if the local coverage badge is not
up to date.

## Limitations and Next Steps

The current agent performs well in the evaluated setup, but the work could be
extended by:

- running more seeds per experiment to better measure stability;
- tracking full learning curves in the README or dashboard;
- comparing PPO with other RL algorithms;
- adding integration tests for the Streamlit dashboard;
- publishing experiment artifacts through a more formal model registry.
