# CLAUDE.md

Instructions for any agent working in this repo. Section 1 is how to work here; Section 2 is what this app is. Before you start a task, find your moment in the router and open what it names, even if you read it earlier in the session.

Precedence: the owner's words in the current session > this file > `~/.claude/CLAUDE.md` > skill defaults.

## Section 1 — How to work here

### Commands

| Do | Command |
|---|---|
| Install | `uv sync` |
| Start MLflow (port 5001) | `docker compose up -d --wait mlflow` |
| Run every stage | `uv run trip-duration all` (about 15 min) |
| One stage | `uv run trip-duration download`, `uv run trip-duration train`, `uv run trip-duration tune`, `uv run trip-duration select`, `uv run trip-duration promote`, `uv run trip-duration evaluate` |
| Start the API (port 8000) | `docker compose up -d --wait api`; after a promotion: `docker compose restart api` |
| Try it | `curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d @sample-request.json` |
| Tests (CI set) | `uv run pytest` (behaviour tests skip unless MLflow is up) |
| Behaviour tests on the real model | `uv run pytest -m integration` |
| Lint | `uv run ruff format . && uv run ruff check .` |
| Rebuild the slides | see the comment at the top of `docs/slides/presentation.html` |

Services: MLflow on `127.0.0.1:5001` (macOS AirPlay holds 5000), API on `127.0.0.1:8000`. MLflow data lives in `./mlflow/`, downloaded data in `./data/raw/`; both are gitignored and never committed.

### Router

| When you are… | The rule that applies | Open |
|---|---|---|
| adding or changing a model input (a feature, a request field) | Only what is known **at booking time**: zones and pickup time. Driven distance, fare, tip, dropoff time leak the answer. Receipt: 2026-11-18, version 1 used `trip_distance` and was retired. | `docs/decisions.md` #4 and #19, `src/trip_duration/features.py`, `transformers.py` |
| retraining or tuning | Runs go to experiment `trip-duration-booking-time`. The selection rule (decision 11) is fixed; don't change it after seeing results. `all` registers a new version and moves `@production`; restart the API. | `docs/decisions.md` #10-11, `src/trip_duration/selection.py` |
| changing the API request or response | Update `app/schemas.py`, `sample-request.json`, the README API section and `tests/test_api.py` in the same change. Unknown fields stay forbidden. | `app/`, `docs/decisions.md` #13-15 |
| writing a number into the README, docs, notebooks or slides | Every figure must come from `trip-duration evaluate`, a notebook output or MLflow. Update all four places together. Receipt: 2026-11-18, a review found quality figures with no source in the repo. | `src/trip_duration/evaluate.py`, `notebooks/02-evaluate.ipynb` |
| editing Python | Run the lint command before committing; CI fails on format. Ruff treats `mlflow` as third-party because `./mlflow/` is a data folder. | `pyproject.toml` |
| saying a stage is done | CI green on the PR; for model or pipeline changes, a clean-copy rerun reproduces the winner; an independent review (`ai-shipping:independent-review`) after the first trained model and before publishing. | `README.md` Quick start |
| changing a command, port, service or convention | Update this file in the same PR. `tests/test_claude_md.py` fails when a CLI command or published port is missing here. | this file |
| doing anything public (publishing, visibility, posting) | Get the owner's clear go first. | — |

### Working rules

- Small PRs, one per milestone. Merge after CI passes; the owner reviews results, not PRs. Report results with evidence and the commands to check them.
- Commit with a heredoc (`git commit -F -`), ending with the `Co-Authored-By` line.
- Docs are plain English, short sentences, exact numbers. Decisions go to `docs/decisions.md` as Decision / Why / Not picked.
- Tests and doc lines go in the same change as the code, never a later step.

## Section 2 — This app

- **What:** predicts NYC yellow-taxi trip duration from pickup zone, dropoff zone and pickup time. Trained on January 2025 TLC data (train 1-21 Jan, validate 22-31 Jan).
- **Shape:** `src/trip_duration/` (pipeline and CLI) → MLflow experiment and registry → `app/` (FastAPI, loads `@production` once at startup) → Docker Compose (`mlflow`, `api`).
- **Audience:** course project (Spiced Academy MLE) and public portfolio piece: github.com/sean-mas/end-to-end-ml-microservice.
- **Status:** version 2 in production (RMSE 4.619 min). Version 1 retired (distance leak). Repo public since 2026-11-19.
