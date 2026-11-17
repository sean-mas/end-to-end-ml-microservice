# Decisions

Each choice made while building this project, why it was made, and what was not picked.

## Project setup

### 1. One uv project with a `src/` package and an `app/` package

- **Decision:** Training code lives in `src/trip_duration/` and the API lives in `app/`. Both share one `pyproject.toml` and one lock file.
- **Why:** The API has to load a scikit-learn model, so it needs the same libraries as training. A second project would duplicate them without making the image smaller.
- **Not picked:** A separate `api/` project with its own dependencies.

### 2. Runs without a `.env` file

- **Decision:** Every setting has a default in `config.py`. A `.env` file or shell variable only overrides them.
- **Why:** A fresh clone works with no setup step. Docker Compose sets its own values.
- **Not picked:** Making `.env` required.

## Data and features

### 3. Validate on later days, not on a random sample

- **Decision:** Train on trips from 1 to 21 January 2025 and validate on 22 to 31 January.
- **Why:** The service predicts trips that haven't happened yet. A random split lets the model see trips from the same afternoon in both sets, which flatters the score.
- **Not picked:** A random 80/20 split.

### 4. Only features known when the trip is booked

- **Decision:** Pickup zone, dropoff zone, trip distance, pickup hour and pickup weekday.
- **Why:** Fare, tip, tolls and total are only known after the trip ends. A model that uses them would look good in testing and be useless in the API. The distance is the planned distance a booking app would send.
- **Not picked:** Every column in the file.

### 5. Keep trips of 1-60 minutes and 0.1-100 miles in valid zones

- **Decision:** Drop trips outside 1-60 minutes, 0.1-100 miles, zones 1-265, or with a pickup outside January. This keeps 3,325,872 of 3,475,226 trips (95.7%).
- **Why:** These rules come from the course. The histogram in `notebooks/01-explore.ipynb` shows that 97.9% of trips last 1-60 minutes, and the rest include negative durations and multi-day meters. Zero-distance trips (2.6%) carry no signal about speed.
- **Not picked:** Cut-offs chosen by percentile, which would change every month.

### 6. Pickup hour and weekday as features

- **Decision:** Add the pickup hour (0-23) and weekday (Monday = 0) as numbers.
- **Why:** The median speed drops from 16 mph at 05:00 to about 8.5 mph from 11:00 to 18:00, and weekdays are slower than Sundays. Both are known at booking.
- **Not picked:** Holidays or weather. They would need a second data source, which the brief doesn't ask for.

## Experiments and model choice

### 7. MLflow server with SQLite and served artifacts, on port 5001

- **Decision:** `mlflow server --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts --port 5001`.
- **Why:** The model registry needs a database behind it, and SQLite is a single file. With `--artifacts-destination`, clients upload and download models through the server, so a Docker container only needs the server's address, not a mounted folder. Port 5000 is taken by macOS AirPlay Receiver.
- **Not picked:** A plain `./mlruns` folder (no registry), or `--default-artifact-root`, which records paths on this laptop that a container can't see.

### 8. A ladder of four baselines before any tuning

- **Decision:** Log four runs, each adding one idea: always guess the average, the course's linear model on one-hot routes, a Random Forest on zones and distance, then the same forest plus hour and weekday.
- **Why:** Each step shows what one idea is worth. The ladder also proves the forest earns its size: the course's linear model is 0.5 MB.
- **Not picked:** Tuning straight away, which would hide where the gain comes from.

### 9. The forest starts with leaves of at least 50 trips

- **Decision:** 50 trees, depth up to 20, at least 50 trips per leaf.
- **Why:** sklearn's defaults (100 trees, one trip per leaf) would grow millions of leaves per tree and need tens of GB for 2.2 million trips.
- **Not picked:** sklearn defaults.

### 10. Optuna for tuning; trials save scores, not models

- **Decision:** 20 Optuna trials over number of trees, depth, leaf size and share of features per split. Each trial is a child run under one parent run in MLflow.
- **Why:** Optuna picks each next setting from how earlier ones scored, so 20 trials go further than a 20-point grid. Saving 20 forests of 100 MB+ would fill gigabytes, so only the winner is retrained and saved.
- **Not picked:** GridSearchCV, which also uses random cross-validation folds instead of the date split.

### 11. Selection rule, fixed before tuning

- **Decision:** The run with the lowest validation RMSE wins. Runs within 0.05 minutes (3 seconds) of the best count as tied, and the smallest model among them wins.
- **Why:** Writing the rule down first stops it from being bent after seeing results. 3 seconds of error is invisible to a rider, but a model half the size loads faster and costs less memory in the API container.
- **Not picked:** Lowest RMSE alone; best training score (rewards overfitting).

### 12. Models saved with skops, trusting only sklearn and project types

- **Decision:** MLflow saves sklearn models in the skops format. The trust list holds only types from `sklearn.` and `trip_duration.`.
- **Why:** Unlike pickle, skops refuses to load code it wasn't told to trust, so a tampered model file can't run arbitrary code in the API.
- **Not picked:** Pickle, or trusting every type the file contains.

## Serving

### 13. The API loads the model once at startup, by alias

- **Decision:** At startup the API loads `models:/trip-duration@production` from the MLflow registry and keeps it in memory. `/health` and every answer name the version.
- **Why:** A 190 MB model takes about a second to load, which would be added to every request if loaded per call. Loading by alias means promoting a new version needs only an API restart, not a code change.
- **Not picked:** Loading per request (the course demo), or copying a model file into the image.

### 14. The API starts even when MLflow is down

- **Decision:** If the model can't be loaded, the API still starts. `/health` answers 503 with the reason, and `/predict` answers 503.
- **Why:** A container that crashes on startup restarts in a loop and hides the cause. A clear 503 tells the operator what is wrong.
- **Not picked:** Failing at startup.

### 15. Requests are checked against the training ranges

- **Decision:** Zones must be 1-265 and distance 0.1-100 miles, the same bounds as the cleaning rule. Anything else gets a 422 before the model sees it.
- **Why:** The model has never seen trips outside these ranges, so its answer there would be a guess dressed up as a prediction.
- **Not picked:** Accepting any number.

### 16. Behaviour tests run on the real model, outside CI

- **Decision:** API tests use a fake model and run in CI. Five behaviour tests (longer trips take longer, rush hour is slower, the minute of pickup doesn't matter, predictions stay in range, the API matches the model) load the real production model and are marked `integration`.
- **Why:** The real model lives in a local MLflow registry that CI can't reach. The fast tests guard the code; the behaviour tests guard the model.
- **Not picked:** Training a toy model in CI, which would test a model nobody serves.
