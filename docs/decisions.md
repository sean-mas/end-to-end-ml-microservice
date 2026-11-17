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
