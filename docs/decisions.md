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
