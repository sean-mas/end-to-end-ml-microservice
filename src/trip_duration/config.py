"""Settings shared by data preparation, training, promotion and the API."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Resolve paths from this file so commands work from any current directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# An optional .env file overrides the MLflow defaults below. Variables already set in
# the shell win, so tests, CI and Docker Compose can point elsewhere.
load_dotenv(PROJECT_ROOT / ".env")

# --- Data source -------------------------------------------------------------------
# Public NYC TLC trip records, one Parquet file per taxi type and month.
SOURCE_BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"
TAXI_TYPE = "yellow"
MONTH = "2025-01"
DOWNLOAD_TIMEOUT_SECONDS = 120
DOWNLOAD_CHUNK_BYTES = 1024 * 1024

# --- Cleaning rules (course standard, checked in notebooks/01-explore.ipynb) --------
MIN_DURATION_MINUTES = 1
MAX_DURATION_MINUTES = 60
MIN_DISTANCE_MILES = 0.1
MAX_DISTANCE_MILES = 100

# Valid TLC taxi zones. 264 and 265 are "unknown" and "outside NYC".
MIN_ZONE_ID = 1
MAX_ZONE_ID = 265

# --- Train / validation split --------------------------------------------------------
# Split by date, not at random: the API predicts trips that haven't happened yet, so
# the model is judged on later days than it learned from.
PERIOD_START = "2025-01-01"
VALIDATION_START = "2025-01-22"
PERIOD_END = "2025-02-01"  # exclusive

# --- Features -----------------------------------------------------------------------
# Only what is known when a trip is booked. Fare, tip and total are known only after
# the trip, so using them would leak the answer into the model.
TARGET = "duration_minutes"
NUMERIC_FEATURES = ["PULocationID", "DOLocationID", "trip_distance", "pickup_hour"]
WEEKDAY_FEATURE = "pickup_weekday"
FEATURES = [*NUMERIC_FEATURES, WEEKDAY_FEATURE]
ROUTE_FEATURE = "trip_route"  # "PU_DO", used by the one-hot comparison run only

RANDOM_STATE = 42

# --- MLflow -------------------------------------------------------------------------
MLFLOW_DEFAULTS = {
    "MLFLOW_TRACKING_URI": "http://127.0.0.1:5000",
    "MLFLOW_EXPERIMENT": "trip-duration",
    "REGISTERED_MODEL_NAME": "trip-duration",
    "MODEL_ALIAS": "production",
}


def setting(name: str) -> str:
    """Return an MLflow setting from the environment or .env, else its default."""
    return os.getenv(name) or MLFLOW_DEFAULTS[name]


def source_url(month: str = MONTH) -> str:
    """Return the TLC download URL for one month."""
    return f"{SOURCE_BASE_URL}/{TAXI_TYPE}_tripdata_{month}.parquet"


def raw_path(month: str = MONTH) -> Path:
    """Return where the downloaded Parquet file for one month is stored."""
    return RAW_DIR / f"{TAXI_TYPE}_tripdata_{month}.parquet"
