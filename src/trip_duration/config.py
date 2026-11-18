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

# Valid TLC taxi zones. 264 ("unknown") and 265 ("outside of NYC") are kept:
# real trips use them, and a booking outside the city still needs an estimate.
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
# the trip, and so is the distance actually driven: using them would leak the answer
# into the model. The driven distance is used once, to learn the typical distance
# between two zones (transformers.ZonePairDistance), and is then dropped.
TARGET = "duration_minutes"
PICKUP_TIME = "tpep_pickup_datetime"
DROPOFF_TIME = "tpep_dropoff_datetime"
ZONES = ["PULocationID", "DOLocationID"]
TIME_FEATURES = ["pickup_hour", "pickup_weekday"]  # weekday: Monday=0 ... Sunday=6
# What the API receives: zones and pickup time.
BOOKING_FEATURES = [*ZONES, *TIME_FEATURES]
# Added inside every model pipeline from the zones (median of training trips).
TYPICAL_DISTANCE = "typical_distance"
# Zone IDs go in as plain numbers: trees split on them directly, so no one-hot needed.
BASE_FEATURES = [*ZONES, TYPICAL_DISTANCE]
FEATURES = [*BASE_FEATURES, *TIME_FEATURES]
ROUTE_FEATURE = "trip_route"  # "PU_DO" string, used by the one-hot comparison run

RANDOM_STATE = 42

# --- MLflow -------------------------------------------------------------------------
# Port 5001, not 5000: macOS uses 5000 for AirPlay Receiver.
MLFLOW_DEFAULTS = {
    "MLFLOW_TRACKING_URI": "http://127.0.0.1:5001",
    "MLFLOW_EXPERIMENT": "trip-duration-booking-time",
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
