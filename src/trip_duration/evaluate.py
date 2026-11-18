"""Stage 5: a closer look at the production model on the validation days.

Every quality figure in the README comes from here, via
`uv run trip-duration evaluate` or notebooks/02-evaluate.ipynb.
"""

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline

from . import config, features

# JFK and LaGuardia taxi zones.
AIRPORT_ZONES = (132, 138)
DURATION_BINS = [1, 5, 10, 20, 30, 45, 60]
WITHIN_MINUTES = (1, 2, 3, 5, 10)


def load_production() -> tuple[Pipeline, str]:
    """Return the model behind the production alias and its version number."""
    mlflow.set_tracking_uri(config.setting("MLFLOW_TRACKING_URI"))
    name = config.setting("REGISTERED_MODEL_NAME")
    version = mlflow.MlflowClient().get_model_version_by_alias(
        name, config.setting("MODEL_ALIAS")
    )
    model = mlflow.sklearn.load_model(f"models:/{name}/{version.version}")
    return model, str(version.version)


def summary(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Headline scores. Bias below zero means predictions are too short."""
    error = predicted - actual
    scores = {
        "rmse": root_mean_squared_error(actual, predicted),
        "mae": float(np.abs(error).mean()),
        "median_abs_error": float(np.median(np.abs(error))),
        "r2": r2_score(actual, predicted),
        "bias": float(error.mean()),
    }
    for minutes in WITHIN_MINUTES:
        scores[f"share_within_{minutes}_min"] = float((np.abs(error) <= minutes).mean())
    return scores


def by_trip_length(actual: np.ndarray, predicted: np.ndarray) -> pd.DataFrame:
    """Error per band of actual duration: where the model is strong and weak."""
    error = predicted - actual
    band = pd.cut(actual, DURATION_BINS, include_lowest=True)
    groups = pd.DataFrame(
        {"band": band, "error": error, "abs_error": np.abs(error)}
    ).groupby("band", observed=True)
    return pd.DataFrame(
        {
            "trips": groups.size(),
            "share": groups.size() / len(actual),
            "mae": groups["abs_error"].mean(),
            "bias": groups["error"].mean(),
        }
    )


def by_day(pickup: pd.Series, actual: np.ndarray, predicted: np.ndarray) -> pd.Series:
    """RMSE per validation day, to check the error is stable over time."""
    squared = pd.Series((predicted - actual) ** 2, index=pickup.index)
    return squared.groupby(pickup.dt.date).mean() ** 0.5


def airport_split(trips: pd.DataFrame, actual, predicted) -> dict[str, float]:
    """MAE for trips to or from JFK/LaGuardia against all other trips."""
    is_airport = (
        trips["PULocationID"].isin(AIRPORT_ZONES)
        | trips["DOLocationID"].isin(AIRPORT_ZONES)
    ).to_numpy()
    abs_error = np.abs(predicted - actual)
    return {
        "airport_share": float(is_airport.mean()),
        "airport_mae": float(abs_error[is_airport].mean()),
        "other_mae": float(abs_error[~is_airport].mean()),
    }


def median_speed_mph(trips: pd.DataFrame) -> float:
    """Median speed, to compare training days with validation days."""
    hours = trips[config.TARGET] / 60
    return float((trips["trip_distance"] / hours).median())


def evaluate(model: Pipeline, trips: pd.DataFrame) -> dict:
    """Score a model on the validation days of raw trips; return every figure."""
    clean = features.filter_outliers(features.add_duration(trips))
    train, val = features.split_by_date(clean)
    actual = val[config.TARGET].to_numpy()
    predicted = model.predict(features.build_features(val))
    daily = by_day(val[config.PICKUP_TIME], actual, predicted)
    return {
        "validation_trips": len(val),
        **summary(actual, predicted),
        "train_rmse": root_mean_squared_error(
            train[config.TARGET], model.predict(features.build_features(train))
        ),
        "daily_rmse_min": float(daily.min()),
        "daily_rmse_max": float(daily.max()),
        **airport_split(val, actual, predicted),
        "median_mph_train": median_speed_mph(train),
        "median_mph_validation": median_speed_mph(val),
        "by_trip_length": by_trip_length(actual, predicted),
    }
