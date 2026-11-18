"""Turn raw trip rows into model features. Training and the API both use this file."""

import pandas as pd

from . import config


def add_duration(trips: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with the target: minutes between pickup and dropoff."""
    trips = trips.copy()
    elapsed = trips[config.DROPOFF_TIME] - trips[config.PICKUP_TIME]
    trips[config.TARGET] = elapsed.dt.total_seconds() / 60
    return trips


def filter_outliers(trips: pd.DataFrame) -> pd.DataFrame:
    """Keep plausible trips: 1-60 minutes, 0.1-100 miles, valid zones, in January.

    Expects the duration column from add_duration. Bounds are inclusive.
    """
    keep = (
        trips[config.TARGET].between(
            config.MIN_DURATION_MINUTES, config.MAX_DURATION_MINUTES
        )
        & trips["trip_distance"].between(
            config.MIN_DISTANCE_MILES, config.MAX_DISTANCE_MILES
        )
        & trips["PULocationID"].between(config.MIN_ZONE_ID, config.MAX_ZONE_ID)
        & trips["DOLocationID"].between(config.MIN_ZONE_ID, config.MAX_ZONE_ID)
        & (trips[config.PICKUP_TIME] >= pd.Timestamp(config.PERIOD_START))
        & (trips[config.PICKUP_TIME] < pd.Timestamp(config.PERIOD_END))
    )
    return trips.loc[keep].reset_index(drop=True)


def build_features(trips: pd.DataFrame) -> pd.DataFrame:
    """Return the model's input columns, in a fixed order.

    Needs PULocationID, DOLocationID, trip_distance and the pickup time. The API
    calls this with the fields of a request, so training and serving can't drift.
    """
    pickup = pd.to_datetime(trips[config.PICKUP_TIME])
    features = trips[config.BASE_FEATURES].copy()
    features["pickup_hour"] = pickup.dt.hour
    features["pickup_weekday"] = pickup.dt.day_of_week
    return features[config.FEATURES]


def add_route(features: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with "PU_DO" route strings, for the course's one-hot variant."""
    features = features.copy()
    features[config.ROUTE_FEATURE] = (
        features["PULocationID"].astype(str)
        + "_"
        + features["DOLocationID"].astype(str)
    )
    return features


def split_by_date(trips: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into training trips (before 22 Jan) and validation trips (from 22 Jan)."""
    is_validation = trips[config.PICKUP_TIME] >= pd.Timestamp(config.VALIDATION_START)
    return (
        trips.loc[~is_validation].reset_index(drop=True),
        trips.loc[is_validation].reset_index(drop=True),
    )


def prepare(
    trips: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Raw trips in; X_train, y_train, X_val, y_val out."""
    clean = filter_outliers(add_duration(trips))
    train, val = split_by_date(clean)
    return (
        build_features(train),
        train[config.TARGET],
        build_features(val),
        val[config.TARGET],
    )
