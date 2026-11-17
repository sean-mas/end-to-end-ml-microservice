"""Model recipes and the selection rule, on tiny hand-made data (no MLflow)."""

import numpy as np
import pandas as pd
import pytest

from trip_duration import config, models
from trip_duration.selection import apply_rule


@pytest.fixture
def tiny():
    """60 trips where duration grows with distance and is slower at 17:00."""
    rng = np.random.default_rng(0)
    n = 60
    X = pd.DataFrame(
        {
            "PULocationID": rng.integers(1, 10, n),
            "DOLocationID": rng.integers(1, 10, n),
            "trip_distance": rng.uniform(0.5, 10, n),
            "pickup_hour": rng.choice([5, 17], n),
            "pickup_weekday": rng.integers(0, 7, n),
        }
    )[config.FEATURES]
    y = 3 * X["trip_distance"] + np.where(X["pickup_hour"] == 17, 8, 0)
    return X, y


@pytest.mark.parametrize(
    "make",
    [
        models.mean_baseline,
        models.linear_route,
        lambda: models.random_forest(n_estimators=5, min_samples_leaf=2),
        lambda: models.random_forest(
            config.BASE_FEATURES, n_estimators=5, min_samples_leaf=2
        ),
    ],
)
def test_every_model_accepts_the_full_feature_table(tiny, make):
    X, y = tiny
    model = make().fit(X, y)
    assert model.predict(X).shape == (len(X),)
    assert models.size_mb(model) > 0


def test_forest_learns_rush_hour(tiny):
    X, y = tiny
    model = models.random_forest(n_estimators=20, min_samples_leaf=1).fit(X, y)
    trip = X.iloc[[0]].assign(trip_distance=3.0)
    assert (
        model.predict(trip.assign(pickup_hour=17))[0]
        > model.predict(trip.assign(pickup_hour=5))[0]
    )


def runs(rows):
    return pd.DataFrame(rows, columns=["run_name", "val_rmse", "model_mb"])


def test_rule_picks_lowest_rmse_when_clear():
    table = runs([("a", 4.30, 50), ("b", 4.10, 300), ("c", 4.50, 10)])
    assert apply_rule(table)["run_name"] == "b"


def test_rule_picks_smaller_model_within_tolerance():
    table = runs([("big", 4.10, 900), ("small", 4.14, 120), ("tiny", 4.30, 5)])
    assert apply_rule(table, tolerance=0.05)["run_name"] == "small"


def test_rule_tolerance_edge_is_inclusive():
    table = runs([("best", 4.10, 900), ("edge", 4.15, 100)])
    assert apply_rule(table, tolerance=0.05)["run_name"] == "edge"
