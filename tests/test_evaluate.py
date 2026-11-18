"""Evaluation figures on numbers small enough to check by hand."""

import numpy as np
import pandas as pd
import pytest

from trip_duration import evaluate

ACTUAL = np.array([4.0, 8.0, 15.0, 50.0])
PREDICTED = np.array([5.0, 8.0, 14.0, 40.0])  # errors: +1, 0, -1, -10


def test_summary_scores():
    s = evaluate.summary(ACTUAL, PREDICTED)
    assert s["mae"] == pytest.approx(3.0)
    assert s["median_abs_error"] == pytest.approx(1.0)
    assert s["bias"] == pytest.approx(-2.5)
    assert s["rmse"] == pytest.approx(np.sqrt((1 + 0 + 1 + 100) / 4))
    assert s["share_within_1_min"] == pytest.approx(0.75)
    assert s["share_within_10_min"] == pytest.approx(1.0)


def test_by_trip_length_shows_long_trips_underestimated():
    table = evaluate.by_trip_length(ACTUAL, PREDICTED)
    assert table["trips"].sum() == 4
    assert table["bias"].iloc[-1] == pytest.approx(-10.0)  # the 45-60 min band


def test_airport_split():
    trips = pd.DataFrame(
        {"PULocationID": [132, 161, 161, 1], "DOLocationID": [236] * 4}
    )
    split = evaluate.airport_split(trips, ACTUAL, PREDICTED)
    assert split["airport_share"] == pytest.approx(0.25)
    assert split["airport_mae"] == pytest.approx(1.0)
    assert split["other_mae"] == pytest.approx(11 / 3)


def test_by_day_groups_by_date():
    pickup = pd.Series(pd.to_datetime(["2025-01-22 08:00", "2025-01-22 09:00"] * 2))
    daily = evaluate.by_day(pickup, ACTUAL, PREDICTED)
    assert len(daily) == 1
