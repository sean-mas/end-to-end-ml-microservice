"""Feature code: duration, cleaning bounds, time features and the date split."""

import pandas as pd
import pytest

from trip_duration import config, features


def make_trips(rows: list[dict]) -> pd.DataFrame:
    """Build raw trip rows; each row gives pickup time and minutes, plus overrides."""
    records = []
    for row in rows:
        pickup = pd.Timestamp(row.get("pickup", "2025-01-10 08:00"))
        records.append(
            {
                config.PICKUP_TIME: pickup,
                config.DROPOFF_TIME: pickup
                + pd.Timedelta(minutes=row.get("minutes", 10)),
                "PULocationID": row.get("pu", 161),
                "DOLocationID": row.get("do", 236),
                "trip_distance": row.get("miles", 2.0),
                "fare_amount": 15.0,
            }
        )
    return pd.DataFrame(records)


def test_add_duration_in_minutes():
    trips = features.add_duration(make_trips([{"minutes": 12.5}]))
    assert trips[config.TARGET].iloc[0] == pytest.approx(12.5)


def test_add_duration_does_not_change_input():
    raw = make_trips([{}])
    features.add_duration(raw)
    assert config.TARGET not in raw.columns


@pytest.mark.parametrize(
    ("minutes", "kept"),
    [(0.9, False), (1, True), (30, True), (60, True), (60.1, False), (-5, False)],
)
def test_duration_bounds(minutes, kept):
    clean = features.filter_outliers(
        features.add_duration(make_trips([{"minutes": minutes}]))
    )
    assert (len(clean) == 1) is kept


@pytest.mark.parametrize(
    ("miles", "kept"),
    [(0.0, False), (0.09, False), (0.1, True), (100, True), (100.01, False)],
)
def test_distance_bounds(miles, kept):
    clean = features.filter_outliers(
        features.add_duration(make_trips([{"miles": miles}]))
    )
    assert (len(clean) == 1) is kept


@pytest.mark.parametrize(
    ("zone", "kept"), [(0, False), (1, True), (265, True), (266, False)]
)
def test_zone_bounds(zone, kept):
    clean = features.filter_outliers(features.add_duration(make_trips([{"pu": zone}])))
    assert (len(clean) == 1) is kept


@pytest.mark.parametrize(
    ("pickup", "kept"),
    [
        ("2024-12-31 23:50", False),
        ("2025-01-01 00:00", True),
        ("2025-02-01 00:00", False),
    ],
)
def test_trips_outside_january_are_dropped(pickup, kept):
    clean = features.filter_outliers(
        features.add_duration(make_trips([{"pickup": pickup}]))
    )
    assert (len(clean) == 1) is kept


def test_build_features_columns_and_time_parts():
    # 2025-01-10 is a Friday (weekday 4).
    X = features.build_features(make_trips([{"pickup": "2025-01-10 17:45"}]))
    assert list(X.columns) == config.FEATURES
    assert X["pickup_hour"].iloc[0] == 17
    assert X["pickup_weekday"].iloc[0] == 4


def test_build_features_never_uses_fare():
    X = features.build_features(make_trips([{}]))
    assert "fare_amount" not in X.columns


def test_build_features_accepts_string_times_from_the_api():
    request_rows = pd.DataFrame(
        [
            {
                "PULocationID": 161,
                "DOLocationID": 236,
                "trip_distance": 2.0,
                config.PICKUP_TIME: "2025-01-10T17:45:00",
            }
        ]
    )
    assert features.build_features(request_rows)["pickup_hour"].iloc[0] == 17


def test_add_route():
    X = features.add_route(features.build_features(make_trips([{"pu": 7, "do": 9}])))
    assert X[config.ROUTE_FEATURE].iloc[0] == "7_9"


def test_split_by_date_puts_22_january_in_validation():
    trips = make_trips([{"pickup": "2025-01-21 23:59"}, {"pickup": "2025-01-22 00:00"}])
    train, val = features.split_by_date(trips)
    assert len(train) == 1 and len(val) == 1
    assert val[config.PICKUP_TIME].iloc[0] == pd.Timestamp("2025-01-22 00:00")


def test_prepare_returns_aligned_sets():
    trips = make_trips(
        [
            {"pickup": "2025-01-05 09:00"},
            {"pickup": "2025-01-25 09:00"},
            {"minutes": 500},
        ]
    )
    X_train, y_train, X_val, y_val = features.prepare(trips)
    assert len(X_train) == len(y_train) == 1
    assert len(X_val) == len(y_val) == 1
