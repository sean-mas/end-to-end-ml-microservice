"""The typical zone-pair distance: learned from training trips, known at booking."""

import pandas as pd
import pytest

from trip_duration import config
from trip_duration.transformers import ZonePairDistance

TRAIN = pd.DataFrame(
    {
        "PULocationID": [1, 1, 1, 2, 2],
        "DOLocationID": [5, 5, 5, 6, 7],
        "trip_distance": [1.0, 2.0, 9.0, 4.0, 8.0],
    }
)


@pytest.fixture
def fitted():
    return ZonePairDistance().fit(TRAIN)


def booking(pairs):
    return pd.DataFrame(pairs, columns=["PULocationID", "DOLocationID"])


def test_known_pair_gets_its_median(fitted):
    out = fitted.transform(booking([(1, 5)]))
    assert out[config.TYPICAL_DISTANCE].iloc[0] == pytest.approx(2.0)


def test_unseen_pair_falls_back_to_pickup_zone(fitted):
    out = fitted.transform(booking([(2, 99)]))
    assert out[config.TYPICAL_DISTANCE].iloc[0] == pytest.approx(6.0)  # median of 4, 8


def test_unseen_pickup_falls_back_to_all_trips(fitted):
    out = fitted.transform(booking([(42, 5)]))
    assert out[config.TYPICAL_DISTANCE].iloc[0] == pytest.approx(4.0)


def test_driven_distance_never_reaches_the_model(fitted):
    """Even if a driven distance is passed in, it is dropped, not used."""
    with_driven = booking([(1, 5)]).assign(trip_distance=100.0)
    out = fitted.transform(with_driven)
    assert "trip_distance" not in out.columns
    assert out[config.TYPICAL_DISTANCE].iloc[0] == pytest.approx(2.0)


def test_rows_keep_their_order_and_index(fitted):
    X = booking([(2, 7), (1, 5), (2, 6)])
    X.index = [10, 11, 12]
    out = fitted.transform(X)
    assert list(out.index) == [10, 11, 12]
    assert list(out[config.TYPICAL_DISTANCE]) == pytest.approx([8.0, 2.0, 4.0])


def test_does_not_change_the_input(fitted):
    X = booking([(1, 5)]).assign(trip_distance=3.0)
    fitted.transform(X)
    assert list(X.columns) == ["PULocationID", "DOLocationID", "trip_distance"]
