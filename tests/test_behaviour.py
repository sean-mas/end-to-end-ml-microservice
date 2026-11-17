"""Behaviour tests on the real production model: does it act like a taxi expert?

These need the MLflow server with a promoted model, so they are marked
`integration` and skipped when the registry can't be reached (as in CI).
Run them with: docker compose up -d mlflow && uv run pytest -m integration
"""

import pandas as pd
import pytest
import requests

from trip_duration import config, features

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def service():
    from app.model import load_service

    # Check quickly first: MLflow's own client retries for minutes before failing.
    uri = config.setting("MLFLOW_TRACKING_URI")
    try:
        requests.get(f"{uri}/health", timeout=2).raise_for_status()
    except requests.RequestException:
        pytest.skip(f"MLflow server not reachable at {uri}")
    return load_service()


def predict(service, **trip) -> float:
    base = {
        "PULocationID": 161,  # Midtown Center
        "DOLocationID": 236,  # Upper East Side North
        "trip_distance": 2.4,
        config.PICKUP_TIME: "2025-01-23 17:30",  # Thursday rush hour
    }
    X = features.build_features(pd.DataFrame([{**base, **trip}]))
    return float(service.model.predict(X)[0])


def test_longer_trips_take_longer(service):
    """Directional: more miles, more minutes."""
    minutes = [predict(service, trip_distance=d) for d in (1.0, 3.0, 7.0, 15.0)]
    assert minutes == sorted(minutes)
    assert minutes[-1] > 2 * minutes[0]


def test_rush_hour_is_slower_than_early_sunday(service):
    """Directional: Thursday 17:30 takes longer than Sunday 05:30."""
    rush = predict(service, **{config.PICKUP_TIME: "2025-01-23 17:30"})
    quiet = predict(service, **{config.PICKUP_TIME: "2025-01-26 05:30"})
    assert rush > quiet * 1.2


def test_minute_of_pickup_does_not_matter(service):
    """Invariance: 17:05 and 17:55 are the same hour, so the same prediction."""
    early = predict(service, **{config.PICKUP_TIME: "2025-01-23 17:05"})
    late = predict(service, **{config.PICKUP_TIME: "2025-01-23 17:55"})
    assert early == pytest.approx(late)


def test_predictions_stay_in_a_plausible_range(service):
    """Minimum functionality: every prediction lies inside the training range."""
    for distance in (0.1, 1.0, 10.0, 30.0, 100.0):
        assert 1 <= predict(service, trip_distance=distance) <= 60


def test_api_matches_the_registry_model(service):
    """The API's feature path gives the same answer as calling the model directly."""
    from app.schemas import TripRequest

    trip = TripRequest(
        PULocationID=161,
        DOLocationID=236,
        trip_distance=2.4,
        pickup_datetime="2025-01-23T17:30:00",
    )
    [via_api] = service.predict([trip])
    assert via_api == pytest.approx(predict(service), abs=0.01)
