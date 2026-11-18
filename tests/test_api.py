"""API routes with a fake model, so these tests need no MLflow server."""

import pytest
from fastapi.testclient import TestClient

from app import main
from app.model import ModelService
from trip_duration import config

TRIP = {
    "PULocationID": 161,
    "DOLocationID": 236,
    "pickup_datetime": "2025-01-23T17:30:00",
}


class FakeModel:
    """Predicts pickup hour + 5 minutes, and records the table it was given."""

    def __init__(self):
        self.seen = None

    def predict(self, X):
        self.seen = X
        return X["pickup_hour"].to_numpy() + 5.0


@pytest.fixture
def fake_model():
    return FakeModel()


@pytest.fixture
def client(monkeypatch, fake_model):
    service = ModelService(model=fake_model, name="trip-duration", version="7")
    monkeypatch.setattr(main.model_module, "load_service", lambda: service)
    with TestClient(main.app) as c:
        yield c


@pytest.fixture
def client_without_model(monkeypatch):
    def fail():
        raise ConnectionError("MLflow not reachable")

    monkeypatch.setattr(main.model_module, "load_service", fail)
    with TestClient(main.app) as c:
        yield c


def test_health_reports_loaded_version(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_version"] == "7"


def test_predict_returns_minutes_and_version(client):
    response = client.post("/predict", json=TRIP)
    assert response.status_code == 200
    assert response.json() == {
        "predicted_duration_minutes": 22.0,
        "model_name": "trip-duration",
        "model_version": "7",
    }


def test_predict_sends_only_booking_time_features(client, fake_model):
    client.post("/predict", json=TRIP)
    row = fake_model.seen.iloc[0]
    assert list(fake_model.seen.columns) == config.BOOKING_FEATURES
    assert row["pickup_hour"] == 17
    assert row["pickup_weekday"] == 3  # 23 January 2025 is a Thursday


def test_batch_keeps_request_order(client):
    times = ["2025-01-23T01:00:00", "2025-01-23T09:00:00", "2025-01-23T04:00:00"]
    trips = [{**TRIP, "pickup_datetime": t} for t in times]
    response = client.post("/predict_batch", json=trips)
    assert response.json()["predicted_duration_minutes"] == [6.0, 14.0, 9.0]


def test_empty_batch_returns_empty_list(client):
    response = client.post("/predict_batch", json=[])
    assert response.json()["predicted_duration_minutes"] == []


def test_batch_size_is_limited(client):
    response = client.post("/predict_batch", json=[TRIP] * (main.MAX_BATCH + 1))
    assert response.status_code == 422


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("PULocationID", 0),
        ("DOLocationID", 266),
        ("pickup_datetime", "not a date"),
    ],
)
def test_invalid_trips_are_rejected(client, field, value):
    response = client.post("/predict", json={**TRIP, field: value})
    assert response.status_code == 422


def test_missing_field_is_rejected(client):
    trip = {k: v for k, v in TRIP.items() if k != "pickup_datetime"}
    assert client.post("/predict", json=trip).status_code == 422


def test_a_driven_distance_is_refused_not_ignored(client):
    """Old clients that still send trip_distance learn it is no longer an input."""
    response = client.post("/predict", json={**TRIP, "trip_distance": 2.4})
    assert response.status_code == 422


def test_api_starts_without_mlflow_and_says_why(client_without_model):
    health = client_without_model.get("/health")
    assert health.status_code == 503
    assert "MLflow not reachable" in health.json()["detail"]
    assert client_without_model.post("/predict", json=TRIP).status_code == 503
