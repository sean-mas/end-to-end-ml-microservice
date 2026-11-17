"""API routes with a fake model, so these tests need no MLflow server."""

import pytest
from fastapi.testclient import TestClient

from app import main
from app.model import ModelService

TRIP = {
    "PULocationID": 161,
    "DOLocationID": 236,
    "trip_distance": 2.4,
    "pickup_datetime": "2025-01-23T17:30:00",
}


class FakeModel:
    """Predicts 10 minutes per mile, and records the feature table it was given."""

    def __init__(self):
        self.seen = None

    def predict(self, X):
        self.seen = X
        return X["trip_distance"].to_numpy() * 10


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
        "predicted_duration_minutes": 24.0,
        "model_name": "trip-duration",
        "model_version": "7",
    }


def test_predict_builds_the_training_features(client, fake_model):
    client.post("/predict", json=TRIP)
    row = fake_model.seen.iloc[0]
    assert list(fake_model.seen.columns) == [
        "PULocationID",
        "DOLocationID",
        "trip_distance",
        "pickup_hour",
        "pickup_weekday",
    ]
    assert row["pickup_hour"] == 17
    assert row["pickup_weekday"] == 3  # 23 January 2025 is a Thursday


def test_batch_keeps_request_order(client):
    trips = [{**TRIP, "trip_distance": d} for d in (1.0, 5.0, 2.0)]
    response = client.post("/predict_batch", json=trips)
    assert response.json()["predicted_duration_minutes"] == [10.0, 50.0, 20.0]


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
        ("trip_distance", 0.05),
        ("trip_distance", 150),
        ("pickup_datetime", "not a date"),
    ],
)
def test_invalid_trips_are_rejected(client, field, value):
    response = client.post("/predict", json={**TRIP, field: value})
    assert response.status_code == 422


def test_missing_field_is_rejected(client):
    trip = {k: v for k, v in TRIP.items() if k != "trip_distance"}
    assert client.post("/predict", json=trip).status_code == 422


def test_api_starts_without_mlflow_and_says_why(client_without_model):
    health = client_without_model.get("/health")
    assert health.status_code == 503
    assert "MLflow not reachable" in health.json()["detail"]
    assert client_without_model.post("/predict", json=TRIP).status_code == 503
