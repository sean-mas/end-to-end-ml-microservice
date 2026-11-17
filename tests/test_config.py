"""Settings resolve from the environment first, then fall back to defaults."""

from trip_duration import config


def test_setting_uses_default_without_env(monkeypatch):
    monkeypatch.delenv("MODEL_ALIAS", raising=False)
    assert config.setting("MODEL_ALIAS") == "production"


def test_setting_prefers_environment(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    assert config.setting("MLFLOW_TRACKING_URI") == "http://mlflow:5000"


def test_source_url_points_to_yellow_taxi_month():
    assert config.source_url("2025-01").endswith("/yellow_tripdata_2025-01.parquet")


def test_split_dates_are_in_order():
    assert config.PERIOD_START < config.VALIDATION_START < config.PERIOD_END
