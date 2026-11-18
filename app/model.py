"""Load the production model from the MLflow registry once, and predict with it."""

import logging
from dataclasses import dataclass

import mlflow
import pandas as pd

from trip_duration import config, features

from .schemas import TripRequest

logger = logging.getLogger(__name__)


@dataclass
class ModelService:
    """A loaded model plus the registry version it came from."""

    model: object
    name: str
    version: str

    def predict(self, trips: list[TripRequest]) -> list[float]:
        """Turn requests into the training feature table and predict minutes."""
        rows = pd.DataFrame(
            [
                {
                    "PULocationID": t.PULocationID,
                    "DOLocationID": t.DOLocationID,
                    config.PICKUP_TIME: t.pickup_datetime,
                }
                for t in trips
            ]
        )
        # Same function as training, so the API can't build features differently.
        predictions = self.model.predict(features.build_features(rows))
        return [round(float(p), 2) for p in predictions]


def load_service() -> ModelService:
    """Load `models:/<name>@<alias>` from MLflow. Raises if it isn't reachable."""
    mlflow.set_tracking_uri(config.setting("MLFLOW_TRACKING_URI"))
    name = config.setting("REGISTERED_MODEL_NAME")
    alias = config.setting("MODEL_ALIAS")
    version = mlflow.MlflowClient().get_model_version_by_alias(name, alias)
    # Load by the version number just resolved, so the model and the version the
    # API reports can never differ, even if the alias moves in between.
    model = mlflow.sklearn.load_model(f"models:/{name}/{version.version}")
    logger.info("Loaded %s version %s (@%s)", name, version.version, alias)
    return ModelService(model=model, name=name, version=str(version.version))
