"""Fit, score and log one model run to MLflow. Every comparison uses this file."""

import logging
import time
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")  # draw plots to files; no window needed
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import skops.io as sio
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.pipeline import Pipeline

from . import config, features, models

logger = logging.getLogger(__name__)

PLOT_SAMPLE = 5_000  # trips drawn in the predicted-vs-actual plot


@dataclass(frozen=True)
class Dataset:
    """Training and validation sets, built once and shared by every run."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series

    @classmethod
    def from_trips(cls, trips: pd.DataFrame) -> "Dataset":
        return cls(*features.prepare(trips))


def connect() -> None:
    """Point MLflow at the tracking server and the project's experiment."""
    mlflow.set_tracking_uri(config.setting("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment(config.setting("MLFLOW_EXPERIMENT"))


def fit_and_score(model: Pipeline, data: Dataset) -> dict[str, float]:
    """Fit on the training days and return the scores used to compare runs."""
    start = time.perf_counter()
    model.fit(data.X_train, data.y_train)
    fit_seconds = time.perf_counter() - start
    val_pred = model.predict(data.X_val)
    return {
        "val_rmse": root_mean_squared_error(data.y_val, val_pred),
        "val_mae": mean_absolute_error(data.y_val, val_pred),
        "train_rmse": root_mean_squared_error(
            data.y_train, model.predict(data.X_train)
        ),
        "fit_seconds": fit_seconds,
        "model_mb": models.size_mb(model),
    }


def prediction_plot(model: Pipeline, data: Dataset):
    """Scatter of predicted vs actual minutes on a sample of validation trips."""
    sample = data.X_val.sample(
        min(PLOT_SAMPLE, len(data.X_val)), random_state=config.RANDOM_STATE
    )
    actual = data.y_val.loc[sample.index]
    predicted = model.predict(sample)
    fig, ax = plt.subplots(figsize=(5.5, 5.5), dpi=110)
    ax.scatter(actual, predicted, s=4, alpha=0.25, color="#2a78d6", linewidths=0)
    ax.plot([0, 60], [0, 60], color="#eb6834", linewidth=1.5, label="perfect")
    ax.set(
        xlim=(0, 62),
        ylim=(0, 62),
        xlabel="actual minutes",
        ylabel="predicted minutes",
        title=f"Validation trips (sample of {len(sample):,})",
    )
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    return fig


def _own_types(model: Pipeline) -> list[str]:
    """Return the sklearn and project types inside a model, for skops to trust."""
    untrusted = sio.get_untrusted_types(data=sio.dumps(model))
    return [n for n in untrusted if n.startswith(("sklearn.", "trip_duration."))]


def log_run(
    name: str,
    model: Pipeline,
    data: Dataset,
    params: dict,
    tags: dict,
    save_model: bool = True,
) -> str:
    """Fit, score and log one run; return its run ID."""
    with mlflow.start_run(run_name=name, nested=mlflow.active_run() is not None):
        mlflow.set_tags(tags)
        mlflow.log_params(params)
        metrics = fit_and_score(model, data)
        mlflow.log_metrics(metrics)
        if save_model:
            fig = prediction_plot(model, data)
            mlflow.log_figure(fig, "predicted_vs_actual.png")
            plt.close(fig)
            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                input_example=data.X_val.head(3),
                # MLflow saves with skops, which refuses types it doesn't know.
                # Only scikit-learn's and this project's own types are trusted.
                skops_trusted_types=_own_types(model),
            )
        logger.info(
            "%-28s val_rmse=%.3f  train_rmse=%.3f  %4.0fs  %6.1f MB",
            name,
            metrics["val_rmse"],
            metrics["train_rmse"],
            metrics["fit_seconds"],
            metrics["model_mb"],
        )
        return mlflow.active_run().info.run_id
