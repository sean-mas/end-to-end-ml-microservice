"""Stage 3: pick the winner by a rule fixed before tuning, and save it as a candidate.

The rule (docs/decisions.md): lowest validation RMSE wins. Runs within
TOLERANCE_MINUTES of the best count as a tie, and the smallest model among
them wins, because a smaller model loads and answers faster in the API.
"""

import logging

import mlflow
import pandas as pd

from . import config, models, tracking

logger = logging.getLogger(__name__)

TOLERANCE_MINUTES = 0.05
COMPARED_STAGES = ("baseline", "tuning")
TUNED_PARAMS = ["n_estimators", "max_depth", "min_samples_leaf", "max_features"]


def apply_rule(runs: pd.DataFrame, tolerance: float = TOLERANCE_MINUTES) -> pd.Series:
    """Return the winning row. Needs columns val_rmse and model_mb."""
    best = runs["val_rmse"].min()
    # The small extra margin stops float rounding (4.10 + 0.05 = 4.1499...) from
    # excluding a run that sits exactly on the edge.
    tied = runs[runs["val_rmse"] <= best + tolerance + 1e-9]
    return tied.sort_values(["model_mb", "val_rmse"]).iloc[0]


def compared_runs() -> pd.DataFrame:
    """All baseline and tuning runs of the experiment, one row per run."""
    tracking.connect()
    runs = mlflow.search_runs(
        experiment_names=[config.setting("MLFLOW_EXPERIMENT")],
        filter_string="attributes.status = 'FINISHED'",
    )
    runs = runs[runs["tags.stage"].isin(COMPARED_STAGES)].rename(
        columns={
            "metrics.val_rmse": "val_rmse",
            "metrics.model_mb": "model_mb",
            "tags.mlflow.runName": "run_name",
        }
    )
    return runs.dropna(subset=["val_rmse", "model_mb"])


def _params_of(row: pd.Series) -> dict:
    """Read the forest settings back from MLflow's string params."""
    raw = {p: row[f"params.{p}"] for p in TUNED_PARAMS if pd.notna(row[f"params.{p}"])}
    return {k: float(v) if k == "max_features" else int(v) for k, v in raw.items()}


def select(data: tracking.Dataset) -> str:
    """Apply the rule, tag the winner, save it as a candidate run; return its ID."""
    runs = compared_runs()
    winner = apply_rule(runs)
    best = runs["val_rmse"].min()
    logger.info(
        "Winner: %s (val_rmse=%.3f, %.0f MB). Best RMSE overall: %.3f. %d runs tied.",
        winner["run_name"],
        winner["val_rmse"],
        winner["model_mb"],
        best,
        (runs["val_rmse"] <= best + TOLERANCE_MINUTES).sum(),
    )
    client = mlflow.MlflowClient()
    client.set_tag(winner["run_id"], "selected", "true")

    if winner["tags.stage"] == "baseline":
        # Baseline runs already saved their model; that run is the candidate.
        client.set_tag(winner["run_id"], "stage", "candidate")
        return winner["run_id"]

    # Tuning trials saved no model: retrain the winning settings and save it.
    params = _params_of(winner)
    return tracking.log_run(
        "selected-candidate",
        models.random_forest(config.FEATURES, **params),
        data,
        params={"model": "RandomForest", "source_run": winner["run_id"], **params},
        tags={
            "stage": "candidate",
            "selected_from": winner["run_name"],
            "rule": f"lowest val_rmse; ties within {TOLERANCE_MINUTES} min -> smallest",
        },
    )
