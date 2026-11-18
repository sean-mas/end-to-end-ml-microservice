"""Stage 3: pick the winner by a rule set before tuning ran, save it as a candidate.

The rule (docs/decisions.md): lowest validation RMSE wins. Runs within
TOLERANCE_MINUTES of the best count as a tie, and the smallest model among
them wins, because a smaller model loads and answers faster in the API.
"""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw plots to files; no window needed
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from matplotlib.figure import Figure

from . import config, models, tracking

logger = logging.getLogger(__name__)

TOLERANCE_MINUTES = 0.05
COMPARED_STAGES = ("baseline", "tuning")
TUNED_PARAMS = ["n_estimators", "max_depth", "min_samples_leaf", "max_features"]


def tied_runs(runs: pd.DataFrame, tolerance: float = TOLERANCE_MINUTES) -> pd.DataFrame:
    """Return the runs within `tolerance` minutes of the lowest validation RMSE."""
    best = runs["val_rmse"].min()
    # The small extra margin stops float rounding (4.10 + 0.05 = 4.1499...) from
    # excluding a run that sits exactly on the edge.
    return runs[runs["val_rmse"] <= best + tolerance + 1e-9]


def apply_rule(runs: pd.DataFrame, tolerance: float = TOLERANCE_MINUTES) -> pd.Series:
    """Return the winning row. Needs columns val_rmse and model_mb."""
    return tied_runs(runs, tolerance).sort_values(["model_mb", "val_rmse"]).iloc[0]


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


def selection_plot(runs: pd.DataFrame, winner: pd.Series) -> Figure:
    """Validation RMSE against model size, with the tie band and the winner marked.

    The mean baseline (9.6 min) is left off so the axis can show the close race.
    """
    shown = runs[runs["val_rmse"] < 6]
    best = shown["val_rmse"].min()
    fig, ax = plt.subplots(figsize=(9, 5), dpi=120)
    ax.axhspan(best, best + TOLERANCE_MINUTES, color="#eda100", alpha=0.18, linewidth=0)
    for stage, color, label in [
        ("tuning", "#2a78d6", "Optuna trial"),
        ("baseline", "#52514e", "baseline"),
    ]:
        part = shown[shown["tags.stage"] == stage]
        ax.scatter(part["model_mb"], part["val_rmse"], s=36, color=color, label=label)
    ax.scatter(
        winner["model_mb"],
        winner["val_rmse"],
        s=140,
        facecolors="none",
        edgecolors="#eb6834",
        linewidths=2.2,
        label="selected",
    )
    for _, run in shown.iterrows():
        if run["tags.stage"] == "baseline" or run["val_rmse"] == best:
            ax.annotate(
                run["run_name"],
                (run["model_mb"], run["val_rmse"]),
                xytext=(6, 4),
                textcoords="offset points",
                fontsize=8,
                color="#52514e",
            )
    ax.annotate(
        f"{winner['run_name']}: {winner['val_rmse']:.3f} min, {winner['model_mb']:.0f} MB",
        (winner["model_mb"], winner["val_rmse"]),
        xytext=(10, -16),
        textcoords="offset points",
        fontsize=9,
        color="#eb6834",
        fontweight="bold",
    )
    ax.text(
        shown["model_mb"].min(),
        best + TOLERANCE_MINUTES,
        f"  tie band: within {TOLERANCE_MINUTES} min of the best",
        va="bottom",
        fontsize=8,
        color="#8a6400",
    )
    ax.set_xscale("log")
    ax.set(
        xlabel="model size (MB, log scale)",
        ylabel="validation RMSE (minutes)",
        title="Lowest error wins; inside the tie band the smallest model wins",
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#e4e3df", linewidth=0.6)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    return fig


def select(data: tracking.Dataset, chart_path: Path | None = None) -> str:
    """Apply the rule, tag the winner, save it as a candidate run; return its ID.

    The selection chart is always logged to MLflow, and also saved to
    `chart_path` when one is given (the CLI passes the README's image).
    """
    runs = compared_runs()
    winner = apply_rule(runs)
    best = runs["val_rmse"].min()
    logger.info(
        "Winner: %s (val_rmse=%.3f, %.0f MB). Best RMSE overall: %.3f. %d runs tied.",
        winner["run_name"],
        winner["val_rmse"],
        winner["model_mb"],
        best,
        len(tied_runs(runs)),
    )
    client = mlflow.MlflowClient()
    client.set_tag(winner["run_id"], "selected", "true")
    fig = selection_plot(runs, winner)
    if chart_path is not None:
        fig.savefig(chart_path)

    if winner["tags.stage"] == "baseline":
        # Baseline runs already saved their model; that run is the candidate.
        client.set_tag(winner["run_id"], "stage", "candidate")
        client.log_figure(winner["run_id"], fig, "model_selection.png")
        plt.close(fig)
        return winner["run_id"]

    # Tuning trials saved no model: retrain the winning settings and save it.
    params = _params_of(winner)
    run_id = tracking.log_run(
        "selected-candidate",
        models.random_forest(config.FEATURES, **params),
        data,
        params={"model": "RandomForest", "source_run": winner["run_id"], **params},
        tags={
            "stage": "candidate",
            "selected_from": winner["run_name"],
            "rule": f"lowest val_rmse; ties within {TOLERANCE_MINUTES} min -> smallest",
        },
    ).run_id
    client.log_figure(run_id, fig, "model_selection.png")
    plt.close(fig)
    return run_id
