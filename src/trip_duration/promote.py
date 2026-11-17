"""Stage 4: register the candidate model and point the `production` alias at it.

The API loads `models:/<name>@production`, so promoting a new version needs no
code change and no redeploy of the API image, only a restart of the API.
"""

import logging

import mlflow

from . import config, tracking

logger = logging.getLogger(__name__)


def latest_candidate() -> str:
    """Return the run ID of the newest run tagged stage=candidate."""
    tracking.connect()
    runs = mlflow.search_runs(
        experiment_names=[config.setting("MLFLOW_EXPERIMENT")],
        filter_string="tags.stage = 'candidate' and attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if runs.empty:
        raise LookupError("No candidate run. Run `trip-duration select` first.")
    return runs.iloc[0]["run_id"]


def promote(run_id: str | None = None) -> str:
    """Register the run's model as a new version and set the alias; return it."""
    tracking.connect()
    run_id = run_id or latest_candidate()
    name = config.setting("REGISTERED_MODEL_NAME")
    alias = config.setting("MODEL_ALIAS")
    run = mlflow.get_run(run_id)

    version = mlflow.register_model(f"runs:/{run_id}/model", name)
    client = mlflow.MlflowClient()
    client.set_registered_model_alias(name, alias, version.version)
    metrics = run.data.metrics
    client.update_model_version(
        name,
        version.version,
        description=(
            f"Random Forest from run {run.info.run_name} "
            f"(selected from {run.data.tags.get('selected_from', 'n/a')}). "
            f"Validation RMSE {metrics['val_rmse']:.3f} min, "
            f"MAE {metrics['val_mae']:.3f} min, {metrics['model_mb']:.0f} MB."
        ),
    )
    for key in ("val_rmse", "val_mae", "model_mb"):
        client.set_model_version_tag(name, version.version, key, f"{metrics[key]:.3f}")
    logger.info("%s version %s is now @%s", name, version.version, alias)
    return version.version
