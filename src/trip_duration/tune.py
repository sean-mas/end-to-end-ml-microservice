"""Stage 2: tune the Random Forest with Optuna; every trial is a child run in MLflow."""

import logging

import mlflow
import optuna

from . import config, models, tracking

logger = logging.getLogger(__name__)


def suggest_params(trial: optuna.Trial) -> dict:
    """The search space. Leaves of at least 20 trips keep any model under ~2 GB."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 20, 100, step=10),
        "max_depth": trial.suggest_int("max_depth", 10, 30),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 20, 300, log=True),
        "max_features": trial.suggest_float("max_features", 0.4, 1.0),
    }


def tune(data: tracking.Dataset, n_trials: int) -> dict:
    """Run the study under one parent run and return the best trial's settings.

    Trials log scores only, not the model: 20 saved forests would fill gigabytes.
    The selection step retrains the winner and saves that one model.
    """
    tracking.connect()

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)
        name = f"trial-{trial.number:02d}"
        result = tracking.log_run(
            name,
            models.random_forest(config.FEATURES, **params),
            data,
            params={"model": "RandomForest", **params},
            tags={"stage": "tuning", "trial": str(trial.number)},
            save_model=False,
        )
        return result.metrics["val_rmse"]

    with mlflow.start_run(run_name="rf-optuna-study") as parent:
        mlflow.set_tags({"stage": "tuning-study", "sampler": "TPE"})
        mlflow.log_params({"n_trials": n_trials, "objective": "val_rmse"})
        sampler = optuna.samplers.TPESampler(seed=config.RANDOM_STATE)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=n_trials)
        mlflow.log_metric("best_val_rmse", study.best_value)
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        logger.info("Best trial %s: %s", study.best_trial.number, study.best_params)
        logger.info("Study run: %s", parent.info.run_id)
    return study.best_params
