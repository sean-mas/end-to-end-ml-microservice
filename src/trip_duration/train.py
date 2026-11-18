"""Stage 1: four baseline runs, each adding one idea, logged side by side in MLflow."""

from sklearn.pipeline import Pipeline

from . import config, models, tracking


def baseline_runs() -> list[tuple[str, Pipeline, dict, dict]]:
    """Return (run name, model, params, tags) for each baseline, simplest first."""
    return [
        (
            "mean-baseline",
            models.mean_baseline(),
            {"model": "DummyRegressor", "features": "none"},
            {"stage": "baseline", "idea": "always guess the average"},
        ),
        (
            "linear-route-onehot",
            models.linear_route(),
            {
                "model": "LinearRegression",
                "features": "route one-hot + typical distance",
            },
            {"stage": "baseline", "idea": "course reference model"},
        ),
        (
            "rf-zones-distance",
            models.random_forest(config.BASE_FEATURES),
            {
                "model": "RandomForest",
                "features": "zones + typical distance",
                **models.RF_START,
            },
            {"stage": "baseline", "idea": "Random Forest, no time features"},
        ),
        (
            "rf-zones-distance-time",
            models.random_forest(config.FEATURES),
            {
                "model": "RandomForest",
                "features": "zones + typical distance + hour + weekday",
                **models.RF_START,
            },
            {"stage": "baseline", "idea": "add pickup hour and weekday"},
        ),
    ]


def train_baselines(data: tracking.Dataset) -> dict[str, str]:
    """Log every baseline run and return {run name: run ID}."""
    tracking.connect()
    return {
        name: tracking.log_run(name, model, data, params, tags).run_id
        for name, model, params, tags in baseline_runs()
    }
