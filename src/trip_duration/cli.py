"""Command line: `uv run trip-duration <step>`, one command per pipeline step."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import click

from . import config, data

if TYPE_CHECKING:
    from .tracking import Dataset

# The selection chart shown in the README; `select` and `all` overwrite it.
SELECTION_CHART = config.PROJECT_ROOT / "docs" / "images" / "model_selection.png"


@click.group()
def cli() -> None:
    """From raw trip data to a model the API can serve."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


@cli.command()
@click.option("--month", default=config.MONTH, show_default=True)
@click.option("--force", is_flag=True, help="Download again even if the file exists.")
def download(month: str, force: bool) -> None:
    """Download one month of raw trips to data/raw/."""
    data.download(month, force=force)


def _dataset() -> "Dataset":
    """Load, clean and split January once per command."""
    from . import tracking

    return tracking.Dataset.from_trips(data.load_trips())


@cli.command()
def train() -> None:
    """Stage 1: log the four baseline runs to MLflow."""
    from .train import train_baselines

    train_baselines(_dataset())


@cli.command()
@click.option("--trials", default=20, show_default=True)
def tune(trials: int) -> None:
    """Stage 2: tune the Random Forest with Optuna, one child run per trial."""
    from .tune import tune as run_study

    run_study(_dataset(), n_trials=trials)


@cli.command()
@click.option(
    "--chart",
    type=click.Path(dir_okay=False, path_type=Path),
    default=SELECTION_CHART,
    show_default=True,
    help="Where to save the selection chart (it is also logged to MLflow).",
)
def select(chart: Path) -> None:
    """Stage 3: pick the winner by the rule in docs/decisions.md, save the candidate."""
    from .selection import select as pick

    run_id = pick(_dataset(), chart_path=chart)
    click.echo(f"Candidate run: {run_id}")


@cli.command()
@click.option(
    "--run-id", default=None, help="Run to promote. Default: newest candidate."
)
def promote(run_id: str | None) -> None:
    """Stage 4: register the candidate and point the production alias at it."""
    from .promote import promote as register

    version = register(run_id)
    click.echo(f"Version {version} is now @{config.setting('MODEL_ALIAS')}")


@cli.command(name="all")
@click.option("--trials", default=20, show_default=True)
def run_all(trials: int) -> None:
    """Every stage in order: download, train, tune, select, promote."""
    from .promote import promote as register
    from .selection import select as pick
    from .train import train_baselines
    from .tune import tune as run_study

    dataset = _dataset()  # load and split once, reuse for every stage
    train_baselines(dataset)
    run_study(dataset, n_trials=trials)
    run_id = pick(dataset, chart_path=SELECTION_CHART)
    version = register(run_id)
    click.echo(f"Version {version} is now @{config.setting('MODEL_ALIAS')}")


@cli.command()
def evaluate() -> None:
    """Stage 5: score the production model on the validation days in detail."""
    from .evaluate import evaluate as score
    from .evaluate import load_production

    model, version = load_production()
    report = score(model, data.load_trips())
    table = report.pop("by_trip_length")
    click.echo(f"Production model: version {version}")
    for key, value in report.items():
        click.echo(f"  {key:<24} {value:,.3f}")
    click.echo("\nError by actual trip length (minutes):")
    click.echo(table.round(2).to_string())
