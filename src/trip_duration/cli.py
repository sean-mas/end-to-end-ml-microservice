"""Command line: `uv run trip-duration <step>`, one command per pipeline step."""

import logging

import click

from . import config, data


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


def _dataset():
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
def select() -> None:
    """Stage 3: pick the winner by the rule in docs/decisions.md, save the candidate."""
    from .selection import select as pick

    run_id = pick(_dataset())
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
