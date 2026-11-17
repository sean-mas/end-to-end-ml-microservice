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
