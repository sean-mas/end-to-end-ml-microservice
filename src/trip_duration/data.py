"""Download the raw TLC trip file once and load it."""

import logging
from pathlib import Path

import pandas as pd
import requests

from . import config

logger = logging.getLogger(__name__)


def download(month: str = config.MONTH, force: bool = False) -> Path:
    """Download one month of trips to data/raw/ and return the file path.

    A file that is already there is reused, so running this twice costs nothing.
    """
    path = config.raw_path(month)
    if path.exists() and not force:
        logger.info("Reusing %s", path.name)
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    url = config.source_url(month)
    logger.info("Downloading %s", url)
    # Write to a temporary name first so a broken download never looks complete.
    partial = path.with_suffix(".part")
    with requests.get(url, stream=True, timeout=config.DOWNLOAD_TIMEOUT_SECONDS) as r:
        r.raise_for_status()
        with partial.open("wb") as f:
            for chunk in r.iter_content(chunk_size=config.DOWNLOAD_CHUNK_BYTES):
                f.write(chunk)
    partial.rename(path)
    logger.info("Saved %s (%.0f MB)", path.name, path.stat().st_size / 1e6)
    return path


def load_trips(month: str = config.MONTH) -> pd.DataFrame:
    """Return the raw trips for one month, downloading the file if needed."""
    return pd.read_parquet(download(month))
