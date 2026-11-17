"""Download step: reuses an existing file and never leaves a half-written one."""

import pytest
import requests

from trip_duration import config, data


@pytest.fixture
def raw_dir(tmp_path, monkeypatch):
    """Point data/raw/ at a temporary folder so tests never touch real data."""
    monkeypatch.setattr(config, "RAW_DIR", tmp_path)
    return tmp_path


class FakeResponse:
    def __init__(self, chunks, status=200):
        self.chunks, self.status = chunks, status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        if self.status >= 400:
            raise requests.HTTPError(f"status {self.status}")

    def iter_content(self, chunk_size):
        yield from self.chunks


def test_download_writes_file(raw_dir, monkeypatch):
    monkeypatch.setattr(
        data.requests, "get", lambda *a, **k: FakeResponse([b"ab", b"c"])
    )
    path = data.download("2025-01")
    assert path == raw_dir / "yellow_tripdata_2025-01.parquet"
    assert path.read_bytes() == b"abc"


def test_download_reuses_existing_file(raw_dir, monkeypatch):
    existing = raw_dir / "yellow_tripdata_2025-01.parquet"
    existing.write_bytes(b"old")

    def fail(*a, **k):
        raise AssertionError("should not download again")

    monkeypatch.setattr(data.requests, "get", fail)
    assert data.download("2025-01").read_bytes() == b"old"


def test_failed_download_leaves_no_file(raw_dir, monkeypatch):
    monkeypatch.setattr(data.requests, "get", lambda *a, **k: FakeResponse([], 404))
    with pytest.raises(requests.HTTPError):
        data.download("2025-01")
    assert not (raw_dir / "yellow_tripdata_2025-01.parquet").exists()
