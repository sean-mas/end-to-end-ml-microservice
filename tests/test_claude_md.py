"""CLAUDE.md stays true: it must name every CLI command and every published port.

When a command or port changes and CLAUDE.md doesn't, CI fails here, so the
instruction file can't silently drift from the code.
"""

import re

from trip_duration import cli, config

CLAUDE_MD = (config.PROJECT_ROOT / "CLAUDE.md").read_text()
COMPOSE = (config.PROJECT_ROOT / "docker-compose.yaml").read_text()


def test_every_cli_command_is_documented():
    missing = [
        name for name in cli.cli.commands if f"trip-duration {name}" not in CLAUDE_MD
    ]
    assert not missing, f"Add these commands to CLAUDE.md: {missing}"


def test_every_published_port_is_documented():
    ports = re.findall(r'"127\.0\.0\.1:(\d+):\d+"', COMPOSE)
    assert ports, "expected published ports in docker-compose.yaml"
    missing = [p for p in ports if f"127.0.0.1:{p}" not in CLAUDE_MD]
    assert not missing, f"Add these ports to CLAUDE.md: {missing}"


def test_default_mlflow_port_is_documented():
    port = config.MLFLOW_DEFAULTS["MLFLOW_TRACKING_URI"].rsplit(":", 1)[1]
    assert f"127.0.0.1:{port}" in CLAUDE_MD
