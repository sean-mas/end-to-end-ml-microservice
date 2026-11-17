"""The command line runs the stages in the right order (stages faked, no MLflow)."""

from click.testing import CliRunner

from trip_duration import cli, promote, selection, train, tune


def test_all_runs_every_stage_in_order(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "_dataset", lambda: "data")
    monkeypatch.setattr(train, "train_baselines", lambda d: calls.append("train"))
    monkeypatch.setattr(
        tune, "tune", lambda d, n_trials: calls.append(f"tune:{n_trials}")
    )
    monkeypatch.setattr(selection, "select", lambda d: calls.append("select") or "r1")
    monkeypatch.setattr(
        promote, "promote", lambda r: calls.append(f"promote:{r}") or "3"
    )

    result = CliRunner().invoke(cli.cli, ["all", "--trials", "2"])

    assert result.exit_code == 0, result.output
    assert calls == ["train", "tune:2", "select", "promote:r1"]
    assert "Version 3 is now @production" in result.output


def test_every_stage_has_a_command():
    assert set(cli.cli.commands) == {
        "download",
        "train",
        "tune",
        "select",
        "promote",
        "all",
    }
