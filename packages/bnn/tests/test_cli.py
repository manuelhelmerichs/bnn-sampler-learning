"""The command line runs, reports and resumes a run."""

from __future__ import annotations

import yaml
from bnn import cli

from .conftest import BASE, merge, regression_table


def write_config(tmp_path, **changes):
    document = merge(BASE, changes)
    document["data"]["path"] = str(regression_table(tmp_path / "table.npy"))
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False))
    return path


def test_run_status_and_resume_through_the_command_line(tmp_path, capsys):
    path = write_config(tmp_path)
    arguments = ["run", str(path), "--root", str(tmp_path / "runs"), "--allow-dirty"]
    assert cli.main([*arguments, "--quiet", "--stages", "fit", "--workers", "1"]) == 0
    assert cli.main(["status", str(tmp_path / "runs")]) == 0
    report = capsys.readouterr().out
    assert "fit       complete" in report and "sample    pending" in report
    assert cli.main([*arguments, "--quiet", "--workers", "1"]) == 0
    assert "complete" in capsys.readouterr().out
    # A complete run is left untouched.
    assert cli.main([*arguments, "--quiet", "--workers", "1"]) == 0
