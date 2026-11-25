"""One command line plans, resumes, inspects and summarizes every experiment."""

import pytest
import runner.cli as cli
import runner.lifecycle as lifecycle
from runner.artifacts import read_json


@pytest.fixture
def launched(monkeypatch):
    batches = []

    async def launch(jobs, *, cpus, memory_gb, gpus, accelerator):
        assert cpus == 2
        batches.append(len(jobs))
        for job in jobs:
            manifest, cell_id = job.command[-2:]
            cli.main(package, ["run-cell", manifest, cell_id])
        return 0

    monkeypatch.setattr(cli, "run_jobs", launch)
    monkeypatch.setattr(
        lifecycle, "checkout", lambda module: {"commit": "a", "dirty": False}
    )
    return batches


package = None


def test_run_resume_status_and_summarize(tmp_path, toy_package, launched, capsys):
    global package
    package, study = toy_package
    root = ["--root", str(tmp_path / "results")]

    def main(*argv):
        return cli.main(package, list(argv))

    assert main("list") == 0
    assert capsys.readouterr().out.split() == ["26-11-17-study"]
    assert main("status", "26-11-17-study", "--profile", "smoke", *root) == 0
    assert "unplanned" in capsys.readouterr().out
    run = ("run", "26-11-17-study", "--profile", "smoke", "--cpus", "2", *root)
    assert main(*run) == 0
    summary = tmp_path / "results/26-11-17-study/smoke/summary.json"
    assert read_json(summary)["total"] == 3
    assert main(*run) == 0
    assert launched == [2, 0]
    assert main("status", "--profile", "smoke", *root) == 0
    assert '"completed": 2' in capsys.readouterr().out
    assert main("summarize", "26-11-17-study", "--profile", "smoke", *root) == 0
