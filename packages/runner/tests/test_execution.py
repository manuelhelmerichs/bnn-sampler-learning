from dataclasses import replace

import pytest
from runner.artifacts import read_json
from runner.experiment import Experiment
from runner.lifecycle import plan, run_cell, summarize
from runner.manifest import read_manifest


def run_all(path, experiment):
    return tuple(
        run_cell(path, cell.cell_id, experiment) for cell in read_manifest(path).cells
    )


def _plan(study, root, profile="smoke", experiment=None, **kwargs):
    return plan(
        experiment or study.EXPERIMENT,
        name="26-11-17-study",
        module=study.__name__,
        profile=profile,
        root=root,
        package="toy",
        **kwargs,
    )


def _summarize_part(manifest, results):
    stages = {cell.stage for cell in manifest.cells}
    assert len(stages) == 1
    return {
        "cell_metrics": [row["metrics"] for row in results],
        "total": sum(row["metrics"]["value"] for row in results),
        "derived": sorted(stages),
    }


def test_plan_run_resume_and_summarize(tmp_path, toy_package):
    _, study = toy_package
    path = _plan(study, tmp_path / "result")
    manifest = read_manifest(path)
    assert manifest.experiment == "26-11-17-study"
    for cell in manifest.cells:
        result = run_cell(path, cell.cell_id, study.EXPERIMENT)
        assert run_cell(path, cell.cell_id, study.EXPERIMENT) == result
    summary = read_json(summarize(path, study.EXPERIMENT))
    assert summary["total"] == 3
    assert summary["experiment"] == "26-11-17-study"
    assert summary["consumed_cell_ids"] == [cell.cell_id for cell in manifest.cells]
    assert run_all(path, study.EXPERIMENT) == tuple(
        tmp_path / "result/cells" / cell.cell_id / "result.json"
        for cell in manifest.cells
    )


def test_replanning_reuses_compatible_cells(tmp_path, toy_package, monkeypatch):
    _, study = toy_package
    path = _plan(study, tmp_path / "result")
    cell = read_manifest(path).cells[0]
    result = run_cell(path, cell.cell_id, study.EXPERIMENT)
    original = read_json(result)
    monkeypatch.setattr(
        study, "design", lambda: replace(study.EXPERIMENT.design(), question="Changed")
    )
    changed = replace(study.EXPERIMENT, design=study.design)
    second = _plan(study, tmp_path / "result", experiment=changed)
    assert read_manifest(second).question == "Changed"
    assert read_json(result) == original
    assert run_cell(second, cell.cell_id, changed) == result


def _at(monkeypatch, commit):
    import runner.lifecycle as lifecycle

    monkeypatch.setattr(
        lifecycle, "checkout", lambda module: {"commit": commit, "dirty": False}
    )


def test_resuming_after_a_code_change_keeps_each_cells_commit(
    tmp_path, toy_package, monkeypatch
):
    _, study = toy_package
    _at(monkeypatch, "a")
    path = _plan(study, tmp_path, profile="pilot", store=tmp_path)
    cell = read_manifest(path).cells[0]
    result = run_cell(path, cell.cell_id, study.EXPERIMENT)
    assert read_json(result.with_name("record.json"))["code"]["commit"] == "a"
    _at(monkeypatch, "b")
    _plan(study, tmp_path, profile="pilot", store=tmp_path, allow_code_change=True)
    assert run_cell(path, cell.cell_id, study.EXPERIMENT) == result
    run_all(path, study.EXPERIMENT)
    assert read_json(summarize(path, study.EXPERIMENT))["code"]["cells"] == ["a", "b"]


def test_joined_parts_keep_disjoint_stages_and_summaries(tmp_path, toy_package):
    _, study = toy_package

    def renamed(stage):
        def design():
            base = study.design()
            return replace(
                base,
                cells=tuple(replace(c, stage=stage, cell_id="") for c in base.cells),
            )

        return Experiment(design, study.measure, _summarize_part)

    joined = Experiment.join(
        "Do joined parts work?", {"left": renamed("left"), "right": renamed("right")}
    )
    path = _plan(study, tmp_path, experiment=joined)
    run_all(path, joined)
    summary = read_json(summarize(path, joined))
    assert summary["parts"]["left"]["total"] == 3
    assert summary["parts"]["right"]["total"] == 3
    cells = read_manifest(path).cells
    for name, part in summary["parts"].items():
        assert part["derived"] == [name]
        assert "cell_metrics" not in part
        assert part["consumed_cell_ids"] == [
            cell.cell_id for cell in cells if cell.stage == name
        ]
    assert len(summary["cell_metrics"]) == len(cells)
    manifest = replace(read_manifest(path), profile="production")
    results = [{"metrics": {"value": 1}} for _ in manifest.cells]
    assert joined.reduce(manifest, results)["publishable"]
    incomplete = renamed("right")
    incomplete = replace(
        incomplete,
        summarize=lambda m, r: (
            study.summarize(m, r) | {"accepted": False, "publishable": False}
        ),
    )
    blocked = Experiment.join(
        "Do joined parts work?", {"left": renamed("left"), "right": incomplete}
    )
    rejected = blocked.reduce(manifest, results)
    assert not rejected["accepted"] and not rejected["publishable"]


def test_summarize_refuses_metrics_that_contradict_factors(tmp_path, toy_package):
    _, study = toy_package
    shifted = replace(
        study.EXPERIMENT,
        measure=lambda cell: {"metrics": {"value": cell.factors["value"] + 1}},
    )
    path = _plan(study, tmp_path, experiment=shifted)
    run_all(path, shifted)
    with pytest.raises(RuntimeError, match=r"other values: \['value'\]"):
        summarize(path, shifted)
    assert not (tmp_path / "summary.json").exists()
