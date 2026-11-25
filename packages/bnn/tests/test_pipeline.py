"""A run completes, pauses, resumes and parallelizes without changing its results."""

from __future__ import annotations

import jax
import numpy as np
import pytest
from bnn.pipeline.workers import Resources, Stop


def leaves_equal(left, right) -> bool:
    return all(
        np.array_equal(a, b)
        for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True)
    )


class StopAfter(Stop):
    """Request a pause once the engine has asked ``count`` times."""

    def __init__(self, count: int):
        super().__init__()
        self.count = count

    def __call__(self) -> bool:
        self.count -= 1
        if self.count <= 0:
            self.requested = 1
        return self.requested > 0


def test_interrupted_run_continues_bit_for_bit(make_config, run_config):
    config = make_config(
        fit={"name": "mode", "steps": 40, "optimizer": {"name": "adam"}},
        sampler={"warmup": 60, "steps": 80, "thinning": 3},
    )
    reference = run_config(config, "reference")
    assert reference.state == "complete"

    interruptions = 0
    while True:
        outcome = run_config(config, "interrupted", stop=StopAfter(2))
        if outcome.state == "complete":
            break
        assert outcome.state == "paused"
        interruptions += 1
    assert interruptions > 5

    a, b = reference.run, outcome.run
    assert leaves_equal(a.fitted(), b.fitted())
    assert leaves_equal(a.samples().load(), b.samples().load())
    assert leaves_equal(a.diagnostics(), b.diagnostics())
    assert leaves_equal(a.parameters(1), b.parameters(1))
    assert list(a.samples().draw_ids) == list(range(0, 80, 3))
    assert not (b.directory.root / "work").exists()


def test_block_length_and_worker_processes_do_not_change_draws(make_config, run_config):
    config = make_config()
    inline = run_config(config, "inline")
    other_blocks = run_config(make_config(execution={"block_steps": 7}), "blocks")
    processes = run_config(config, "processes", resources=Resources(workers=2))
    assert processes.state == "complete"
    expected = inline.run.samples().load()
    assert leaves_equal(expected, other_blocks.run.samples().load())
    assert leaves_equal(expected, processes.run.samples().load())


def test_shards_share_a_run_and_any_of_them_finalizes(make_config, run_config):
    config = make_config(chains=3)
    assert run_config(config, shard=(0, 2), stages=("fit",)).state == "waiting"
    assert run_config(config, shard=(1, 2), stages=("fit",)).state == "complete"
    assert run_config(config, shard=(1, 2)).state == "waiting"
    outcome = run_config(config, shard=(0, 2))
    assert outcome.state == "complete"
    assert outcome.run.samples().chain_ids == (0, 1, 2)


def test_mode_fit_resumes_bit_for_bit(make_config, run_config):
    # Sampling from a fit, and resuming it, is covered by the first test above.
    fit = {"name": "mode", "steps": 30, "optimizer": {"name": "sgd"}}
    config = make_config(fit=fit, sampler=None, model={"hidden_structure": [12, 1]})
    reference = run_config(config, "reference")
    while (outcome := run_config(config, "paused", stop=StopAfter(2))).state != (
        "complete"
    ):
        pass
    starts = reference.run.fitted()
    assert leaves_equal(starts, outcome.run.fitted())
    assert np.isfinite(starts["sigma"]).all()


def test_a_sampler_continues_a_run_that_ended_with_its_fit(
    make_config, run_config, tmp_path
):
    from bnn import run_of
    from bnn.config import dump_config
    from bnn.pipeline.run import RunError

    fit = {"name": "mode", "steps": 30, "optimizer": {"name": "adam"}}
    first = run_config(make_config(fit=fit, sampler=None))
    positions = first.run.fitted()
    fit_only, sampled = tmp_path / "fit.yaml", tmp_path / "sampled.yaml"
    fit_only.write_text(dump_config(make_config(fit=fit, sampler=None)))
    sampled.write_text(dump_config(make_config(fit=fit)))
    assert run_of(fit_only, tmp_path / "runs").directory == first.run.directory
    with pytest.raises(RunError, match="Unfinished stages .*: sample"):
        run_of(sampled, tmp_path / "runs")

    continued = run_config(make_config(fit=fit))
    assert continued.state == "complete"
    assert leaves_equal(positions, continued.run.fitted())
    direct = run_config(make_config(fit=fit), "direct")
    assert leaves_equal(direct.run.samples().load(), continued.run.samples().load())
    # Both configurations describe the sampled run.
    assert run_of(sampled, tmp_path / "runs").directory == first.run.directory
    assert run_of(fit_only, tmp_path / "runs").directory == first.run.directory
