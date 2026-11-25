"""Bring a run to completion, stage by stage.

``advance`` is idempotent. It creates the run if needed, skips what is complete
and continues what is not, so starting, resuming after an interruption and
adding a stage later are the same call.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from bnn.config import STAGES, RunConfig
from bnn.pipeline import provenance
from bnn.pipeline.run import Run, RunError, create_or_resume
from bnn.pipeline.stages import build_stages
from bnn.pipeline.workers import Resources, Stop, execute, failure_message


def configure_jax(config: RunConfig, cache: str | Path | None = None) -> None:
    """Set the precision of a run and enable the persistent compilation cache.

    Worker processes inherit the cache location through the environment, so
    every chain group after the first skips most of its compilation.
    """
    import jax

    jax.config.update("jax_enable_x64", config.x64)
    if cache is not None:
        os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(Path(cache).resolve()))
    location = os.environ.get("JAX_COMPILATION_CACHE_DIR")
    if location:
        try:
            Path(location).mkdir(parents=True, exist_ok=True)
        except OSError:
            return
        jax.config.update("jax_compilation_cache_dir", location)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.2)


def default_group_size(config: RunConfig, resources: Resources) -> int:
    """Chains per compiled program: one on CPU, all per accelerator otherwise."""
    if config.execution.group_size is not None:
        return min(config.execution.group_size, config.chains)
    import jax

    if jax.default_backend() == "cpu":
        return 1
    return -(-config.chains // max(1, len(resources.gpus)))


@dataclass(frozen=True)
class Outcome:
    """How far an invocation brought a run."""

    run: Run
    state: str  # complete | paused | waiting | failed
    stage: str | None = None
    detail: str = ""

    @property
    def ok(self) -> bool:
        """Whether nothing failed."""
        return self.state != "failed"


def advance(
    config: RunConfig,
    root: str | Path = "runs",
    *,
    stages: tuple[str, ...] | None = None,
    resources: Resources | None = None,
    shard: tuple[int, int] | None = None,
    allow_dirty: bool = False,
    allow_code_change: bool = False,
    stop: Stop | None = None,
    on_progress: Callable[[Run, str], None] = lambda run, stage: None,
) -> Outcome:
    """Create or continue the run of ``config`` below ``root``.

    ``stages`` limits the stages to work on. ``shard=(i, n)`` restricts this
    invocation to every ``n``-th task starting at ``i``, so several machines
    can share a run; a stage is finalized by whichever invocation finds all of
    its tasks complete.
    """
    resources = resources or Resources()
    stop = stop or Stop()
    code = provenance.code_identity(config.imports)
    if not allow_dirty:
        provenance.require_clean(code)
    configure_jax(config, Path(root) / ".cache" / "jax")
    run = create_or_resume(
        config,
        root,
        code,
        group_size=default_group_size(config, resources),
        allow_code_change=allow_code_change,
    )
    contexts: list = []

    def context():
        if not contexts:
            contexts.append(run.context())
        return contexts[0]

    available = build_stages()
    selected = STAGES if stages is None else stages
    for name in STAGES:
        stage = available[name]
        if name not in selected or not stage.enabled(run):
            continue
        if run.stage_status(name) == "complete":
            continue
        for required in STAGES[: STAGES.index(name)]:
            if available[required].enabled(run) and run.stage_status(required) != (
                "complete"
            ):
                return Outcome(
                    run, "waiting", name, f"stage {required!r} is not complete"
                )
        run.set_stage(name, "running")
        stage.prepare(run, context())
        tasks = stage.tasks(run)
        mine = [
            task
            for index, task in enumerate(tasks)
            if not stage.is_done(run, task)
            and (shard is None or index % shard[1] == shard[0])
        ]
        states = execute(
            run,
            stage,
            mine,
            resources=resources,
            context_factory=context,
            stop=stop,
            on_progress=lambda name=name: on_progress(run, name),
        )
        failed = [task for task in mine if states.get(task.id) == "failed"]
        if failed:
            return Outcome(
                run,
                "failed",
                name,
                "\n".join(failure_message(run, task) for task in failed),
            )
        if not all(stage.is_done(run, task) for task in tasks):
            if stop():
                return Outcome(
                    run, "paused", name, "interrupted; run again to continue"
                )
            return Outcome(
                run, "waiting", name, "other tasks of this stage are still open"
            )
        if stop():
            return Outcome(run, "paused", name, "interrupted; run again to continue")
        stage.finalize(run, context())
        run.set_stage(name, "complete")
        on_progress(run, name)
        # Checkpoints and progress records have served their purpose.
        shutil.rmtree(run.directory.root / "work" / name, ignore_errors=True)
    work = run.directory.root / "work"
    if work.exists() and not any(work.iterdir()):
        work.rmdir()
    return Outcome(run, "complete")


__all__ = ["Outcome", "RunError", "advance", "configure_jax"]
