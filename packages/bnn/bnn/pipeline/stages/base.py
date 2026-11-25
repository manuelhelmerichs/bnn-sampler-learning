"""What every stage of a run shares, and the tasks a stage consists of.

A stage splits into tasks that are independent of each other, usually one per
chain group. A task commits its progress as it goes and can be interrupted and
continued in any process. The parts of a stage that need all chains run before
(``prepare``) or after (``finalize``) its tasks, in the coordinating process.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from bnn.engine import Destinations, Pacing
from bnn.pipeline.context import Context
from bnn.pipeline.run import Run
from bnn.store.io import write_json
from bnn.trees import tree_all_finite


@dataclass(frozen=True)
class Task:
    """An independently executable part of a stage."""

    stage: str
    id: str
    chains: tuple[int, ...]


@dataclass
class Controls:
    """How a running task is asked to pause and how it reports progress."""

    should_stop: Callable[[], bool] = lambda: False
    report: Callable[[dict], None] = lambda event: None


class Stage:
    """Base of all stages."""

    name: str

    def enabled(self, run: Run) -> bool:
        """Whether the configuration requests this stage."""
        return run.config.requests(self.name)

    def tasks(self, run: Run) -> list[Task]:
        """The tasks of this stage, one per chain group by default."""
        return [
            Task(self.name, f"group_{index:04d}", tuple(group))
            for index, group in enumerate(run.groups)
        ]

    def prepare(self, run: Run, context: Context) -> None:
        """Work required before any task, in the coordinating process."""

    def run_task(
        self, run: Run, context: Context, task: Task, controls: Controls
    ) -> bool:
        """Run or continue a task; return whether it completed."""
        raise NotImplementedError

    def finalize(self, run: Run, context: Context) -> None:
        """Work that needs all completed tasks, in the coordinating process."""

    # --- task bookkeeping ---------------------------------------------------

    def is_done(self, run: Run, task: Task) -> bool:
        """Whether a task has completed."""
        return (run.directory.work(task.stage, task.id) / "done.json").exists()

    def mark_done(self, run: Run, task: Task) -> None:
        """Record completion and drop the checkpoints of a task."""
        work = run.directory.work(task.stage, task.id)
        write_json(work / "done.json", {"chains": list(task.chains)})
        shutil.rmtree(work / "state", ignore_errors=True)

    def _pacing(self, run: Run) -> Pacing:
        execution = run.config.execution
        return Pacing(
            interval=execution.checkpoint_interval,
            buffer_bytes=execution.buffer_bytes,
            block_steps=execution.block_steps,
        )

    def _destinations(self, run: Run, task: Task, stores: dict) -> Destinations:
        execution = run.config.execution
        return Destinations(
            state=run.directory.work(task.stage, task.id) / "state",
            stores=stores,
            diagnostics=run.directory.diagnostics(task.stage)
            if execution.diagnostics
            else None,
            chunk_bytes=execution.chunk_bytes,
            shard_bytes=execution.shard_bytes,
        )


def require_finite(tree: Any, chains, what: str) -> None:
    """Raise if a tree with a leading chain axis holds a non-finite value."""
    finite = np.asarray(tree_all_finite(tree, batch_dims=1))
    if not finite.all():
        raise FloatingPointError(
            f"{what} of chains {np.asarray(chains)[~finite].tolist()} is not finite."
        )
