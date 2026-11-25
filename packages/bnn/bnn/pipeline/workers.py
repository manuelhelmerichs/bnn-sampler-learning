"""Execute the tasks of a stage, inline or across worker processes.

Chains are independent, so a stage scales by running its chain groups in
separate processes. Every worker is a fresh interpreter that owns one group,
with its own threads or its own accelerator, and communicates only through the
run directory: progress files while it works, a completion marker at the end.
That makes the degree of parallelism a free choice of every invocation; it
never affects results.
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from bnn.pipeline.run import Run
from bnn.pipeline.stages import Controls, Stage, Task
from bnn.store.io import lock, read_json, write_json

PAUSED = 75  # EX_TEMPFAIL: the task was interrupted and can be continued.
BUSY = 76  # Another process is running the task.


class Stop:
    """A request to pause, set by a signal or by the caller."""

    def __init__(self) -> None:
        """Start unset."""
        self.requested = 0

    def request(self) -> None:
        """Ask running tasks to commit and stop."""
        self.requested += 1

    def __call__(self) -> bool:
        """Whether a pause was requested."""
        return self.requested > 0


class ProgressWriter:
    """Publish the progress of a task for the coordinator and ``bnn status``."""

    def __init__(self, path: Path, task: Task, minimum_interval: float = 0.5):
        """Write the initial record."""
        self.path = path
        self.minimum_interval = minimum_interval
        self.record = {
            "stage": task.stage,
            "task": task.id,
            "chains": list(task.chains),
            "state": "starting",
            "host": socket.gethostname(),
            "pid": os.getpid(),
        }
        self._written = 0.0
        self._rate = None
        self._phase = None
        self.write(force=True)

    def write(self, *, force: bool = False) -> None:
        """Persist the record, at most once per interval unless forced."""
        now = time.time()
        if force or now - self._written >= self.minimum_interval:
            self.record["updated"] = now
            write_json(self.path, self.record)
            self._written = now

    def report(self, event: dict) -> None:
        """Take a progress event of the engine."""
        if event["phase_index"] != self._phase:
            # The first block of a phase includes its compilation.
            self._phase, self._rate = event["phase_index"], None
        else:
            rate = event["block"] / max(event["seconds"], 1e-9)
            self._rate = rate if self._rate is None else 0.7 * self._rate + 0.3 * rate
        self.record.update(
            state="running",
            phase=event["phase"],
            phase_index=event["phase_index"],
            phases=event["phases"],
            step=event["step"],
            steps=event["steps"],
            rate=self._rate,
        )
        self.write()

    def finish(self, state: str, message: str | None = None) -> None:
        """Record how the task ended."""
        self.record["state"] = state
        if message:
            self.record["message"] = message
        self.write(force=True)


def run_task(
    run: Run, stage: Stage, task: Task, context, should_stop: Callable[[], bool]
) -> str:
    """Run one task in this process; return ``done``, ``paused`` or ``busy``."""
    work = run.directory.work(task.stage, task.id)
    work.mkdir(parents=True, exist_ok=True)
    if stage.is_done(run, task):
        return "done"
    try:
        with lock(work / ".task.lock", blocking=False):
            progress = ProgressWriter(work / "progress.json", task)
            try:
                completed = stage.run_task(
                    run, context, task, Controls(should_stop, progress.report)
                )
            except BaseException as error:
                progress.finish("failed", f"{type(error).__name__}: {error}")
                raise
            progress.finish("done" if completed else "paused")
            return "done" if completed else "paused"
    except BlockingIOError:
        return "busy"


@dataclass(frozen=True)
class Resources:
    """How many processes run tasks and what each of them may use.

    Without a number of workers there is one per task, limited by the cores or
    the given accelerators. A single worker runs the tasks in this process.
    """

    workers: int | None = None
    threads: int | None = None
    gpus: tuple[str, ...] = ()

    def resolved(self, tasks: int) -> Resources:
        """Fix the number of workers: one per task, at most one per core."""
        if self.workers is not None:
            return self
        limit = len(self.gpus) or os.cpu_count() or 1
        return Resources(max(1, min(tasks, limit)), self.threads, self.gpus)

    def environment(self, slot: int) -> dict[str, str]:
        """Environment of the worker in a slot."""
        environment = dict(os.environ)
        threads = self.threads or max(1, (os.cpu_count() or 1) // max(1, self.workers))
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            environment[name] = str(threads)
        flags = [
            flag
            for flag in environment.get("XLA_FLAGS", "").split()
            if not flag.startswith(
                ("--xla_cpu_multi_thread_eigen", "intra_op_parallelism_threads")
            )
        ]
        flags.append(
            f"--xla_cpu_multi_thread_eigen={'true' if threads > 1 else 'false'}"
        )
        flags.append(f"intra_op_parallelism_threads={threads}")
        environment["XLA_FLAGS"] = " ".join(flags)
        environment["NPROC"] = str(threads)
        if self.gpus:
            environment["CUDA_VISIBLE_DEVICES"] = self.gpus[slot % len(self.gpus)]
        return environment


def execute(
    run: Run,
    stage: Stage,
    tasks: list[Task],
    *,
    resources: Resources,
    context_factory: Callable[[], object],
    stop: Stop,
    on_progress: Callable[[], None] = lambda: None,
    poll_interval: float = 0.5,
) -> dict[str, str]:
    """Run tasks until they are done, paused or failed; return their states.

    With one worker the tasks run in this process. With more, each task runs
    in its own interpreter and at most ``workers`` of them at a time. A stop
    request lets running tasks commit and exit; a second request kills them,
    which loses at most the work since their last commit.
    """
    states: dict[str, str] = {}
    resources = resources.resolved(len(tasks))
    if resources.workers <= 1:
        context = context_factory()
        for task in tasks:
            if stop():
                states[task.id] = "paused"
                continue
            states[task.id] = run_task(run, stage, task, context, stop)
            on_progress()
        return states

    run.directory.logs.mkdir(parents=True, exist_ok=True)
    pending = list(tasks)
    running: dict[int, tuple[Task, subprocess.Popen, object]] = {}
    signalled = 0
    while pending or running:
        while pending and len(running) < resources.workers and not stop():
            task = pending.pop(0)
            slot = next(i for i in range(resources.workers) if i not in running)
            log = (run.directory.logs / f"{task.stage}-{task.id}.log").open("ab")
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "bnn.pipeline.worker",
                    str(run.directory.root),
                    task.stage,
                    task.id,
                ],
                env=resources.environment(slot),
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            running[slot] = (task, process, log)
        if stop() and pending:
            states.update({task.id: "paused" for task in pending})
            pending.clear()
        if stop.requested > signalled:
            for _, process, _ in running.values():
                if process.poll() is None:
                    process.send_signal(
                        signal.SIGTERM if stop.requested == 1 else signal.SIGKILL
                    )
            signalled = stop.requested
        for slot, (task, process, log) in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            log.close()
            del running[slot]
            states[task.id] = (
                "done"
                if code == 0
                else "paused"
                if code == PAUSED or (code < 0 and stop())
                else "busy"
                if code == BUSY
                else "failed"
            )
        on_progress()
        if running:
            time.sleep(poll_interval)
    return states


def failure_message(run: Run, task: Task) -> str:
    """What a failed task reported, and where its log is."""
    path = run.directory.work(task.stage, task.id) / "progress.json"
    message = "no message recorded"
    if path.exists():
        message = read_json(path).get("message", message)
    return f"{task.stage}/{task.id}: {message} (log: {run.directory.logs / f'{task.stage}-{task.id}.log'})"
