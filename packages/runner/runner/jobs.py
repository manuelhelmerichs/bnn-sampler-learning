"""Run cell jobs from jobs.tsv files within CPU, memory and GPU budgets."""

from __future__ import annotations

import asyncio
import csv
import math
import os
import shlex
import signal
from collections.abc import Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from runner.artifacts import atomic_json, read_json


@dataclass(frozen=True)
class Job:
    cell_id: str
    command: tuple[str, ...]
    gpu: int
    walltime_minutes: int
    log_path: Path
    cpu: int
    memory_gb: float


class Budget:
    """Admit jobs only while their combined CPU and memory requests fit."""

    def __init__(self, cpu: int, memory_gb: float):
        self.cpu = cpu
        self.memory_gb = memory_gb
        self.condition = asyncio.Condition()

    @asynccontextmanager
    async def reserve(self, job):
        async with self.condition:
            await self.condition.wait_for(
                lambda: job.cpu <= self.cpu and job.memory_gb <= self.memory_gb + 1e-9
            )
            self.cpu -= job.cpu
            self.memory_gb -= job.memory_gb
        try:
            yield
        finally:
            async with self.condition:
                self.cpu += job.cpu
                self.memory_gb += job.memory_gb
                self.condition.notify_all()


def read_jobs(paths: Sequence[Path]) -> tuple[Job, ...]:
    jobs = []
    for path in paths:
        with path.open(newline="") as handle:
            rows = csv.DictReader(handle, delimiter="\t")
            required = {
                "cell_id",
                "cpu",
                "memory_gb",
                "gpu",
                "walltime_minutes",
                "command",
            }
            if rows.fieldnames is None or not required.issubset(rows.fieldnames):
                raise ValueError(f"{path} is missing columns {sorted(required)}")
            for row in rows:
                cell_id = row["cell_id"]
                gpu = int(row["gpu"])
                if gpu not in {0, 1}:
                    raise ValueError(
                        f"Local launcher supports zero or one GPU per cell, got {gpu}"
                    )
                jobs.append(
                    Job(
                        cell_id=cell_id,
                        command=tuple(shlex.split(row["command"])),
                        gpu=gpu,
                        walltime_minutes=int(row["walltime_minutes"]),
                        log_path=path.parent / "logs" / f"{cell_id}.log",
                        cpu=int(row["cpu"]),
                        memory_gb=float(row["memory_gb"]),
                    )
                )
    return tuple(jobs)


async def _worker(
    label: str,
    queue: asyncio.Queue[Job | None],
    failures: list[tuple[str, int, Path]],
    *,
    gpu: str | None,
    accelerator_platform: str,
    budget: Budget,
) -> None:
    while True:
        job = await queue.get()
        if job is None:
            queue.task_done()
            return
        job.log_path.parent.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        environment["CUDA_VISIBLE_DEVICES"] = "" if gpu is None else gpu
        environment["JAX_PLATFORMS"] = "cpu" if gpu is None else accelerator_platform
        for name in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "TF_NUM_INTRAOP_THREADS",
        ):
            environment[name] = str(min(job.cpu, int(environment.get(name, job.cpu))))
        environment["TF_NUM_INTEROP_THREADS"] = "1"
        flags = [
            flag
            for flag in environment.get("XLA_FLAGS", "").split()
            if not flag.startswith(
                ("--xla_cpu_multi_thread_eigen=", "intra_op_parallelism_threads=")
            )
        ]
        environment["XLA_FLAGS"] = " ".join(
            [
                *flags,
                "--xla_cpu_multi_thread_eigen=false",
                f"intra_op_parallelism_threads={min(job.cpu, int(environment['OMP_NUM_THREADS']), int(environment['TF_NUM_INTRAOP_THREADS']))}",
            ]
        )
        device = "CPU" if gpu is None else f"exclusive {accelerator_platform}"
        async with budget.reserve(job):
            print(f"[{label}] start {job.cell_id} ({device})", flush=True)
            with job.log_path.open("wb") as log:
                process = await asyncio.create_subprocess_exec(
                    *job.command,
                    stdout=log,
                    stderr=asyncio.subprocess.STDOUT,
                    env=environment,
                    start_new_session=True,
                )
                try:
                    returncode = await asyncio.wait_for(
                        process.wait(), timeout=job.walltime_minutes * 60
                    )
                except (TimeoutError, asyncio.CancelledError) as error:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        await asyncio.wait_for(process.wait(), timeout=5)
                    except TimeoutError:
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        await process.wait()
                    if isinstance(error, asyncio.CancelledError):
                        raise
                    returncode = 124
                    log.write(b"\nJob exceeded its declared walltime.\n")
        if returncode:
            record_path = (
                job.log_path.parent.parent / "cells" / job.cell_id / "record.json"
            )
            if record_path.is_file():
                record = read_json(record_path)
                if record.get("state") not in {"completed", "failed"}:
                    atomic_json(
                        record_path,
                        {
                            **record,
                            "state": "failed",
                            "exit_code": returncode,
                            "error": "Job exceeded its declared walltime"
                            if returncode == 124
                            else f"Worker exited with code {returncode}; see {job.log_path}",
                        },
                    )
            failures.append((job.cell_id, returncode, job.log_path))
            print(f"[{label}] FAILED {job.cell_id}; see {job.log_path}", flush=True)
        else:
            print(f"[{label}] done  {job.cell_id}", flush=True)
        queue.task_done()


async def run_jobs(
    jobs: Sequence[Job],
    *,
    cpus: int,
    memory_gb: float,
    gpus: Sequence[str] = ("0",),
    accelerator: str = "cpu",
) -> int:
    """Run jobs concurrently while their CPU and memory requests fit the budget.

    Accelerator-tagged jobs hold one exclusive GPU lane each; with the ``cpu``
    accelerator those lanes still serialize large-memory jobs.
    """
    if cpus < 1 or not math.isfinite(memory_gb) or memory_gb <= 0:
        raise ValueError("CPU and memory budgets must be positive and finite")
    cpu_budget = cpus
    accelerator_platform = accelerator
    cpu_workers = cpus
    # Longest-processing-time ordering keeps the tail short when experiments have
    # very different resource envelopes.
    cpu_jobs = sorted(
        (job for job in jobs if job.gpu == 0),
        key=lambda job: job.walltime_minutes,
        reverse=True,
    )
    gpu_jobs = sorted(
        (job for job in jobs if job.gpu == 1),
        key=lambda job: job.walltime_minutes,
        reverse=True,
    )
    if gpu_jobs and not gpus:
        raise ValueError("Accelerator jobs are present but --gpus is empty")
    for job in jobs:
        if (
            job.cpu < 1
            or not math.isfinite(job.memory_gb)
            or job.memory_gb <= 0
            or job.walltime_minutes <= 0
        ):
            raise ValueError(f"Invalid resource request for {job.cell_id}")
        if job.cpu > cpu_budget or job.memory_gb > memory_gb:
            raise ValueError(
                f"Job {job.cell_id} requests {job.cpu} CPUs / {job.memory_gb:g} GiB; "
                f"budget is {cpu_budget} CPUs / {memory_gb:g} GiB. "
                "Raise --cpus and --memory-gb to allow this job."
            )
    budget = Budget(cpu_budget, memory_gb)

    print(
        f"Launching {len(cpu_jobs)} CPU jobs within {cpus} CPUs and "
        f"{len(gpu_jobs)} GPU jobs on {len(gpus)} exclusive GPUs.",
        flush=True,
    )
    cpu_queue: asyncio.Queue[Job | None] = asyncio.Queue()
    gpu_queue: asyncio.Queue[Job | None] = asyncio.Queue()
    for job in cpu_jobs:
        cpu_queue.put_nowait(job)
    for job in gpu_jobs:
        gpu_queue.put_nowait(job)

    failures: list[tuple[str, int, Path]] = []
    workers = [
        asyncio.create_task(
            _worker(
                f"cpu-{index}",
                cpu_queue,
                failures,
                gpu=None,
                accelerator_platform=accelerator_platform,
                budget=budget,
            )
        )
        for index in range(cpu_workers)
    ]
    workers.extend(
        asyncio.create_task(
            _worker(
                f"gpu-{gpu}",
                gpu_queue,
                failures,
                gpu=gpu,
                accelerator_platform=accelerator_platform,
                budget=budget,
            )
        )
        for gpu in gpus
    )
    for _ in range(cpu_workers):
        cpu_queue.put_nowait(None)
    for _ in gpus:
        gpu_queue.put_nowait(None)
    await asyncio.gather(*workers)

    if failures:
        print(f"\n{len(failures)} jobs failed:")
        for cell_id, returncode, log_path in failures:
            print(f"  {cell_id}: exit {returncode}; {log_path}")
        return 1
    print(f"\nCompleted all {len(jobs)} jobs.")
    return 0
