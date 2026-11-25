"""Entry point of a worker process: run one task of one stage of one run."""

from __future__ import annotations

import signal
import sys
import traceback

from bnn.pipeline.workers import BUSY, PAUSED, Stop, run_task


def main(argv: list[str] | None = None) -> int:
    """Run the task named on the command line and report how it ended."""
    root, stage_name, task_id = argv if argv is not None else sys.argv[1:]
    stop = Stop()
    for number in (signal.SIGTERM, signal.SIGINT):
        signal.signal(number, lambda *_: stop.request())

    from bnn.pipeline.pipeline import configure_jax
    from bnn.pipeline.run import open_run
    from bnn.pipeline.stages import build_stages

    run = open_run(root)
    configure_jax(run.config)
    stage = build_stages()[stage_name]
    task = next(task for task in stage.tasks(run) if task.id == task_id)
    try:
        state = run_task(run, stage, task, run.context(), stop)
    except Exception:
        traceback.print_exc()
        return 1
    return {"done": 0, "paused": PAUSED, "busy": BUSY}[state]


if __name__ == "__main__":
    raise SystemExit(main())
