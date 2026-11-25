"""One command line for every experiment of a package.

Experiments are found by location: each package directly under the experiments
package defines one ``EXPERIMENT``, named after its folder. Every path follows
from that name: an experiment writes ``results/<name>/<profile>/``, and the
inputs it prepares itself go to ``prepared/`` within. Folder names such as
``26-11-17-name`` need not be identifiers, since experiments are loaded by name
and import their own modules relatively.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path

from runner.artifacts import ArtifactError
from runner.experiment import Experiment
from runner.jobs import read_jobs, run_jobs
from runner.lifecycle import plan, run_cell, status, summarize
from runner.manifest import read_manifest
from runner.paths import repository_root
from runner.provenance import (
    DirtyWorkspaceError,
    checkout,
    require_clean,
)

PROFILES = ("smoke", "pilot", "production")


@dataclass(frozen=True)
class Entry:
    name: str
    module: str

    def load(self) -> Experiment:
        experiment = getattr(import_module(self.module), "EXPERIMENT", None)
        if not isinstance(experiment, Experiment):
            raise TypeError(f"{self.module} does not define an EXPERIMENT")
        return experiment


def discover(package: str) -> dict[str, Entry]:
    """Every experiment of ``package``, by name, without importing them."""
    root = Path(import_module(package).__file__).parent
    return {
        folder.name: Entry(folder.name, f"{package}.{folder.name}")
        for folder in sorted(root.iterdir())
        if not folder.name.startswith((".", "_")) and (folder / "__init__.py").is_file()
    }


def main(package: str, argv=None) -> int:
    """Plan, execute, inspect, and reduce scientific experiments."""
    experiments = discover(package)
    parser = argparse.ArgumentParser(prog=f"python -m {package}", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="list experiments")
    prepare = commands.add_parser("prepare", help="build experiment-owned inputs")
    planning = commands.add_parser("plan", help="bind inputs and write the cells")
    running = commands.add_parser("run", help="plan, measure pending cells, summarize")
    checking = commands.add_parser("status", help="show progress and missing inputs")
    reducing = commands.add_parser(
        "summarize", help="reduce saved completed cells without rerunning measurements"
    )
    cell = commands.add_parser("run-cell", help=argparse.SUPPRESS)
    cell.add_argument("manifest", type=Path)
    cell.add_argument("cell_id")
    for command in (
        commands.choices["list"],
        prepare,
        planning,
        running,
        checking,
        reducing,
    ):
        command.add_argument(
            "selectors", nargs="*", metavar="experiment", help="default: all of them"
        )
    for command in (prepare, planning, running, checking, reducing):
        command.add_argument("--profile", choices=PROFILES, required=True)
        command.add_argument(
            "--root", type=Path, help="override the results folder [results]"
        )
    for command in (prepare, planning, running, reducing):
        command.add_argument(
            "--allow-dirty",
            action="store_true",
            help="run uncommitted code and record the result as dirty",
        )
    for command in (planning, running):
        command.add_argument(
            "--allow-code-change",
            action="store_true",
            help="keep completed cells that another commit measured",
        )
    running.add_argument("--cpus", type=int, default=4)
    running.add_argument("--memory-gb", type=float, default=8.0)
    running.add_argument("--gpus", default="0", help="comma-separated GPU IDs")
    running.add_argument("--accelerator", choices=("cpu", "mps", "cuda"), default="cpu")
    args = parser.parse_args(argv)

    if args.command == "run-cell":
        manifest = read_manifest(args.manifest)
        entry = next(
            e for e in experiments.values() if e.module == manifest.code["module"]
        )
        print(run_cell(args.manifest, args.cell_id, entry.load()))
        return 0

    unknown = [s for s in args.selectors if s not in experiments]
    if unknown:
        parser.error(f"Unknown experiment {unknown[0]!r}; use list to see them")
    selected = [
        entry
        for name, entry in experiments.items()
        if not args.selectors or name in args.selectors
    ]
    if args.command == "list":
        for entry in selected:
            print(entry.name)
        return 0

    results = (args.root or repository_root() / "results").resolve()
    folder = {entry.name: results / entry.name / args.profile for entry in selected}

    def prepared(entry: Entry, experiment: Experiment) -> Path | None:
        return folder[entry.name] / "prepared" if experiment.prepare else None

    try:
        if args.command == "prepare":
            for entry in selected:
                if args.profile != "smoke" and not args.allow_dirty:
                    require_clean(checkout(entry.module))
                experiment = entry.load()
                if experiment.prepare is None:
                    print(f"{entry.name}: nothing to prepare")
                    continue
                if args.profile == "smoke":
                    print(f"{entry.name}: smoke cells use synthetic inputs")
                    continue
                store = prepared(entry, experiment)
                store.mkdir(parents=True, exist_ok=True)
                experiment.prepare(args.profile, store)
                print(store)
            return 0

        if args.command == "summarize":
            for entry in selected:
                path = folder[entry.name] / "manifest.json"
                if path.is_file():
                    print(summarize(path, entry.load(), allow_dirty=args.allow_dirty))
                elif args.selectors:
                    parser.error(f"No saved manifest: {path}")
            return 0

        if args.command == "status":
            for entry in selected:
                path = folder[entry.name] / "manifest.json"
                report = (
                    status(path)
                    if path.is_file()
                    else {"experiment": entry.name, "counts": {"unplanned": 1}}
                )
                print(json.dumps(report, indent=2))
            return 0

        if args.command in {"plan", "run"}:
            if not args.selectors and args.command == "run":
                parser.error("name the experiments to run")
            paths = []
            for entry in selected:
                experiment = entry.load()
                paths.append(
                    plan(
                        experiment,
                        name=entry.name,
                        module=entry.module,
                        profile=args.profile,
                        root=folder[entry.name],
                        package=package,
                        store=prepared(entry, experiment),
                        allow_dirty=args.allow_dirty,
                        allow_code_change=args.allow_code_change,
                    )
                )
                print(paths[-1])
            if args.command == "plan":
                return 0
    except (ArtifactError, DirtyWorkspaceError, FileNotFoundError) as error:
        parser.error(str(error))

    reports = [status(path) for path in paths]
    blocked = [r for r in reports if "blocked" in r["counts"]]
    if blocked:
        for report in blocked:
            print(json.dumps(report, indent=2))
        return 1
    gpus = tuple(value.strip() for value in args.gpus.split(",") if value.strip())
    if len(gpus) != len(set(gpus)):
        parser.error("--gpus cannot repeat a device")
    completed = {
        row["cell_id"]
        for report in reports
        for row in report["cells"]
        if row["state"] == "completed"
    }
    jobs = [
        job
        for job in read_jobs([path.with_name("jobs.tsv") for path in paths])
        if job.cell_id not in completed
    ]
    try:
        failed = asyncio.run(
            run_jobs(
                jobs,
                cpus=args.cpus,
                memory_gb=args.memory_gb,
                gpus=gpus,
                accelerator=args.accelerator,
            )
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if failed:
        return failed
    for entry, path in zip(selected, paths, strict=True):
        print(summarize(path, entry.load(), allow_dirty=args.allow_dirty))
    return 0
