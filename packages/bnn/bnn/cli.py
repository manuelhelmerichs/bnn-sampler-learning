"""The command line: ``bnn run``, ``status``, ``validate``, ``list`` and ``template``."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from bnn.config import STAGES

PAUSED_EXIT = 75

EPILOG = """\
examples:
  bnn validate configs/airfoil.yaml
      parse the configuration and show the runs it describes

  bnn run configs/airfoil.yaml --workers 8
      create the run, or continue it wherever it stopped

  bnn run configs/airfoil.yaml --shard 3/16
      work on every 16th chain group starting at the fourth, on one of many machines

  bnn status runs/trajectory
      show the progress of all runs below a folder

Ctrl-C pauses a run after the block in progress. A second Ctrl-C stops at once;
the run then continues from its last commit.
"""


def _parse_value(text: str) -> Any:
    return yaml.safe_load(text)


def _overrides(items: list[str]) -> dict[str, Any]:
    overrides = {}
    for item in items:
        key, separator, value = item.partition("=")
        if not separator or not key:
            raise argparse.ArgumentTypeError(f"--set expects key=value, got {item!r}.")
        overrides[key] = _parse_value(value)
    return overrides


def _stage_list(text: str) -> tuple[str, ...]:
    stages = tuple(part.strip() for part in text.split(",") if part.strip())
    unknown = sorted(set(stages) - set(STAGES))
    if not stages or unknown:
        raise argparse.ArgumentTypeError(
            f"choose stages from {', '.join(STAGES)}"
            + (f"; unknown: {', '.join(unknown)}" if unknown else "")
        )
    return stages


def _shard(text: str) -> tuple[int, int]:
    try:
        index, count = (int(part) for part in text.split("/"))
    except ValueError:
        raise argparse.ArgumentTypeError("--shard expects INDEX/COUNT.") from None
    if not 0 <= index < count:
        raise argparse.ArgumentTypeError("--shard needs 0 <= INDEX < COUNT.")
    return index, count


def _workers(text: str) -> int | None:
    if text == "auto":
        return None
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("--workers must be positive or 'auto'.")
    return value


def build_parser() -> argparse.ArgumentParser:
    """The argument parser of the ``bnn`` command."""
    parser = argparse.ArgumentParser(
        prog="bnn",
        description="Fit and sample posteriors of Bayesian neural networks.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    run = commands.add_parser(
        "run",
        help="create runs or continue them where they stopped",
        description="Bring every run of a configuration to completion. The "
        "command is idempotent: it skips complete stages and continues "
        "interrupted ones bit for bit.",
    )
    run.add_argument("config", type=Path, help="YAML file or a folder of YAML files")
    run.add_argument(
        "--root", type=Path, default=Path("runs"), help="folder of all runs [runs]"
    )
    run.add_argument(
        "--stages",
        type=_stage_list,
        metavar="A,B",
        help=f"subset of {','.join(STAGES)}",
    )
    run.add_argument(
        "--workers",
        type=_workers,
        default=None,
        metavar="N",
        help="processes running chain groups in parallel [auto: one per group, "
        "at most one per core]",
    )
    run.add_argument(
        "--threads",
        type=int,
        metavar="N",
        help="compute threads per worker [cores/workers]",
    )
    run.add_argument(
        "--gpus",
        type=lambda text: tuple(part for part in text.split(",") if part),
        default=(),
        metavar="IDS",
        help="accelerators assigned to the workers in turn, e.g. 0,1",
    )
    run.add_argument(
        "--shard",
        type=_shard,
        metavar="I/N",
        help="only work on every N-th task starting at I; for arrays of jobs",
    )
    run.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override an option, e.g. sampler.steps=2000",
    )
    run.add_argument(
        "--allow-dirty",
        action="store_true",
        help="run uncommitted code and record the run as dirty",
    )
    run.add_argument(
        "--allow-code-change",
        action="store_true",
        help="continue a run that was started with different code",
    )
    run.add_argument("--quiet", action="store_true", help="print only the outcome")

    status = commands.add_parser("status", help="show the progress of runs")
    status.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[Path("runs")],
        help="runs or folders of runs",
    )

    validate = commands.add_parser(
        "validate", help="check configurations without running"
    )
    validate.add_argument("config", type=Path, nargs="+")
    validate.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE"
    )

    listing = commands.add_parser("list", help="list the available components")
    listing.add_argument(
        "family",
        nargs="?",
        choices=("models", "fits", "samplers", "priors"),
    )

    template = commands.add_parser("template", help="print a documented configuration")
    template.add_argument("--model", default="FCN")
    template.add_argument("--fit", default="mode")
    template.add_argument(
        "--sampler",
        default="mclmc",
        help="a sampler, or 'none' for a run that ends with its fit",
    )
    return parser


# --- status ---------------------------------------------------------------------


def _find_runs(paths: list[Path]) -> list[Path]:
    found = []
    for path in paths:
        if (path / "run.json").is_file():
            found.append(path)
        elif path.is_dir():
            found.extend(sorted(p.parent for p in path.rglob("run.json")))
    return found


def _duration(seconds: float) -> str:
    seconds = int(max(0, seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, seconds = divmod(rest, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    return f"{minutes}m{seconds:02d}s" if minutes else f"{seconds}s"


def status_lines(run, *, now: float | None = None) -> list[str]:
    """A plain-text progress report of one run."""
    from bnn.pipeline.run import chain_progress
    from bnn.pipeline.stages import build_stages

    now = time.time() if now is None else now
    stages = build_stages()
    lines = [str(run.directory.root)]
    for name in STAGES:
        stage = stages[name]
        if not stage.enabled(run):
            continue
        status = run.stage_status(name)
        tasks = stage.tasks(run)
        done = sum(stage.is_done(run, task) for task in tasks)
        summary = f"  {name:<9} {status:<9}"
        if status != "complete" and (done or status == "running"):
            summary += f" {done}/{len(tasks)} tasks"
        lines.append(summary.rstrip())
        if status == "complete":
            continue
        for record in chain_progress(run, name):
            if record.get("state") == "done":
                continue
            chains = ",".join(map(str, record["chains"]))
            line = f"    {record['task']:<11} chains {chains:<8} {record['state']:<8}"
            if "steps" in record:
                fraction = record["step"] / max(record["steps"], 1)
                line += (
                    f" {record['phase']} ({record['phase_index'] + 1}/{record['phases']})"
                    f" {record['step']:,}/{record['steps']:,} {fraction:4.0%}"
                )
                rate = record.get("rate")
                if rate and record["state"] == "running":
                    stale = now - record.get("updated", now) > 120
                    line += f"  {rate:,.0f} it/s"
                    line += (
                        "  (no update for 2 min)"
                        if stale
                        else f"  eta {_duration((record['steps'] - record['step']) / rate)}"
                    )
            if record.get("message"):
                line += f"  {record['message']}"
            lines.append(line)
    return lines


# --- commands -------------------------------------------------------------------


class _Display:
    """Progress while a run advances: a live view on a terminal, lines otherwise."""

    def __init__(self, quiet: bool):
        self.quiet = quiet
        self.live = None
        self._last = 0.0
        if not quiet and sys.stdout.isatty():
            from rich.console import Console
            from rich.live import Live

            self.live = Live(console=Console(), refresh_per_second=4, transient=True)
            self.live.start()

    def update(self, run, stage: str) -> None:
        if self.quiet:
            return
        now = time.time()
        if self.live is not None:
            if now - self._last >= 0.25:
                from rich.text import Text

                self.live.update(Text("\n".join(status_lines(run, now=now))))
                self._last = now
        elif now - self._last >= 30:
            print("\n".join(status_lines(run, now=now)), flush=True)
            self._last = now

    def close(self) -> None:
        if self.live is not None:
            self.live.stop()


def _run(arguments) -> int:
    from bnn.config import load_configs
    from bnn.pipeline.pipeline import advance
    from bnn.pipeline.provenance import DirtyWorkspaceError
    from bnn.pipeline.run import RunError
    from bnn.pipeline.workers import Resources, Stop

    configs = load_configs(arguments.config, _overrides(arguments.overrides))
    stop = Stop()

    def interrupt(signum, frame):
        stop.request()
        if stop.requested == 1:
            print(
                "\nPausing after the block in progress. Interrupt again to stop now.",
                file=sys.stderr,
                flush=True,
            )
        else:
            raise KeyboardInterrupt

    previous = {
        number: signal.signal(number, interrupt)
        for number in (signal.SIGINT, signal.SIGTERM)
    }
    resources = Resources(arguments.workers, arguments.threads, arguments.gpus)
    exit_code = 0
    try:
        for config in configs:
            display = _Display(arguments.quiet)
            try:
                outcome = advance(
                    config,
                    arguments.root,
                    stages=arguments.stages,
                    resources=resources,
                    shard=arguments.shard,
                    allow_dirty=arguments.allow_dirty,
                    allow_code_change=arguments.allow_code_change,
                    stop=stop,
                    on_progress=display.update,
                )
            except (RunError, DirtyWorkspaceError) as error:
                display.close()
                print(f"error: {error}", file=sys.stderr)
                return 1
            except KeyboardInterrupt:
                display.close()
                print("Stopped. Run the same command to continue.", file=sys.stderr)
                return PAUSED_EXIT
            finally:
                display.close()
            location = outcome.run.directory.root
            if outcome.state == "complete":
                print(f"complete  {location}")
            elif outcome.state == "failed":
                print(f"failed    {location}  [{outcome.stage}]\n{outcome.detail}")
                exit_code = 1
            else:
                print(
                    f"{outcome.state:<9} {location}  [{outcome.stage}] {outcome.detail}"
                )
                if outcome.state == "paused":
                    return PAUSED_EXIT
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)
    return exit_code


def _status(arguments) -> int:
    from bnn.pipeline.run import open_run

    runs = _find_runs(arguments.paths)
    if not runs:
        print("No runs found.")
        return 1
    for path in runs:
        try:
            print("\n".join(status_lines(open_run(path))))
        except (OSError, ValueError, KeyError) as error:
            reason = str(error).splitlines()[0] if str(error) else type(error).__name__
            print(f"{path}\n  unreadable: {reason}")
    return 0


def _validate(arguments) -> int:
    from bnn.config import load_configs

    overrides = _overrides(arguments.overrides)
    for path in arguments.config:
        for config in load_configs(path, overrides):
            sampler = config.sampler
            sampling = (
                "no sampler"
                if sampler is None
                else f"{sampler.name} {sampler.steps} steps (warmup "
                f"{sampler.warmup}, thinning {sampler.thinning})"
            )
            stages = [stage for stage in STAGES if config.requests(stage)]
            print(
                f"{config.relative_directory}: {config.model.name} on "
                f"{Path(config.data.path).name}, {config.chains} chains, fit "
                f"{config.fit.name}, {sampling}; stages {', '.join(stages)}"
            )
    return 0


def _members(family) -> dict[str, type]:
    return dict(family.members.items())


def _list(arguments) -> int:
    from bnn.fitting.config import FitConfig
    from bnn.models import ModelConfig
    from bnn.samplers import SamplerConfig
    from bnn.target import PriorDist

    def describe(members):
        return [
            f"{name:<30} {(cls.__doc__ or '').strip().splitlines()[0]}"
            for name, cls in sorted(members.items())
        ]

    sections = {
        "models": describe(_members(ModelConfig)),
        "fits": describe(_members(FitConfig)),
        "samplers": describe(_members(SamplerConfig)),
        "priors": [member.value for member in PriorDist],
    }
    for name, lines in sections.items():
        if arguments.family in (None, name):
            print(f"{name}:")
            print("\n".join(f"  {line}" for line in lines))
    return 0


def _template_lines(cls, values: dict, indent: int = 0) -> list[str]:
    """YAML of a configuration class with its option descriptions as comments."""
    from bnn.registry import Config

    pad = "  " * indent
    lines = []
    for name, field in cls.model_fields.items():
        value = values.get(name, field.default)
        if field.description:
            lines.append(f"{pad}# {field.description}")
        if isinstance(value, Config):
            lines.append(f"{pad}{name}:")
            lines += _template_lines(type(value), value.to_dict(), indent + 1)
        elif field.is_required() and name not in values:
            lines.append(f"{pad}{name}:  # required")
        else:
            if hasattr(value, "value"):
                value = value.value
            lines.append(f"{pad}{name}: {json.dumps(value)}")
    return lines


def _template(arguments) -> int:
    from bnn.config import ExecutionConfig, RunConfig
    from bnn.data.config import DataConfig
    from bnn.fitting.config import FitConfig
    from bnn.models import ModelConfig
    from bnn.samplers import SamplerConfig
    from bnn.target import PriorConfig

    try:
        sections: list[tuple[str, type, dict]] = [
            ("data", DataConfig, {}),
            ("model", ModelConfig.members[arguments.model], {}),
            ("prior", PriorConfig, {}),
            ("fit", FitConfig.members[arguments.fit], {}),
        ]
        if arguments.sampler != "none":
            sections.append(("sampler", SamplerConfig.members[arguments.sampler], {}))
    except KeyError as error:
        raise ValueError(error.args[0]) from None
    sections.append(("execution", ExecutionConfig, {}))
    lines = []
    for name in ("name", "group", "seed", "x64", "chains"):
        field = RunConfig.model_fields[name]
        if field.description:
            lines.append(f"# {field.description}")
        default = "" if name == "name" else field.default
        suffix = "  # required" if name == "name" else ""
        lines.append(f"{name}: {json.dumps(default)}{suffix}")
    for name, cls, values in sections:
        lines.append(f"{name}:")
        lines += _template_lines(cls, values, 1)
    print("\n".join(lines))
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the ``bnn`` command."""
    arguments = build_parser().parse_args(argv)
    handler = {
        "run": _run,
        "status": _status,
        "validate": _validate,
        "list": _list,
        "template": _template,
    }[arguments.command]
    try:
        return handler(arguments)
    except (FileNotFoundError, ValueError) as error:
        if os.environ.get("BNN_TRACEBACK"):
            raise
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
