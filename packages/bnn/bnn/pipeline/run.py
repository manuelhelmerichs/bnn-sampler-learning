"""A run on disk: its manifest, its chain groups and read access to its results.

``open_run`` and ``run_of`` are how other code reads a run. They never start
computation.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import socket
import sys
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import yaml

from bnn.config import STAGES, RunConfig, dump_config, load_configs
from bnn.pipeline.layout import RunDirectory, resolve_path
from bnn.pipeline.provenance import file_digest
from bnn.store.io import lock, read_json, write_json
from bnn.store.positions import PositionIndex
from bnn.store.trees import load_array_tree
from bnn.trees import tree_stack


class RunError(RuntimeError):
    """A run cannot be created, continued or read as requested."""


def _now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")


def plan_groups(chains: int, group_size: int) -> list[list[int]]:
    """Partition chains into consecutive groups that are compiled together."""
    return [
        list(range(start, min(start + group_size, chains)))
        for start in range(0, chains, group_size)
    ]


@dataclass(frozen=True)
class Run:
    """An existing run."""

    directory: RunDirectory

    @cached_property
    def config(self) -> RunConfig:
        """The configuration the run was created with."""
        return RunConfig.model_validate(
            yaml.safe_load(self.directory.config.read_text())
        )

    @property
    def manifest(self) -> dict:
        """The current manifest."""
        return read_json(self.directory.manifest)

    @property
    def fingerprint(self) -> str:
        """Hash of everything the stored results depend on.

        These are the options of every stage, the data and the code of every
        invocation. Two runs with equal fingerprints hold the same results; a
        run sampled again with other code gets a new one.
        """
        manifest = self.manifest
        codes = {
            (session["code"]["source_sha256"], session["code"]["lock_sha256"])
            for session in manifest["sessions"]
        }
        document = {
            "identities": manifest["identities"],
            "inputs": manifest["inputs"],
            "code": sorted(codes),
        }
        encoded = json.dumps(document, sort_keys=True).encode()
        return hashlib.sha256(encoded).hexdigest()

    @property
    def groups(self) -> list[list[int]]:
        """The chain groups, fixed when the run was created."""
        return self.manifest["groups"]

    def stage_status(self, stage: str) -> str:
        """``pending``, ``running`` or ``complete``."""
        return self.manifest["stages"].get(stage, {}).get("status", "pending")

    def update(self, change) -> dict:
        """Apply ``change(manifest)`` under the run lock."""
        with lock(self.directory.lock()):
            manifest = self.manifest
            change(manifest)
            write_json(self.directory.manifest, manifest)
            return manifest

    def set_stage(self, stage: str, status: str) -> None:
        """Record the status of a stage."""

        def change(manifest):
            record = manifest["stages"].setdefault(stage, {})
            record["status"] = status
            record["updated"] = _now()

        self.update(change)

    def require(self, stage: str) -> None:
        """Fail unless a stage has completed."""
        if self.stage_status(stage) != "complete":
            raise RunError(
                f"Stage {stage!r} of {self.directory.root} is "
                f"{self.stage_status(stage)}; run `bnn run` to complete it."
            )

    # --- results ------------------------------------------------------------

    def context(self):
        """Dataset, network and posterior; requires this run's precision."""
        from bnn.pipeline.context import Context

        # The first context saves the partition; later ones verify it.
        with lock(self.directory.root / ".context.lock"):
            return Context(self.config, self.directory)

    def dataset(self):
        """The dataset with this run's partition; independent of the precision."""
        from bnn.pipeline.context import load_dataset

        with lock(self.directory.root / ".context.lock"):
            return load_dataset(self.config, self.directory)

    def module(self):
        """The Flax network of this run."""
        from bnn.models import build_module

        return build_module(self.config.model)

    def model(self):
        """The posterior of this run: its network, prior and likelihood.

        Unlike ``context``, it loads no data and evaluates at whatever
        precision its inputs have.
        """
        from bnn.target import ProbabilisticModel

        self.require("fit")
        state = self.directory.model_state
        return ProbabilisticModel(
            self.module(),
            self.config.prior.build(),
            self.config.data.task,
            load_array_tree(state)[0] if state.exists() else {},
        )

    def fitted(self, chains=None) -> Any:
        """Fitted positions with a leading chain axis, on the host.

        They are the result of a run without sampler and otherwise the starts
        of its chains.
        """
        self.require("fit")
        chains = range(self.config.chains) if chains is None else chains
        loaded = [load_array_tree(self.directory.fitted(chain))[0] for chain in chains]
        return tree_stack([item["position"] for item in loaded])

    def samples(self, *, require_complete: bool = True) -> PositionIndex:
        """The posterior draws."""
        if require_complete:
            self.require("sample")
        return PositionIndex.open(
            self.directory.samples, require_complete=require_complete
        )

    def warmup(self) -> PositionIndex:
        """The stored warmup positions."""
        self.require("sample")
        return PositionIndex.open(self.directory.warmup)

    def parameters(self, chain: int) -> dict:
        """Adapted kernel parameters of one chain."""
        self.require("sample")
        return load_array_tree(self.directory.parameters(chain))[0]

    def diagnostics(self, phase: str = "sample", stage: str = "sample") -> dict:
        """Per-transition diagnostics of a phase, with chain and step axes."""
        directory = self.directory.diagnostics(stage) / phase
        index = PositionIndex.open(directory)
        return {**index.load(), "step": index.draw_ids.to_numpy()}


def open_run(path: str | Path) -> Run:
    """Open an existing run directory."""
    directory = RunDirectory(Path(path))
    if not directory.manifest.is_file():
        raise FileNotFoundError(f"Not a run directory: {path}")
    return Run(directory)


def run_of(config: str | Path, root: str | Path = "runs") -> Run:
    """The finished run of a configuration file, checked against that file.

    The file describes one run. Its run below ``root`` must have been computed
    with exactly the file's options and data, and every stage the file
    requests must be complete.
    """
    configs = load_configs(config)
    if len(configs) != 1:
        raise RunError(f"{config} describes {len(configs)} runs instead of one.")
    resolved = configs[0]
    directory = RunDirectory(Path(root) / resolved.relative_directory)
    command = f"`bnn run {config}`"
    if directory.root.exists() and not directory.manifest.is_file():
        raise RunError(
            f"{directory.root} is not a bnn run; move it away and create the run "
            f"with {command}."
        )
    if not directory.manifest.is_file():
        raise RunError(f"{directory.root} does not exist; create it with {command}.")
    run = Run(directory)
    manifest = run.manifest
    changed, pending = [], []
    for stage in STAGES:
        if not resolved.requests(stage):
            continue
        status = run.stage_status(stage)
        # A stage that has not started takes the file's options when it runs.
        if status != "pending" and (
            manifest["identities"].get(stage) != resolved.identity(stage)
        ):
            changed.append(stage)
        elif status != "complete":
            pending.append(stage)
    if manifest["inputs"] != _inputs(resolved, directory):
        changed.append("data")
    if changed:
        raise RunError(
            f"{directory.root} differs from {config} in {', '.join(changed)}; "
            f"move it away and sample it again with {command}."
        )
    if pending:
        raise RunError(
            f"Unfinished stages of {directory.root}: {', '.join(pending)}; "
            f"continue it with {command}."
        )
    return run


def _inputs(config: RunConfig, directory: RunDirectory) -> dict:
    """Hashes of the input files a run depends on."""
    source = resolve_path(config.data.path, directory.root)
    if not source.is_file():
        raise RunError(f"The data file {config.data.path} does not exist.")
    return {"data_sha256": file_digest(source)}


def create_or_resume(
    config: RunConfig,
    root: str | Path,
    code: dict,
    *,
    group_size: int,
    allow_code_change: bool = False,
) -> Run:
    """Create the run of a configuration, or check that it can be continued.

    A stage keeps its results only for the options it was computed with, so
    changed options of a started stage are an error. A stage that is not
    requested keeps what it computed, so a sampler can be added to a run that
    ended with its fit.
    """
    directory = RunDirectory(Path(root) / config.relative_directory)
    identities = {stage: config.identity(stage) for stage in STAGES}
    inputs = _inputs(config, directory)
    session = {
        "started": _now(),
        "host": socket.gethostname(),
        "command": " ".join(sys.argv),
        "code": code,
    }
    directory.root.mkdir(parents=True, exist_ok=True)
    with lock(directory.lock()):
        if not directory.manifest.exists():
            if any(p.name != ".run.lock" for p in directory.root.iterdir()):
                raise RunError(
                    f"{directory.root} exists but is not a run; choose another name."
                )
            directory.config.write_text(dump_config(config))
            write_json(
                directory.manifest,
                {
                    "name": config.name,
                    "created": _now(),
                    "identities": identities,
                    "inputs": inputs,
                    "groups": plan_groups(config.chains, group_size),
                    "sessions": [session],
                    "stages": {},
                },
            )
            return Run(directory)
        manifest = read_json(directory.manifest)
        first = manifest["sessions"][0]["code"]
        if not allow_code_change and (
            first["source_sha256"] != code["source_sha256"]
            or first["lock_sha256"] != code["lock_sha256"]
        ):
            raise RunError(
                f"{directory.root} was started with different code (commit "
                f"{first['commit']}). Continue it with that code, start a new run, "
                "or pass --allow-code-change to accept results of mixed code."
            )
        if manifest["inputs"] != inputs:
            raise RunError(
                f"The data of {directory.root} changed since the run was created."
            )
        for stage in STAGES:
            started = manifest["stages"].get(stage, {}).get("status", "pending")
            if not config.requests(stage):
                # A stage that is no longer requested keeps what it computed.
                identities[stage] = manifest["identities"][stage]
                continue
            if manifest["identities"][stage] == identities[stage]:
                continue
            if started == "pending":
                continue
            raise RunError(
                f"Stage {stage!r} of {directory.root} was computed with different "
                "options; use a new run name."
            )
        manifest["identities"] = identities
        manifest["sessions"].append(session)
        write_json(directory.manifest, manifest)
        directory.config.write_text(dump_config(config))
    return Run(directory)


def chain_progress(run: Run, stage: str) -> list[dict]:
    """Progress records of the tasks of a stage, most recent state first."""
    root = run.directory.root / "work" / stage
    records = []
    if root.exists():
        for path in sorted(root.glob("*/progress.json")):
            try:
                records.append(read_json(path))
            except (OSError, ValueError):
                continue
    return records
