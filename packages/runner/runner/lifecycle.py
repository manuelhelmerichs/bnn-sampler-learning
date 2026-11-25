"""Plan, measure, inspect and summarize the cells of one experiment.

Every cell records the commit that measured it and the fingerprint of every
``bnn`` run it read. A cell reads a run only if the run matches its committed
configuration. Outside smoke tests, planning and summarizing refuse uncommitted
code, and a results folder keeps cells of one commit and one version of each
run unless a code change is explicitly allowed.
"""

from __future__ import annotations

import json
import math
import platform
import resource
import shlex
import socket
import sys
import traceback
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from runner.artifacts import (
    ArtifactError,
    atomic_json,
    atomic_text,
    read_json,
)
from runner.cells import CellSpec
from runner.manifest import (
    RunManifest,
    build_manifest,
    read_manifest,
    write_manifest,
)
from runner.paths import relative
from runner.provenance import checkout, require_clean, runtime_provenance
from runner.results.records import validated
from runner.serialization import json_value

if TYPE_CHECKING:
    from runner.experiment import Experiment


_CELL_FOLDER: ContextVar[Path] = ContextVar("cell_folder")
_RESULT_FOLDER: ContextVar[Path] = ContextVar("result_folder")


def result_folder() -> Path:
    """The saved result directory available while a scientific reduction runs."""
    try:
        return _RESULT_FOLDER.get()
    except LookupError:
        raise RuntimeError(
            "result_folder() is only available during reduction"
        ) from None


@contextmanager
def reducing_in(folder: Path):
    """Make saved cell artifacts available to a reduction without running cells."""
    token = _RESULT_FOLDER.set(Path(folder))
    try:
        yield Path(folder)
    finally:
        _RESULT_FOLDER.reset(token)


def cell_folder() -> Path:
    """The folder of the cell being measured, for artifacts beside its result."""
    try:
        return _CELL_FOLDER.get()
    except LookupError:
        raise RuntimeError(
            "cell_folder() is only available while a cell runs"
        ) from None


def _record_path(root: Path, cell_id: str) -> Path:
    return root / "cells" / cell_id / "record.json"


def _result_path(root: Path, cell_id: str) -> Path:
    return root / "cells" / cell_id / "result.json"


def _completed(record: dict[str, Any], result: dict[str, Any], cell: CellSpec) -> bool:
    return (
        record.get("state") == "completed"
        and result.get("cell", {}).get("cell_id") == cell.cell_id
        and set(cell.expected_schema) <= result.keys()
    )


def _runs(cell: CellSpec) -> dict[str, dict[str, Any]]:
    """Check every bound input of a cell, and describe the runs among them.

    A prepared input without a ``root`` is unbound, as in smoke cells.
    """
    runs = {}
    for name, value in cell.inputs.items():
        if isinstance(value, str):
            run = cell.run(name)
            runs[name] = {
                "config": value,
                "run": relative(run.directory.root),
                "fingerprint": run.fingerprint,
                "commit": run.manifest["sessions"][0]["code"]["commit"],
            }
        elif "root" in value and not cell.path(name).exists():
            raise FileNotFoundError(f"Missing input: {relative(cell.path(name))}")
    return runs


def _jobs_tsv(path: Path, manifest: RunManifest, package: str) -> None:
    rows = [
        {
            "cell_id": cell.cell_id,
            "stage": cell.stage,
            "cpu": cell.resources.cpu,
            "gpu": cell.resources.gpu,
            "memory_gb": cell.resources.memory_gb,
            "walltime_minutes": cell.resources.walltime_minutes,
            "command": shlex.join(
                [
                    "uv",
                    "run",
                    "--no-sync",
                    "python",
                    "-m",
                    package,
                    "run-cell",
                    str(path.with_name("manifest.json")),
                    cell.cell_id,
                ]
            ),
        }
        for cell in manifest.cells
    ]
    fieldnames = list(rows[0])
    lines = ["\t".join(fieldnames)]
    lines.extend("\t".join(str(row[name]) for name in fieldnames) for row in rows)
    atomic_text(path, "\n".join(lines) + "\n")


def plan(
    experiment: Experiment,
    *,
    name: str,
    module: str,
    profile: str,
    root: Path,
    package: str,
    store: Path | None = None,
    allow_dirty: bool = False,
    allow_code_change: bool = False,
) -> Path:
    """Bind inputs and write the manifest, cell records and ``jobs.tsv``.

    Completed cells are kept. Outside smoke tests, a completed cell measured at
    another commit, or on another version of a run that is now available, is
    an error unless ``allow_code_change``.
    """
    code = checkout(module)
    if profile != "smoke" and not allow_dirty:
        require_clean(code)
    design = experiment.bind(profile, store)
    manifest = build_manifest(name, design, profile, {"module": module, **code})
    root = Path(root).resolve()
    path = root / "manifest.json"
    if path.exists():
        saved = read_manifest(path)
        if (saved.experiment, saved.profile) != (name, profile):
            raise ArtifactError(
                f"{root} belongs to {saved.experiment} ({saved.profile}); "
                "choose another result folder"
            )
    reusable = set()
    for cell in manifest.cells:
        record_path = _record_path(root, cell.cell_id)
        result_path = _result_path(root, cell.cell_id)
        if result_path.exists() and not record_path.exists():
            raise ArtifactError(f"Result chunk {cell.cell_id} has no record")
        if not (record_path.exists() and result_path.exists()):
            continue
        record = read_json(record_path)
        if not _completed(record, read_json(result_path), cell):
            raise ArtifactError(
                f"Result chunk {cell.cell_id} is incompatible or corrupt"
            )
        measured = record.get("code", {}).get("commit")
        if profile != "smoke" and not allow_code_change:
            if measured != code["commit"]:
                raise ArtifactError(
                    f"Cell {cell.cell_id} was measured at commit {measured}, not "
                    f"{code['commit']}. Use a new results folder, or pass "
                    "--allow-code-change to keep it."
                )
            read = record.get("runs", {})
            try:
                current = _runs(cell) if read else {}
            except (OSError, ValueError, RuntimeError):
                current = {}  # Nothing to compare with until the run is back.
            for name, used in read.items():
                if (
                    name in current
                    and current[name]["fingerprint"] != used["fingerprint"]
                ):
                    raise ArtifactError(
                        f"Cell {cell.cell_id} read another version of "
                        f"{used['run']}. Use a new results folder, or pass "
                        "--allow-code-change to keep it."
                    )
        reusable.add(cell.cell_id)
    write_manifest(path, manifest)
    _jobs_tsv(root / "jobs.tsv", manifest, package)
    for cell in manifest.cells:
        if cell.cell_id in reusable:
            continue
        atomic_json(
            _record_path(root, cell.cell_id),
            {
                "cell_id": cell.cell_id,
                "state": "planned",
                "stage": cell.stage,
                "factors": cell.factors,
                "seeds": cell.seeds,
                "inputs": cell.inputs,
                "resources": cell.resources,
            },
        )
    return path


def _accelerator_peak_memory_mb() -> float | None:
    """Best available JAX allocator peak, without requiring an accelerator."""
    try:
        import jax

        peaks = []
        for device in jax.devices():
            if device.platform == "cpu":
                continue
            stats = device.memory_stats() or {}
            for key in ("peak_bytes_in_use", "peak_bytes_in_use.max"):
                if key in stats:
                    peaks.append(float(stats[key]) / (1024.0**2))
                    break
        return max(peaks) if peaks else None
    except (ImportError, RuntimeError):
        return None


def _peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024.0**2) if platform.system() == "Darwin" else value / 1024.0


def _require_finite_metrics(value: Any, *, label: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _require_finite_metrics(child, label=f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _require_finite_metrics(child, label=f"{label}[{index}]")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(value):
            raise ValueError(f"{label} is not finite")


def finite(entry: dict[str, Any]) -> dict[str, Any]:
    """``entry`` without its undefined measurements; summaries refuse them."""
    return {
        name: value
        for name, value in entry.items()
        if not isinstance(value, (float, list)) or np.all(np.isfinite(value))
    }


def run_cell(manifest_path: str | Path, cell_id: str, experiment: Experiment) -> Path:
    manifest = read_manifest(manifest_path)
    root = Path(manifest_path).resolve().parent
    cells = {cell.cell_id: cell for cell in manifest.cells}
    if cell_id not in cells:
        raise ArtifactError(f"Cell {cell_id!r} is not in the manifest")
    cell = cells[cell_id]
    result_path = _result_path(root, cell_id)
    record_path = _record_path(root, cell_id)
    if result_path.exists():
        if _completed(read_json(record_path), read_json(result_path), cell):
            return result_path
        raise ArtifactError(f"Cell {cell_id} has an incompatible existing result")
    code = checkout(manifest.code["module"])
    planned = manifest.code
    if manifest.profile != "smoke" and (
        code["commit"] != planned.get("commit")
        or (code["dirty"] and not planned.get("dirty"))
    ):
        raise ArtifactError("The code changed after planning; plan again")
    runs = _runs(cell)
    base = {
        "cell_id": cell_id,
        "stage": cell.stage,
        "started_at": datetime.now(UTC).isoformat(),
        "host": socket.gethostname(),
        "code": code,
        "runtime": runtime_provenance(),
        "command": sys.argv,
        "factors": cell.factors,
        "seeds": cell.seeds,
        "inputs": cell.inputs,
        "runs": runs,
        "resources_requested": cell.resources,
    }
    atomic_json(record_path, {**base, "state": "running"})
    token = _CELL_FOLDER.set(record_path.parent)
    try:
        result = experiment.execute(cell)
        result = {
            **result,
            "cell": {
                "cell_id": cell_id,
                "stage": cell.stage,
                "factors": cell.factors,
                "seeds": cell.seeds,
                "inputs": cell.inputs,
            },
            "resources": {
                "peak_rss_mb": _peak_rss_mb(),
                "accelerator_peak_memory_mb": _accelerator_peak_memory_mb(),
                "payload_bytes": len(
                    json.dumps(json_value(result), sort_keys=True).encode("utf-8")
                ),
            },
        }
        missing = set(cell.expected_schema) - set(result)
        if missing:
            raise ArtifactError(f"Cell result is missing fields: {sorted(missing)}")
        atomic_json(result_path, result)
    except Exception as error:
        atomic_json(
            record_path,
            {
                **base,
                "state": "failed",
                "finished_at": datetime.now(UTC).isoformat(),
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            },
        )
        raise
    finally:
        _CELL_FOLDER.reset(token)
    atomic_json(
        record_path,
        {**base, "state": "completed", "finished_at": datetime.now(UTC).isoformat()},
    )
    return result_path


def status(manifest_path: str | Path) -> dict[str, Any]:
    """The state of every cell, and the inputs that block pending ones."""
    manifest = read_manifest(manifest_path)
    root = Path(manifest_path).resolve().parent
    rows = []
    for cell in manifest.cells:
        state, detail = "planned", ""
        try:
            record = read_json(_record_path(root, cell.cell_id))
            state = record["state"]
            detail = record.get("error", "")
            if state == "completed" and not _completed(
                record, read_json(_result_path(root, cell.cell_id)), cell
            ):
                raise ArtifactError("Result differs from its manifest")
            if state != "completed" and manifest.profile != "smoke":
                _runs(cell)
        except (OSError, ValueError, KeyError, ArtifactError, RuntimeError) as error:
            state, detail = "blocked", str(error)
        rows.append({"cell_id": cell.cell_id, "state": state, "detail": detail})
    return {
        "experiment": manifest.experiment,
        "profile": manifest.profile,
        "counts": dict(Counter(row["state"] for row in rows)),
        "cells": rows,
    }


def summarize(
    manifest_path: str | Path, experiment: Experiment, *, allow_dirty: bool = False
) -> Path:
    """Reduce the completed cells into ``summary.json``, with every commit involved."""
    manifest = read_manifest(manifest_path)
    code = checkout(manifest.code["module"])
    if manifest.profile != "smoke" and not allow_dirty:
        require_clean(code)
    root = Path(manifest_path).resolve().parent
    results, measured, runs = [], [], {}
    for cell in manifest.cells:
        record = read_json(_record_path(root, cell.cell_id))
        if record.get("state") != "completed":
            raise ArtifactError(f"Cell {cell.cell_id} is not completed")
        result = read_json(_result_path(root, cell.cell_id))
        if not _completed(record, result, cell):
            raise ArtifactError(f"Cell {cell.cell_id} does not match its manifest")
        _require_finite_metrics(
            result.get("metrics", {}), label=f"cell {cell.cell_id} metrics"
        )
        results.append(result)
        measured.append(record["code"])
        for used in record.get("runs", {}).values():
            runs[used["fingerprint"]] = used
    with reducing_in(root):
        reduced = experiment.reduce(manifest, results)
    payload = json_value(
        {
            "experiment": manifest.experiment,
            "profile": manifest.profile,
            "consumed_cell_ids": [cell.cell_id for cell in manifest.cells],
            "code": {
                "cells": sorted({entry["commit"] or "" for entry in measured}),
                "dirty_cells": any(entry["dirty"] for entry in measured),
                "summary": code,
            },
            "runs": sorted(
                runs.values(), key=lambda used: (used["run"], used["commit"] or "")
            ),
            **reduced,
        }
    )
    validated(root, payload)  # Save only what load_record can read back.
    path = root / "summary.json"
    atomic_json(path, payload)
    return path
