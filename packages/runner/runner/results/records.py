"""Read complete scientific records without importing their experiments."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from runner.artifacts import read_json


@dataclass(frozen=True)
class Record:
    experiment: str
    summary: dict[str, Any]
    manifest: dict[str, Any]
    rows: list[dict[str, Any]]
    root: Path

    def cell(self, cell_id: str) -> Path:
        """The folder holding a completed cell's result and artifacts."""
        if not cell_id or Path(cell_id).name != cell_id or cell_id in {".", ".."}:
            raise ValueError(f"Invalid cell identity: {cell_id}")
        return self.root / "cells" / cell_id

    def artifact(self, cell_id: str, relative: str, sha256: str | None = None) -> Path:
        """Resolve an artifact within its cell and optionally verify its bytes."""
        folder = self.cell(cell_id).resolve()
        path = (folder / relative).resolve()
        if not path.is_relative_to(folder):
            raise ValueError(f"Artifact escapes its cell: {relative}")
        if not path.is_file():
            raise FileNotFoundError(f"Missing scientific artifact: {path}")
        if sha256 is not None:
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != sha256:
                raise ValueError(f"Artifact identity differs: {path}")
        return path


def load_record(root: Path) -> Record | None:
    """Read a complete record of any profile, including rejected outcomes.

    Public-export eligibility is a decision of the consuming application.
    """
    root = Path(root).resolve()
    path = root / "summary.json"
    if not path.is_file():
        return None
    return validated(root, read_json(path))


def validated(root: Path, summary: dict[str, Any]) -> Record:
    """``summary`` as the record of the manifest and cells saved in ``root``.

    Validate saved fields and artifacts directly. ``summarize`` checks a summary
    with this before saving it, so every saved summary can be read back.
    """
    path = root / "summary.json"
    missing = {"experiment", "profile", "consumed_cell_ids"} - set(summary)
    if missing:
        raise RuntimeError(f"{path} is missing {sorted(missing)}")
    experiment = summary["experiment"]
    if not summary["consumed_cell_ids"]:
        raise RuntimeError(f"{path} consumed no cells")
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest["experiment"] != experiment
        or manifest["profile"] != summary["profile"]
    ):
        raise RuntimeError(f"{manifest_path} has the wrong experiment or profile")
    cells = manifest["cells"]
    cell_ids = [cell["cell_id"] for cell in cells]
    if len(cell_ids) != len(set(cell_ids)):
        raise RuntimeError(f"{manifest_path} repeats cell IDs")
    if cell_ids != summary["consumed_cell_ids"]:
        raise RuntimeError(f"{manifest_path} cell order does not match summary.json")
    derived = summary.get("row_summaries", {})
    if set(derived) - set(cell_ids):
        raise RuntimeError(f"{experiment} has summaries for unknown cells")
    metrics = summary.get("cell_metrics", [])
    if len(metrics) != len(cells):
        raise RuntimeError(f"{experiment} has incomplete cell metrics")
    record = Record(experiment, summary, manifest, [], root)
    for cell, row in zip(cells, metrics, strict=True):
        factors = cell["factors"]
        clashes = sorted(k for k in row.keys() & factors.keys() if row[k] != factors[k])
        if clashes:
            raise RuntimeError(
                f"{experiment} cell {cell['cell_id']} has metrics named like its "
                f"factors but with other values: {clashes}"
            )
        if set(derived.get(cell["cell_id"], {})) & (factors.keys() | row.keys()):
            raise RuntimeError(f"{experiment} derived summaries overwrite measurements")
        folder = record.cell(cell["cell_id"])
        result = json.loads((folder / "result.json").read_text())
        state = json.loads((folder / "record.json").read_text())
        saved = result.get("cell", {})
        if (
            state.get("state") != "completed"
            or saved.get("cell_id") != cell["cell_id"]
            or saved.get("factors") != factors
            or saved.get("seeds") != cell["seeds"]
            or saved.get("inputs", {}) != cell.get("inputs", {})
            or result.get("metrics") != row
        ):
            raise RuntimeError(f"{folder} disagrees with its saved summary")
        artifact = row.get("artifact")
        if isinstance(artifact, dict) and "path" in artifact:
            record.artifact(cell["cell_id"], artifact["path"], artifact.get("sha256"))
    rows = [
        dict(cell["factors"])
        | dict(row)
        | summary.get("row_summaries", {}).get(cell["cell_id"], {})
        for cell, row in zip(cells, metrics, strict=True)
    ]
    return Record(experiment, summary, manifest, rows, root)


def part_record(record, part):
    """The record of one part, selected by its recorded cell IDs."""
    from dataclasses import replace

    summary, manifest = record.summary, record.manifest
    metadata = summary["parts"][part]
    owned = set(metadata["consumed_cell_ids"])
    selected = [
        (cell, row, metrics)
        for cell, row, metrics in zip(
            manifest["cells"], record.rows, summary["cell_metrics"], strict=True
        )
        if cell["cell_id"] in owned
    ]
    if [cell["cell_id"] for cell, _, _ in selected] != metadata["consumed_cell_ids"]:
        raise RuntimeError(
            f"{record.experiment}/{part} has inconsistent cell membership"
        )
    return replace(
        record,
        summary={
            **summary,
            **metadata,
            "part": part,
            "cell_metrics": [metrics for _, _, metrics in selected],
            "row_summaries": {
                key: value
                for key, value in summary.get("row_summaries", {}).items()
                if key in owned
            },
        },
        manifest={**manifest, "cells": [cell for cell, _, _ in selected]},
        rows=[row for _, row, _ in selected],
    )
