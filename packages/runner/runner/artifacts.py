"""Atomic JSON, text and numeric artifacts, read back with validation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from runner.serialization import json_value


class ArtifactError(RuntimeError):
    """An artifact is absent, malformed, or incompatible."""


def atomic_text(path: str | Path, text: str) -> Path:
    """Atomically replace a text file, leaving no partial destination."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{uuid4().hex}.tmp"
    )
    try:
        temporary.write_text(text)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def atomic_json(path: str | Path, payload: Any) -> Path:
    text = json.dumps(json_value(payload), indent=2, sort_keys=True, allow_nan=False)
    return atomic_text(path, text + "\n")


def read_json(path: str | Path) -> Any:
    source = Path(path)
    try:
        return json.loads(source.read_text())
    except FileNotFoundError as error:
        raise ArtifactError(f"Missing artifact: {source}") from error
    except json.JSONDecodeError as error:
        raise ArtifactError(f"Invalid JSON artifact: {source}") from error


def atomic_npz(path: str | Path, **arrays) -> Path:
    """Atomically write a compressed numeric checkpoint."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{uuid4().hex}.npz")
    try:
        np.savez_compressed(temporary, **arrays)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def read_arrays(path: str | Path) -> dict[str, np.ndarray]:
    """The arrays of a checkpoint, refusing empty or nonfinite ones."""
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    if not arrays or any(not np.isfinite(value).all() for value in arrays.values()):
        raise ArtifactError(f"Empty or nonfinite numeric checkpoint: {path}")
    return arrays
