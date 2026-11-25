"""Typed numeric trajectory bundles in the sharded Zarr position store.

Each named array preserves its full scientific shape and dtype. Unicode labels
are declarative manifest metadata; numeric arrays use sharded Zarr. Its first axis
is stored as draws of one physical chain; that storage axis does not relabel a
study's scientific chain, replica, or time axes. Numeric result checkpoints can
use NPZ separately.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from uuid import uuid4

import numpy as np

from bnn.store.io import write_json
from bnn.store.positions import (
    PositionIndex,
    PositionWriter,
    begin_chains,
    complete_chains,
)


def _header(array):
    return {"shape": list(array.shape), "dtype": str(array.dtype)}


def _array_hash(array):
    digest = hashlib.sha256(json.dumps(_header(array), sort_keys=True).encode())
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _bundle_hash(fields):
    scientific = [
        {key: value for key, value in row.items() if key != "path"} for row in fields
    ]
    return hashlib.sha256(json.dumps(scientific, sort_keys=True).encode()).hexdigest()


def _numeric(value):
    array = np.asarray(value)
    if array.dtype.kind not in "biufcU" and str(array.dtype) != "bfloat16":
        raise TypeError(
            "Trajectory fields must contain numeric arrays or declarative string labels"
        )
    return array


def save(path, *, max_buffer_bytes=64 * 1024**2, **arrays):
    """Atomically publish an immutable collection; never overwrite a trajectory."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Trajectory bundle already exists: {path}")
    if not arrays or any(not isinstance(name, str) or not name for name in arrays):
        raise ValueError("At least one named numeric trajectory field is required")
    arrays = {name: _numeric(value) for name, value in sorted(arrays.items())}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}-{uuid4().hex}")
    try:
        temporary.mkdir()
        fields = []
        for index, (name, array) in enumerate(arrays.items()):
            relative = f"arrays/{index:06d}"
            row = {
                "name": name,
                "path": relative,
                **_header(array),
                "sha256": _array_hash(array),
            }
            fields.append(row)
            if array.dtype.kind in "U":
                row["labels"] = array.tolist()
                continue
            if array.size == 0:
                continue
            directory = temporary / relative / "samples"
            begin_chains(directory, (0,), metadata={"field": name})
            with PositionWriter(directory, max_buffer_bytes=max_buffer_bytes) as writer:
                values = array if array.ndim else array.reshape(1)
                for draw, value in enumerate(values):
                    writer.append({"value": value}, 0, draw)
            complete_chains(directory, (0,))
        write_json(
            temporary / "trajectory.json",
            {"fields": fields, "sha256": _bundle_hash(fields)},
        )
        if path.exists():
            raise FileExistsError(f"Trajectory bundle already exists: {path}")
        temporary.rename(path)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    return path


def _manifest(path):
    path = Path(path)
    document = json.loads((path / "trajectory.json").read_text())
    fields = document["fields"]
    if not fields or _bundle_hash(fields) != document["sha256"]:
        raise ValueError("Invalid trajectory bundle schema or identity")
    names = [row["name"] for row in fields]
    if names != sorted(set(names)):
        raise ValueError("Trajectory fields must have distinct canonical names")
    for index, row in enumerate(fields):
        if row["path"] != f"arrays/{index:06d}" or any(
            type(size) is not int or size < 0 for size in row["shape"]
        ):
            raise ValueError("Invalid trajectory field location or shape")
        dtype = np.dtype(row["dtype"])
        if dtype.kind not in "biufcU" and str(dtype) != "bfloat16":
            raise ValueError("Invalid trajectory array dtype")
    return document


def _field(path, row):
    shape, dtype = tuple(row["shape"]), np.dtype(row["dtype"])
    if dtype.kind in "U":
        result = np.asarray(row["labels"], dtype=dtype).reshape(shape)
        if _array_hash(result) != row["sha256"]:
            raise ValueError("Trajectory label metadata changed")
        return result
    if 0 in shape:
        return np.empty(shape, dtype=dtype)
    index = PositionIndex.open(Path(path) / row["path"] / "samples", chains=(0,))
    count = shape[0] if shape else 1
    if len(index.draw_ids) != count or tuple(index.draw_ids) != tuple(range(count)):
        raise ValueError("Trajectory field draw identities changed")
    values = index.load()["value"]
    result = values[0].reshape(shape)
    if _header(result) != {"shape": list(shape), "dtype": str(dtype)}:
        raise ValueError("Trajectory field shape or dtype changed")
    if _array_hash(result) != row["sha256"]:
        raise ValueError("Trajectory field payload changed")
    return result


def load(path):
    """Read a completed collection."""
    document = _manifest(path)
    return {row["name"]: _field(path, row) for row in document["fields"]}


def checksum(path):
    """Hash a verified trajectory's numeric content or the bytes of an ordinary file."""
    path = Path(path)
    if path.is_file():
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while block := stream.read(8 * 1024**2):
                digest.update(block)
        return digest.hexdigest()
    document = _manifest(path)
    for row in document["fields"]:
        shape, dtype = tuple(row["shape"]), np.dtype(row["dtype"])
        if dtype.kind in "U":
            _field(path, row)
            continue
        digest = hashlib.sha256(
            json.dumps(
                {"shape": list(shape), "dtype": str(dtype)}, sort_keys=True
            ).encode()
        )
        if 0 not in shape:
            index = PositionIndex.open(path / row["path"] / "samples", chains=(0,))
            count = shape[0] if shape else 1
            if len(index.draw_ids) != count or tuple(index.draw_ids) != tuple(
                range(count)
            ):
                raise ValueError("Trajectory field draw identities changed")
            batch_size = max(
                1, (16 * 1024**2) // max(1, index.schema["bytes_per_draw"])
            )
            for start in range(0, count, batch_size):
                values = index.load(draws=range(start, min(count, start + batch_size)))[
                    "value"
                ]
                if values.dtype != dtype or tuple(values.shape[2:]) != (
                    shape[1:] if shape else ()
                ):
                    raise ValueError("Trajectory field schema changed")
                digest.update(np.ascontiguousarray(values[0]).tobytes())
        if digest.hexdigest() != row["sha256"]:
            raise ValueError("Trajectory field payload changed")
    return document["sha256"]
