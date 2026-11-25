"""Self-describing array trees: one compressed file holding structure and leaves.

The structure is declarative JSON (dictionaries, lists, tuples, ``None`` and
array leaves), so loading never executes pickled objects. Shapes and dtypes are
recorded next to the data and checked on every read.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax.core import FrozenDict

from bnn.store.io import atomic_write

_METADATA = "__metadata__"


def encode_structure(value: Any) -> dict:
    """Describe the containers of a tree, replacing every array by a leaf marker."""
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("Stored dictionaries require string keys.")
        return {
            "kind": "frozen_dict" if isinstance(value, FrozenDict) else "dict",
            "items": [[key, encode_structure(value[key])] for key in sorted(value)],
        }
    if isinstance(value, (list, tuple)):
        if hasattr(value, "_fields"):
            raise TypeError("Named tuples are not a declarative container.")
        return {
            "kind": "tuple" if isinstance(value, tuple) else "list",
            "items": [encode_structure(item) for item in value],
        }
    return {"kind": "none" if value is None else "leaf"}


def decode_structure(node: dict, leaves: Iterator[Any]) -> Any:
    """Rebuild a tree from its description, consuming leaves in canonical order."""
    try:
        kind = node["kind"]
        if kind == "leaf":
            return next(leaves)
        if kind == "none":
            return None
        if kind in ("dict", "frozen_dict"):
            keys = [key for key, _ in node["items"]]
            if keys != sorted(set(keys)):
                raise ValueError("Stored mapping keys must be unique and sorted.")
            value = {
                key: decode_structure(child, leaves) for key, child in node["items"]
            }
            return FrozenDict(value) if kind == "frozen_dict" else value
        if kind in ("list", "tuple"):
            items = [decode_structure(child, leaves) for child in node["items"]]
            return tuple(items) if kind == "tuple" else items
    except (KeyError, TypeError, StopIteration) as error:
        raise ValueError("Invalid stored tree structure.") from error
    raise ValueError(f"Unknown stored tree node kind: {kind!r}.")


def structure_of(tree: Any) -> dict:
    """Describe a tree and require that the description reproduces it exactly."""
    leaves, definition = jax.tree.flatten(tree)
    structure = encode_structure(tree)
    restored = decode_structure(structure, iter(range(len(leaves))))
    if jax.tree.structure(restored) != definition:
        raise TypeError(
            "Stored trees consist of dictionaries, lists, tuples and arrays."
        )
    return structure


def is_storable(dtype: np.dtype) -> bool:
    """Numeric dtypes with a stable byte representation, including bfloat16."""
    return not dtype.hasobject and (dtype.kind in "biufc" or str(dtype) == "bfloat16")


def save_array_tree(path: str | Path, tree: Any, metadata: dict | None = None) -> None:
    """Atomically save numeric arrays with their tree, shapes, dtypes and metadata."""
    arrays = [np.asarray(leaf) for leaf in jax.tree.leaves(tree)]
    for array in arrays:
        if not is_storable(array.dtype):
            raise TypeError(f"Unsupported stored array dtype: {array.dtype}.")
    document = {
        "structure": structure_of(tree),
        "leaves": [{"shape": list(a.shape), "dtype": str(a.dtype)} for a in arrays],
        "metadata": {} if metadata is None else metadata,
    }
    data = {f"leaf_{index:06d}": array for index, array in enumerate(arrays)}
    data[_METADATA] = np.asarray(json.dumps(document, allow_nan=False))
    atomic_write(path, lambda file: np.savez_compressed(file, **data))


def load_array_tree(path: str | Path, *, on_host: bool = True) -> tuple[Any, dict]:
    """Load a tree and its metadata, checking every declared shape and dtype."""
    with np.load(path, allow_pickle=False) as archive:
        if _METADATA not in archive:
            raise ValueError(f"Not an array tree: {path}")
        document = json.loads(str(archive[_METADATA]))
        names = [f"leaf_{index:06d}" for index in range(len(document["leaves"]))]
        if set(archive.files) != {*names, _METADATA}:
            raise ValueError(f"Array tree keys do not match its schema: {path}")
        leaves = []
        for name, spec in zip(names, document["leaves"], strict=True):
            array = archive[name]
            if spec["dtype"] == "bfloat16" and array.dtype.kind == "V":
                array = array.view(jnp.bfloat16)
            if list(array.shape) != spec["shape"] or str(array.dtype) != spec["dtype"]:
                raise ValueError(f"Array {name} does not match its schema: {path}")
            leaves.append(array)
    tree = decode_structure(document["structure"], iter(leaves))
    if jax.tree.structure(tree).num_leaves != len(leaves):
        raise ValueError(f"Array tree structure does not match its leaves: {path}")
    return (tree if on_host else to_device(tree)), document["metadata"]


def to_device(tree: Any, device: Any = None) -> Any:
    """Transfer a tree to JAX, refusing to silently reduce its stored precision."""

    def transfer(leaf):
        array = np.asarray(leaf)
        dtype = jax.dtypes.canonicalize_dtype(array.dtype)
        if dtype != array.dtype:
            raise ValueError(
                f"Loading would change dtype {array.dtype} to {dtype}; enable JAX "
                "x64 or keep the array on the host."
            )
        return jax.device_put(array, device)

    return jax.tree.map(transfer, tree)


def signature(tree: Any) -> list[dict]:
    """Shapes and dtypes of all leaves, in canonical order."""
    return [
        {"shape": list(np.shape(leaf)), "dtype": str(np.asarray(leaf).dtype)}
        for leaf in jax.tree.leaves(tree)
    ]
