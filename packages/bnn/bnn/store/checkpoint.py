"""Checkpoints of arbitrary JAX state, restored against a structural template.

State may contain named tuples, dataclasses and typed random keys. Only its
leaves are stored; the code that produced the state supplies its structure as a
template when the state is restored.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax
import numpy as np

from bnn.store.io import atomic_write

_METADATA = "__metadata__"


def _is_key(leaf: Any) -> bool:
    return hasattr(leaf, "dtype") and jax.dtypes.issubdtype(
        leaf.dtype, jax.dtypes.prng_key
    )


def save_state(path: str | Path, state: Any, metadata: dict | None = None) -> None:
    """Atomically store the leaves of ``state`` with JSON metadata."""
    leaves = [
        np.asarray(jax.random.key_data(leaf) if _is_key(leaf) else leaf)
        for leaf in jax.tree.leaves(state)
    ]
    data = {f"leaf_{index:06d}": leaf for index, leaf in enumerate(leaves)}
    document = {
        "leaves": [{"shape": list(a.shape), "dtype": str(a.dtype)} for a in leaves],
        "metadata": metadata or {},
    }
    data[_METADATA] = np.asarray(json.dumps(document, allow_nan=False))
    atomic_write(path, lambda file: np.savez(file, **data))


def load_metadata(path: str | Path) -> dict:
    """Read only the metadata of a checkpoint."""
    with np.load(path, allow_pickle=False) as archive:
        return json.loads(str(archive[_METADATA]))["metadata"]


def load_state(path: str | Path, template: Any) -> tuple[Any, dict]:
    """Restore state into the structure, shapes and dtypes of ``template``.

    Template leaves need a ``shape`` and a ``dtype``; their values are unused.
    """
    template_leaves, definition = jax.tree.flatten(template)
    with np.load(path, allow_pickle=False) as archive:
        document = json.loads(str(archive[_METADATA]))
        if len(document["leaves"]) != len(template_leaves):
            raise ValueError(f"Checkpoint does not match the current program: {path}")
        leaves = []
        for index, expected in enumerate(template_leaves):
            array = archive[f"leaf_{index:06d}"]
            if _is_key(expected):
                leaves.append(jax.random.wrap_key_data(array))
                continue
            if tuple(array.shape) != tuple(expected.shape) or array.dtype != np.dtype(
                expected.dtype
            ):
                raise ValueError(
                    f"Checkpoint leaf {index} has shape {array.shape} and dtype "
                    f"{array.dtype}; the current program expects {tuple(expected.shape)} "
                    f"and {expected.dtype}: {path}"
                )
            leaves.append(array)
    return definition.unflatten(leaves), document["metadata"]
