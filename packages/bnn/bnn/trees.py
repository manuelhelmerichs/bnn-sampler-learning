"""Parameter-tree types, the tree operations every layer shares and the flat layout."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax.core import frozen_dict

type ParamTree = Any
type PRNGKey = jax.Array


def split_position(position: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Split a sampler position ``{"params": ..., **auxiliary}`` into its parts.

    Auxiliary leaves are sampled alongside the network weights, such as the log
    observation scale ``sigma`` of mean regression.
    """
    auxiliary = {key: value for key, value in position.items() if key != "params"}
    return position["params"], auxiliary


def split_key_by_tree(key: PRNGKey, tree: ParamTree) -> ParamTree:
    """Create one independent random key for each array leaf."""
    structure = jax.tree.structure(tree)
    return structure.unflatten(list(jax.random.split(key, structure.num_leaves)))


def tree_stack(trees: Sequence[ParamTree]) -> ParamTree:
    """Stack identically structured trees along a new leading axis on the host."""
    return jax.tree.map(lambda *leaves: np.stack(leaves), *trees)


def tree_index(tree: ParamTree, index: Any) -> ParamTree:
    """Index every leaf along its leading axes."""
    return jax.tree.map(lambda leaf: leaf[index], tree)


def tree_size(tree: ParamTree) -> int:
    """Number of scalar entries in a tree."""
    return sum(int(np.prod(np.shape(leaf))) for leaf in jax.tree.leaves(tree))


def tree_all_finite(tree: ParamTree, batch_dims: int = 0) -> jax.Array:
    """Whether every entry is finite, reduced over all but ``batch_dims`` axes."""
    leaves = [
        leaf
        for leaf in jax.tree.leaves(tree)
        if jnp.issubdtype(jnp.asarray(leaf).dtype, jnp.inexact)
    ]
    if not leaves:
        return jnp.ones((), dtype=bool)
    shape = jnp.shape(leaves[0])[:batch_dims]
    result = jnp.ones(shape, dtype=bool)
    for leaf in leaves:
        leaf = jnp.asarray(leaf)
        result &= jnp.all(jnp.isfinite(leaf).reshape((*shape, -1)), axis=-1)
    return result


def replace_paths(
    params: Mapping[str, Any],
    replacements: Sequence[tuple[tuple[str, ...], Any]],
) -> Mapping[str, Any]:
    """Path-copy mappings and replace leaves without copying untouched arrays."""

    root: dict[str, Any] = dict(params)
    copied: dict[tuple[str, ...], dict[str, Any]] = {(): root}
    for path, value in replacements:
        if not path:
            raise ValueError("Cannot replace the root parameter mapping.")
        source: Mapping[str, Any] = params
        prefix: tuple[str, ...] = ()
        target = root
        for key in path[:-1]:
            prefix = (*prefix, key)
            source_child = source[key]
            if not isinstance(source_child, Mapping):
                raise KeyError(
                    f"Parameter path {path!r} descends through non-mapping {key!r}."
                )
            child = copied.get(prefix)
            if child is None:
                child = dict(source_child)
                copied[prefix] = child
                target[key] = child
            target = child
            source = source_child
        target[path[-1]] = value
    return (
        frozen_dict.freeze(root) if isinstance(params, frozen_dict.FrozenDict) else root
    )


@dataclass(frozen=True)
class TreeLayout:
    """The single bijection between parameter trees and flat vectors.

    ``flatten``/``unflatten`` move host arrays and restore the stored leaf dtypes.
    ``unflatten_traced`` keeps the vector's dtype so it can run inside ``jax.jit``
    and ``jax.grad``.
    """

    treedef: Any
    shapes: tuple[tuple[int, ...], ...]
    dtypes: tuple[np.dtype, ...]
    sizes: tuple[int, ...]

    @classmethod
    def from_tree(cls, tree: Any) -> TreeLayout:
        leaves, treedef = jax.tree_util.tree_flatten(tree)
        arrays = [np.asarray(leaf) for leaf in leaves]
        return cls(
            treedef=treedef,
            shapes=tuple(array.shape for array in arrays),
            dtypes=tuple(array.dtype for array in arrays),
            sizes=tuple(array.size for array in arrays),
        )

    @property
    def dimension(self) -> int:
        return int(sum(self.sizes))

    def flatten(self, tree: Any) -> np.ndarray:
        """Concatenate leaves without promoting stored floating-point precision."""
        return np.concatenate(
            [np.asarray(leaf).reshape(-1) for leaf in jax.tree.leaves(tree)]
        )

    def flatten_batch(self, stacked: Any) -> np.ndarray:
        """Flatten trailing parameter axes with one allocation for the batch."""
        leaves = [np.asarray(leaf) for leaf in jax.tree.leaves(stacked)]
        leading = leaves[0].shape[: leaves[0].ndim - len(self.shapes[0])]
        result = np.empty(
            (*leading, self.dimension), dtype=np.result_type(*self.dtypes)
        )
        start = 0
        for leaf, size in zip(leaves, self.sizes, strict=True):
            result[..., start : start + size] = leaf.reshape(*leading, size)
            start += size
        return result

    def unflatten(self, vector: np.ndarray) -> Any:
        leaves = []
        start = 0
        for size, shape, dtype in zip(
            self.sizes, self.shapes, self.dtypes, strict=True
        ):
            leaves.append(vector[start : start + size].reshape(shape).astype(dtype))
            start += size
        if start != vector.size:
            raise ValueError(f"Expected {start} parameters, got {vector.size}.")
        return jax.tree_util.tree_unflatten(self.treedef, leaves)

    def unflatten_traced(self, vector: jax.Array) -> Any:
        parts = jnp.split(vector, np.cumsum(self.sizes)[:-1])
        leaves = [
            part.reshape(shape) for part, shape in zip(parts, self.shapes, strict=True)
        ]
        return jax.tree_util.tree_unflatten(self.treedef, leaves)

    def stack(self, flat_batch: np.ndarray) -> Any:
        """Stacked tree with a leading batch axis, for vmapped evaluation."""
        leaves = []
        start = 0
        for size, shape in zip(self.sizes, self.shapes, strict=True):
            block = flat_batch[:, start : start + size]
            leaves.append(jnp.asarray(block.reshape((len(flat_batch), *shape))))
            start += size
        return jax.tree_util.tree_unflatten(self.treedef, leaves)
