"""Posterior draws of a run as flat vectors and parameter trees."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import jax
import numpy as np

from bnn.pipeline.run import open_run
from bnn.trees import TreeLayout


@dataclass(frozen=True)
class SampleView:
    """Selected chains and draws of a run, stacked as ``(chains, draws, ...)``."""

    run_root: Path
    stacked: Any
    chain_ids: tuple[int, ...]
    draw_ids: tuple[int, ...]
    layout: TreeLayout

    @cached_property
    def parameters(self) -> np.ndarray:
        """Draws as flat vectors of shape ``(chains, draws, parameters)``."""
        return self.layout.flatten_batch(self.stacked)

    @cached_property
    def chains(self) -> tuple[tuple[Any, ...], ...]:
        """Draws as parameter trees, one tuple per chain."""
        n_chains, n_draws = jax.tree.leaves(self.stacked)[0].shape[:2]
        return tuple(
            tuple(
                jax.tree.map(lambda a, c=c, d=d: a[c, d], self.stacked)
                for d in range(n_draws)
            )
            for c in range(n_chains)
        )


def load_raw(run_root: str | Path, *, chain_ids=None, draw_indices=None) -> SampleView:
    """Load stored draws of a completed run, unaligned.

    ``chain_ids`` select chains by their id and ``draw_indices`` select draws by
    their position in storage; both default to everything stored.
    """
    run_root = Path(run_root).resolve()
    index = open_run(run_root).samples()
    chains = tuple(index.chain_ids if chain_ids is None else chain_ids)
    draws = tuple(range(len(index.draw_ids)) if draw_indices is None else draw_indices)
    stacked = index.load([index.chain_ids.index(c) for c in chains], draws)
    stored = index.draw_ids.to_numpy()
    layout = TreeLayout.from_tree(jax.tree.map(lambda a: a[0, 0], stacked))
    return SampleView(
        run_root, stacked, chains, tuple(int(stored[d]) for d in draws), layout
    )
