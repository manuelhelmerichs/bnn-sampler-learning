"""Stored draws come back as selected, in their stored precision."""

from pathlib import Path
from typing import Any

import jax
import numpy as np
from bnn.config import RunConfig
from bnn.pipeline import provenance
from bnn.pipeline.run import Run, create_or_resume
from bnn.pipeline.samples import load_raw
from bnn.store.positions import append_block, begin_chains, complete_chains
from bnn.store.trees import save_array_tree


def write_run(
    root: str | Path,
    draws: Any,
    *,
    draw_ids=None,
    fitted: Any = None,
    config: dict | None = None,
) -> Run:
    """Write a completed run that holds the given draws, without sampling.

    ``draws`` has chain and draw axes on every leaf; its tree is stored as it
    is and need not belong to the configured model. ``fitted`` has a chain
    axis and defaults to the first draw of every chain. Without ``config`` the
    run describes a tiny regression table written next to its folder.
    """
    root = Path(root)
    chains, count = jax.tree.leaves(draws)[0].shape[:2]
    document = dict(config or {})
    if "data" not in document:
        root.parent.mkdir(parents=True, exist_ok=True)
        table = root.parent / f"{root.name}.table.npy"
        rows = np.arange(24, dtype=np.float32).reshape(8, 3)
        np.save(table, rows)
        document["data"] = {
            "path": str(table),
            "task": "mean_regression",
            "normalize": True,
            "train_split": 0.5,
            "valid_split": 0.25,
            "test_split": 0.25,
        }
    document.setdefault("model", {"name": "FCN", "hidden_structure": [2, 1]})
    document.setdefault("chains", chains)
    document.setdefault("sampler", {"name": "mclmc", "warmup": 20, "steps": count})
    document["name"] = root.name
    run = create_or_resume(
        RunConfig.model_validate(document),
        root.parent,
        provenance.code_identity(),
        group_size=1,
    )
    ids = np.arange(count) if draw_ids is None else np.asarray(draw_ids)
    begin_chains(run.directory.samples, range(chains))
    for chain in range(chains):
        append_block(
            run.directory.samples,
            chain,
            ids,
            jax.tree.map(lambda leaf, chain=chain: np.asarray(leaf)[chain], draws),
        )
        position = (
            jax.tree.map(lambda leaf, chain=chain: np.asarray(leaf)[chain, 0], draws)
            if fitted is None
            else jax.tree.map(lambda leaf, chain=chain: np.asarray(leaf)[chain], fitted)
        )
        save_array_tree(run.directory.fitted(chain), {"position": position})
    complete_chains(run.directory.samples, range(chains))
    for stage in ("fit", "sample"):
        run.set_stage(stage, "complete")
    return run


def _tree(value):
    return {
        "fcn": {
            "layer0": {
                "kernel": np.array([[value, 2.0]], dtype=np.float32),
                "bias": np.zeros(2, dtype=np.float32),
            },
            "layer1": {
                "kernel": np.array([[3.0], [1.0]], dtype=np.float32),
                "bias": np.zeros(1, dtype=np.float32),
            },
        }
    }


def _run(root):
    chains = [[_tree(1.0 + 0.01 * d + 0.1 * c) for d in range(4)] for c in range(2)]
    draws = jax.tree.map(
        lambda *leaves: np.stack(leaves).reshape(2, 4, *leaves[0].shape),
        *[tree for chain in chains for tree in chain],
    )
    run = write_run(root, draws, draw_ids=[0, 2, 4, 6])
    run.dataset()
    return root


def test_selection_keeps_stored_ids_precision_and_chain_order(tmp_path):
    root = _run(tmp_path / "run")
    view = load_raw(root, chain_ids=(1, 0), draw_indices=(1, 3))
    assert view.chain_ids == (1, 0)
    assert view.draw_ids == (2, 6)
    assert view.parameters.shape == (2, 2, 7)
    assert view.parameters.dtype == np.float32
    kernel = view.chains[0][1]["fcn"]["layer0"]["kernel"]
    np.testing.assert_allclose(kernel[0, 0], 1.0 + 0.03 + 0.1)
    np.testing.assert_array_equal(
        view.layout.flatten(view.chains[1][0]), view.parameters[1, 0]
    )
