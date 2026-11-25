"""Posterior draws and array trees survive storage bit for bit."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from bnn.store import (
    PositionIndex,
    PositionWriter,
    append_block,
    begin_chains,
    complete_chains,
    head,
    load_array_tree,
    load_state,
    restore_head,
    save_array_tree,
    save_state,
    truncate,
)
from flax.core import FrozenDict


def draws(count: int, offset: float = 0.0) -> dict:
    values = np.arange(count, dtype=np.float32)[:, None] + offset
    return {
        "params": {"kernel": np.tile(values[..., None], (1, 2, 3)), "bias": values},
        "sigma": np.arange(count, dtype=np.float64),
        "flag": np.arange(count) % 2 == 0,
        "half": np.arange(count).astype(jnp.bfloat16),
    }


def test_blocks_round_trip_exactly_and_select_by_ordinal(tmp_path):
    begin_chains(tmp_path, [0, 4])
    for chain, offset in ((0, 0.0), (4, 100.0)):
        append_block(tmp_path, chain, np.arange(0, 10, 2), draws(5, offset))
        append_block(
            tmp_path,
            chain,
            [10, 13, 17],
            jax.tree.map(lambda a: a[:3], draws(5, offset + 5)),
        )
    complete_chains(tmp_path, [0, 4])

    index = PositionIndex.open(tmp_path)
    assert index.chain_ids == (0, 4)
    assert list(index.draw_ids) == [0, 2, 4, 6, 8, 10, 13, 17]
    everything = index.load()
    assert everything["params"]["kernel"].shape == (2, 8, 2, 3)
    assert everything["sigma"].dtype == np.float64 and everything["flag"].dtype == bool
    assert str(everything["half"].dtype) == "bfloat16"
    np.testing.assert_array_equal(
        everything["params"]["bias"][1, :, 0], [100, 101, 102, 103, 104, 105, 106, 107]
    )

    selected = index.load(chains=[1], draws=[7, 0, 5])
    np.testing.assert_array_equal(selected["params"]["bias"][0, :, 0], [107, 100, 105])
    batches = list(index.iterate(0, batch_size=3))
    assert [len(batch["sigma"]) for batch in batches] == [3, 3, 2]


def test_a_chain_returns_to_an_earlier_committed_prefix(tmp_path):
    first = append_block(tmp_path, 0, [0, 1], draws(2))
    second = append_block(tmp_path, 0, [2, 3, 4], draws(3))
    append_block(tmp_path, 0, [5], draws(1))
    restore_head(tmp_path, 0, second)
    assert head(tmp_path, 0) == second
    assert len(list((tmp_path / "chains" / "0" / "blocks").iterdir())) == 2
    truncate(tmp_path, 0, 2)
    assert head(tmp_path, 0) == first
    truncate(tmp_path, 0, 0)
    assert head(tmp_path, 0)["count"] == 0
    append_block(tmp_path, 0, [0], draws(1))


def test_buffered_writer_commits_in_bounded_blocks(tmp_path):
    begin_chains(tmp_path, [0])
    with PositionWriter(tmp_path, max_buffer_bytes=2000) as writer:
        for draw in range(20):
            writer.append({"w": np.full((16,), draw, np.float32)}, 0, draw)
    complete_chains(tmp_path, [0])
    index = PositionIndex.open(tmp_path)
    assert len(index.blocks[0]) > 1
    np.testing.assert_array_equal(index.load()["w"][0, :, 0], np.arange(20))


def test_array_trees_keep_structure_dtypes_and_metadata(tmp_path):
    tree = {
        "a": (np.arange(3, dtype=np.int16), [np.float64(1.5), None]),
        "b": FrozenDict({"c": np.ones((2, 2), np.float32)}),
    }
    save_array_tree(tmp_path / "tree.npz", tree, metadata={"note": "kept"})
    loaded, metadata = load_array_tree(tmp_path / "tree.npz")
    assert metadata == {"note": "kept"}
    assert isinstance(loaded["a"], tuple) and loaded["a"][1][1] is None
    assert isinstance(loaded["b"], FrozenDict)
    assert loaded["a"][0].dtype == np.int16 and loaded["a"][1][0].dtype == np.float64


def test_state_checkpoints_restore_named_tuples_and_keys(tmp_path):
    from typing import NamedTuple

    class State(NamedTuple):
        position: dict
        key: jax.Array
        step: jax.Array

    state = State({"w": jnp.arange(4.0)}, jax.random.key(3), jnp.asarray(7))
    save_state(tmp_path / "state.npz", state, {"phase": 1})
    template = jax.eval_shape(lambda: state)
    restored, metadata = load_state(tmp_path / "state.npz", template)
    assert metadata == {"phase": 1} and isinstance(restored, State)
    np.testing.assert_array_equal(restored.position["w"], state.position["w"])
    np.testing.assert_array_equal(
        jax.random.key_data(restored.key), jax.random.key_data(state.key)
    )
