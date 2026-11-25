"""The phase engine: results do not depend on blocks, pauses or crashes."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from bnn.engine import Destinations, Pacing, Phase, Program, run_program
from bnn.store.positions import PositionIndex, append_block, head

CHAINS = [3, 5]
STEPS = 103


class Carry(NamedTuple):
    position: dict
    momentum: jax.Array
    scale: jax.Array


def program() -> Program:
    """Two phases that use keys, per-chain inputs, shared inputs and constants."""

    def step(carry, x, constants):
        noise = jax.random.normal(x["key"], (3,))
        momentum = 0.9 * carry.momentum + 0.1 * noise * x["rate"]
        w = carry.position["w"] + carry.scale * momentum + constants * x["batch"].sum()
        position = {"w": w, "count": carry.position["count"] + 1}
        return Carry(position, momentum, carry.scale * 1.001), {
            "norm": jnp.linalg.norm(w),
            "far": jnp.linalg.norm(w) > 1,
        }

    keep = np.arange(STEPS) % 3 == 0
    keep[10:30] = False
    shared = {
        "step": step,
        "position": lambda carry: carry.position,
        "chain_inputs": lambda chains, steps: {
            "batch": (
                chains[:, None, None] + steps[None, :, None] + np.arange(2)
            ).astype(np.int32)
        },
        "shared_inputs": lambda steps: {"rate": (1.0 / (1 + steps)).astype(np.float32)},
    }
    return Program(
        (
            Phase(
                "warmup",
                40,
                lambda initial, key, constants: Carry(
                    initial, jax.random.normal(key, (3,)), jnp.ones(())
                ),
                finish=lambda carry, constants: {
                    "position": carry.position,
                    "scale": carry.scale * 2,
                },
                **shared,
            ),
            Phase(
                "sample",
                STEPS,
                lambda previous, key, constants: Carry(
                    previous["position"], jnp.zeros(3), previous["scale"]
                ),
                keep=keep,
                store="samples",
                must_move=True,
                **shared,
            ),
        ),
        constants=jnp.asarray(1e-3),
    )


def run(root: Path, pacing: Pacing, stop=lambda: False):
    destinations = Destinations(
        root / "state", {"samples": root / "samples"}, root / "diagnostics"
    )
    initial = {"w": jnp.zeros((2, 3)), "count": jnp.zeros((2,), jnp.int32)}
    return run_program(
        program(),
        chain_ids=CHAINS,
        initial=initial,
        key=jax.random.key(7),
        destinations=destinations,
        pacing=pacing,
        should_stop=stop,
    )


def results(root: Path):
    draws = PositionIndex.open(root / "samples")
    diagnostics = PositionIndex.open(root / "diagnostics" / "sample")
    return draws.load(), draws.draw_ids.to_numpy(), diagnostics.load()


def assert_same(left, right):
    for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    root = tmp_path_factory.mktemp("reference")
    completed, final = run(root, Pacing(block_steps=7))
    assert completed
    return results(root), final


def test_draws_are_kept_at_the_declared_steps(reference):
    (draws, ids, diagnostics), _ = reference
    expected = [s for s in range(STEPS) if s % 3 == 0 and not 10 <= s < 30]
    assert ids.tolist() == expected
    assert draws["w"].shape == (2, len(expected), 3)
    np.testing.assert_array_equal(draws["count"][0], 40 + np.asarray(expected) + 1)
    assert diagnostics["norm"].shape == (2, STEPS)


@pytest.mark.parametrize("pacing", [Pacing(block_steps=64), Pacing(interval=0.002)])
def test_block_partition_does_not_change_results(tmp_path, reference, pacing):
    completed, final = run(tmp_path, pacing)
    assert completed
    assert_same(results(tmp_path), reference[0])
    assert_same(final, reference[1])


def test_repeated_pauses_continue_exactly(tmp_path, reference):
    pauses = 0
    while True:
        calls = iter(range(100))
        completed, final = run(
            tmp_path, Pacing(block_steps=5), stop=lambda calls=calls: next(calls) >= 2
        )
        if completed:
            break
        pauses += 1
    assert pauses > 5
    assert_same(results(tmp_path), reference[0])
    assert_same(final, reference[1])


def test_draws_committed_after_the_last_checkpoint_are_rolled_back(tmp_path, reference):
    calls = iter(range(100))
    completed, _ = run(tmp_path, Pacing(block_steps=9), stop=lambda: next(calls) >= 8)
    assert not completed
    # A crash between storing draws and committing the state leaves extra draws.
    last = head(tmp_path / "samples", 3)["last_draw"]
    append_block(
        tmp_path / "samples",
        3,
        [last + 1000],
        {"w": np.ones((1, 3), np.float32), "count": np.ones((1,), np.int32)},
    )
    completed, _ = run(tmp_path, Pacing(block_steps=9))
    assert completed
    assert_same(results(tmp_path), reference[0])


def test_nonfinite_state_fails_without_storing_the_block(tmp_path):
    def step(carry, x, constants):
        return jnp.where(x["index"] == 12, jnp.nan, carry + 1.0), {}

    phase = Phase(
        "sample",
        20,
        lambda initial, key, constants: initial,
        step,
        position=lambda carry: carry,
        keep=np.ones(20, dtype=bool),
        store="samples",
    )
    destinations = Destinations(tmp_path / "state", {"samples": tmp_path / "samples"})
    with pytest.raises(FloatingPointError, match="chains \\[0\\] within steps 8 to 15"):
        run_program(
            Program((phase,)),
            chain_ids=[0],
            initial=jnp.zeros((1,)),
            key=jax.random.key(0),
            destinations=destinations,
            pacing=Pacing(block_steps=8),
        )
    assert head(tmp_path / "samples", 0)["count"] == 8
