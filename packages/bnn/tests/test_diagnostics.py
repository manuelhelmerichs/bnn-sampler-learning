"""R-hat and ESS, each with one definition."""

import json

import numpy as np
import pytest
from bnn.evaluation.diagnostics import (
    bulk_tail_ess,
    chain_diagnostics,
    multichain_ess,
    rank_normalized_split_rhat,
    split_rhat,
)


def test_batched_diagnostics_preserve_large_offset_float32_variance():
    # All 513 coordinates have a known variance and require multiple scratch batches.
    base = np.arange(40, dtype=np.float32) + 1_000_000
    values = np.broadcast_to(
        base[None, :, None] + np.arange(4, dtype=np.float32)[:, None, None],
        (4, 40, 513),
    )
    within = 40 * 41 / 12
    between = 40 * 4 * 5 / 12
    expected = np.sqrt(((40 - 1) * within + between) / (40 * within))
    np.testing.assert_allclose(split_rhat(values, split=False), expected, rtol=1e-12)
    diagnostics = rank_normalized_split_rhat(values)
    for result in diagnostics.values():
        np.testing.assert_array_equal(result, np.repeat(result[0], 513))


def test_split_rhat_near_one_for_stationary_chains():
    rng = np.random.default_rng(0)
    draws = rng.standard_normal((4, 40, 3))

    rhat = split_rhat(draws)

    assert rhat.shape == (3,)
    np.testing.assert_allclose(rhat, 1.0, atol=0.1)


def test_split_rhat_detects_separated_chains_and_ignores_constants():
    rng = np.random.default_rng(1)
    draws = rng.standard_normal((4, 40, 2))
    draws[2:] += 50.0

    assert np.all(split_rhat(draws) > 5.0)
    np.testing.assert_array_equal(split_rhat(np.zeros((2, 4, 2))), 1.0)


def test_rank_normalized_split_rhat_detects_location_and_scale_failures():
    rng = np.random.default_rng(7)
    stationary = rng.standard_normal((4, 200, 2))
    location = stationary.copy()
    location[2:, :, 0] += 3.0
    scale = stationary.copy()
    scale[2:, :, 1] *= 4.0

    stationary_rhat = rank_normalized_split_rhat(stationary)
    location_rhat = rank_normalized_split_rhat(location)
    scale_rhat = rank_normalized_split_rhat(scale)

    assert set(stationary_rhat) == {"bulk", "folded", "max"}
    np.testing.assert_allclose(stationary_rhat["max"], 1.0, atol=0.05)
    assert location_rhat["bulk"][0] > 1.5
    assert scale_rhat["folded"][1] > 1.2
    assert scale_rhat["folded"][1] > scale_rhat["bulk"][1]
    np.testing.assert_array_equal(
        scale_rhat["max"],
        np.maximum(scale_rhat["bulk"], scale_rhat["folded"]),
    )


def test_ess_pools_chains_so_disagreeing_chains_have_few_effective_draws():
    rng = np.random.default_rng(3)
    draws = rng.standard_normal((4, 500, 2))
    draws[:, :, 1] += np.array([-3.0, -1.0, 1.0, 3.0])[:, None]
    ess = bulk_tail_ess(draws)
    assert 1500 < ess["bulk"][0] <= 2000
    assert ess["bulk"][1] < 100


@pytest.mark.parametrize("stalled", ["all", "separated", "one_chain"])
def test_stalled_coordinates_have_zero_ess_without_changing_healthy_ones(stalled):
    values = np.random.default_rng(41).normal(size=(4, 256, 2))
    if stalled == "one_chain":
        values[0, :, 0] = 0
    else:
        values[:, :, 0] = 0 if stalled == "all" else np.arange(4, dtype=float)[:, None]
    with np.errstate(divide="raise", invalid="raise"):
        ess = multichain_ess(values)
        diagnostics = chain_diagnostics(values)
        healthy = chain_diagnostics(values[:, :, 1:])
    assert ess[0] == 0
    assert 0 < ess[1] <= 1024
    for name in ("bulk_ess", "tail_ess"):
        assert diagnostics[name][0] == diagnostics[f"{name}_min"] == 0
        assert diagnostics[name][1] == healthy[name][0] > 0


def test_constant_population_diagnostics_are_finite_and_not_effective_samples():
    with np.errstate(divide="raise", invalid="raise"):
        diagnostics = chain_diagnostics(np.zeros((4, 32, 2)))
    assert diagnostics["bulk_ess"] == diagnostics["tail_ess"] == [0.0, 0.0]
    json.dumps(diagnostics, allow_nan=False)
