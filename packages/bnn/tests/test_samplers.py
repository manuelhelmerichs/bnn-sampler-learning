"""Sampler kernels and programs on Gaussian targets with known stationary laws."""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from bnn.engine import simulate
from bnn.samplers.mclmc import MCLMCConfig

MEAN = np.array([0.75, -0.5], dtype=np.float32)
PRECISION = np.array([[2.0, -0.5], [-0.5, 1.0]], dtype=np.float32)
COVARIANCE = np.linalg.inv(PRECISION)


def adapted_program(config, chains=32):
    """Run the phases of a full-data sampler as a run does, without a run.

    Returns the draws of all chains and what the adaptation hands to sampling.
    """

    def log_density(position, constants):
        delta = position["w"] - MEAN
        return -0.5 * jnp.sum(delta * (delta @ PRECISION))

    *adaptation, sample = config.phases(SimpleNamespace(log_density=log_density))

    def chain(key):
        adapt_key, sample_key = jax.random.split(key)
        adapted = simulate(adaptation, {"w": jnp.zeros(2)}, adapt_key)

        def body(carry, index):
            x = {"index": index, "key": jax.random.fold_in(sample_key, index)}
            carry, _ = sample.step(carry, x, None)
            return carry, sample.position(carry)["w"]

        start = sample.start(adapted, sample_key, None)
        return jax.lax.scan(body, start, jnp.arange(sample.steps))[1], adapted

    return jax.jit(jax.vmap(chain))(jax.random.split(jax.random.key(5), chains))


@pytest.mark.parametrize(
    "config,adapted_variances",
    [
        (MCLMCConfig(warmup=400, steps=4000), None),
        (
            MCLMCConfig(warmup=400, steps=4000, diagonal_preconditioning=True),
            "sqrt_diag_cov",
        ),
    ],
    ids=["mclmc", "preconditioned-mclmc"],
)
def test_adapted_programs_recover_correlated_gaussian(config, adapted_variances):
    draws, adapted = adapted_program(config)
    samples = np.asarray(draws).reshape(-1, 2)
    np.testing.assert_allclose(samples.mean(axis=0), MEAN, atol=0.06, rtol=0)
    np.testing.assert_allclose(np.cov(samples.T), COVARIANCE, atol=0.06, rtol=0)
    if adapted_variances is not None:
        # The diagonal metric that the adaptation hands to the sampling phase.
        metric = np.asarray(config.parameters(adapted)[adapted_variances])
        if adapted_variances == "sqrt_diag_cov":
            metric = metric**2
        np.testing.assert_allclose(
            metric.mean(axis=0), np.diag(COVARIANCE), atol=0, rtol=0.3
        )
