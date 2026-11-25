"""Temperature selection and resampling preserve their declared probabilities."""

import numpy as np
import pytest
from bnn.samplers.tempered_smc import next_temperature, systematic_resample
from scipy.special import logsumexp


def test_tempering_and_resampling_preserve_the_declared_law():
    ll = np.linspace(-20, 5, 128)
    beta = next_temperature(ll, 0)
    lw = beta * ll
    ess = np.exp(2 * logsumexp(lw) - logsumexp(2 * lw))
    assert ess == pytest.approx(0.8 * len(ll))
    assert next_temperature(np.ones(20), 0) == 1
    weights = np.array([0.05, 0.15, 0.3, 0.5])
    rng = np.random.default_rng(7)
    counts = sum(
        np.bincount(systematic_resample(weights, rng), minlength=4) for _ in range(5000)
    )
    np.testing.assert_allclose(counts / counts.sum(), weights, atol=0.01)
