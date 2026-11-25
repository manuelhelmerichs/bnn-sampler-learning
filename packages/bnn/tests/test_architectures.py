"""Architectural invariants of the networks."""

import jax
import jax.numpy as jnp
import numpy as np


def test_fcn_preserves_supported_float64_computation():
    from bnn.models.tabular import FCN, FCNConfig

    module = FCN(FCNConfig(hidden_structure=[1]))
    parameters = {
        "params": {"fcn": {"layer0": {"kernel": np.ones((1, 1)), "bias": np.zeros(1)}}}
    }
    with jax.enable_x64():
        inputs = jnp.asarray([[1.0], [1.0 + 1e-10]], dtype=jnp.float64)
        output = module.apply(parameters, inputs, train=False)
        assert output.dtype == jnp.float64
        np.testing.assert_allclose(
            np.diff(output[:, 0]), [1e-10], rtol=1e-6, atol=1e-15
        )
