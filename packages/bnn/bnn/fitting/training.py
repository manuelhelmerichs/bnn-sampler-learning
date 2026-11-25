"""The noise scale that the mode fit pairs with every fitted member."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def fitted_log_sigma(model, params, x, y) -> jax.Array:
    """Log observation scale that maximizes the training likelihood of a member."""
    mean = model.apply({"params": params}, x)[..., 0]
    mse = jnp.mean(jnp.square(y - mean))
    # The bounds match the likelihood's scale clamp, also for an exact fit.
    return 0.5 * jnp.log(jnp.clip(mse, 1e-12, 1e12))
