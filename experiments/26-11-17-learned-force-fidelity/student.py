"""Shared local arithmetic learners for structured surrogate forces."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def initialize_student(key, *, width=32, dtype=jnp.float32):
    """Two SiLU hidden layers; width 32 has exactly 1,185 parameters."""
    keys = jax.random.split(key, 3)
    dimensions = (2, width, width, 1)
    return tuple(
        (
            jax.random.normal(k, (din, dout), dtype=dtype) / jnp.sqrt(din),
            jnp.zeros(dout, dtype=dtype),
        )
        for k, din, dout in zip(keys, dimensions[:-1], dimensions[1:], strict=True)
    )


def pairwise(params, a, b, *, odd=False):
    """Evaluate the scalar pair map, optionally projected onto separate oddness."""
    a, b = jnp.broadcast_arrays(a, b)

    def mlp(x, y):
        values = jnp.stack((x, y), axis=-1)
        for weight, bias in params[:-1]:
            values = jax.nn.silu(values @ weight + bias)
        weight, bias = params[-1]
        return (values @ weight + bias)[..., 0]

    if odd:
        return (mlp(a, b) - mlp(-a, b) - mlp(a, -b) + mlp(-a, -b)) / 4
    return mlp(a, b)


def local_operation(params, weights, downstream, *, odd=False):
    """Equation (12): normalize neighbors, learn pairs, sum, bound and rescale."""
    # Floor the squared norm before sqrt: differentiating norm(0) first would
    # produce NaNs even when a later maximum selects the positive floor.
    cw = jnp.sqrt(jnp.maximum(jnp.sum(weights**2, axis=-1, keepdims=True), 1e-16))
    cu = jnp.sqrt(jnp.maximum(jnp.sum(downstream**2, axis=-1, keepdims=True), 1e-16))
    value = jnp.sum(pairwise(params, weights / cw, downstream / cu, odd=odd), axis=-1)
    return (cw * cu)[..., 0] * jnp.clip(value, -1.0, 1.0)


def zero_operation(weights, downstream):
    """The zero-output untrained control retains the supplied surrounding rule."""
    return jnp.zeros(
        jnp.broadcast_shapes(weights.shape, downstream.shape)[:-1],
        dtype=jnp.result_type(weights, downstream),
    )


def bilinear_operation(coefficient, weights, downstream):
    """One fitted coefficient and exact bilinear structure, a capacity control."""
    return coefficient * jnp.sum(weights * downstream, axis=-1)
