"""Constant metrics estimated from a chain's own draws.

Both return a factor ``W`` per target, with ``W W^T`` the estimated inverse
metric, from arrays whose first axis is the target.
"""

import numpy as np

# Eigenvalues of a mean Hessian below this share of the largest are raised to it.
FLOOR = 1e-10


def covariance_factor(draws):
    """Cholesky factor of the covariance of ``(target, draw, D)``, shrunk as Stan does."""
    count, d = draws.shape[1:]
    centered = draws - draws.mean(1, keepdims=True)
    covariance = np.einsum("tni,tnj->tij", centered, centered) / (count - 1)
    shrunk = count / (count + 5) * covariance + 1e-3 * 5 / (count + 5) * np.eye(d)
    return np.linalg.cholesky(shrunk)


def curvature_factor(hessians):
    """Inverse root of the mean of ``(target, state, D, D)`` Hessians, made positive."""
    values, vectors = np.linalg.eigh(hessians.mean(1))
    values = np.abs(values)
    values = np.maximum(values, FLOOR * values.max(-1, keepdims=True))
    return vectors / np.sqrt(values)[:, None, :]
