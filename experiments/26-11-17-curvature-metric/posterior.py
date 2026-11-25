"""A campaign's posterior on flat float64 vectors, and metrics as factors.

A metric ``M`` is used through a factor ``W`` with ``W W' = M^-1``: sampling
``z`` with ``theta = center + W z`` and no metric is sampling ``theta`` under
``M``. A diagonal factor is stored as its diagonal.
"""

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg


def posterior(model, layout, x, y):
    """Potential ``U = -log posterior``, local metric and network mean.

    ``model`` is the probabilistic model of a mean-regression run. The local
    metric is the Fisher information of the Gaussian likelihood plus the prior
    precision: ``J'J / sigma^2`` on the weights and ``2N`` on the log noise
    scale, plus the identity of the standard normal prior.
    """
    x, y = jnp.asarray(x, jnp.float64), jnp.asarray(y, jnp.float64)
    dimension = layout.dimension
    scale = int(np.asarray(layout.unflatten(np.arange(dimension))["sigma"]))

    def prior(flat):
        return model.log_prior(layout.unflatten_traced(flat))

    def potential(flat):
        return -model.log_density(layout.unflatten_traced(flat), x, y)

    def mean(flat, inputs):
        return model.apply(layout.unflatten_traced(flat), inputs)[..., 0]

    def metric(flat):
        jacobian = jax.vmap(
            jax.grad(lambda flat, row: mean(flat, row[None])[0]), in_axes=(None, 0)
        )(flat, x)
        fisher = jacobian.T @ jacobian / jnp.exp(2 * flat[scale])
        return fisher.at[scale, scale].add(2.0 * len(y)) + jnp.eye(dimension)

    precision = -jax.hessian(prior)(jnp.zeros(dimension))
    if not np.allclose(precision, np.eye(dimension), atol=1e-8):
        raise ValueError("The Fisher metric here assumes a standard normal prior")
    return {
        "dimension": dimension,
        "observations": len(y),
        "scale": scale,
        "blocks": layer_blocks(layout),
        "potential": potential,
        "mean": mean,
        "metric": jax.jit(metric),
    }


def apply(factor, z):
    """``W z`` for a diagonal (1-D) or dense factor."""
    return factor * z if factor.ndim == 1 else factor @ z


def dense_factor(metric):
    """``W = L^-T`` for ``M = L L'``."""
    return scipy.linalg.solve_triangular(
        np.linalg.cholesky(metric), np.eye(len(metric)), lower=True
    ).T


def diagonal_factor(metric):
    return 1 / np.sqrt(np.diag(metric))


def layer_blocks(layout):
    """Flat indices of each layer's parameters; any other leaf is its own block."""
    blocks = []

    def visit(node):
        leaves = [value for value in node.values() if not isinstance(value, dict)]
        children = [value for value in node.values() if isinstance(value, dict)]
        indices = [np.asarray(leaf).astype(int).ravel() for leaf in leaves]
        if children:
            blocks.extend(indices)
        elif indices:
            blocks.append(np.concatenate(indices))
        for child in children:
            visit(child)

    visit(layout.unflatten(np.arange(layout.dimension)))
    if sorted(np.concatenate(blocks)) != list(range(layout.dimension)):
        raise ValueError("Layer blocks must partition the parameters")
    return blocks


def block_factor(metric, blocks):
    """Dense factor of the metric with every entry between two blocks dropped."""
    factor = np.zeros_like(metric)
    for block in blocks:
        part = np.ix_(block, block)
        factor[part] = dense_factor(metric[part])
    return factor


def low_rank_factor(metric, rank):
    """Factor of the diagonal plus the leading ``rank`` directions of the metric.

    The metric is scaled to unit diagonal. Its ``rank`` stiffest eigendirections
    are kept and the others receive their mean eigenvalue.
    """
    scale = diagonal_factor(metric)
    values, vectors = np.linalg.eigh(metric * np.outer(scale, scale))
    if rank >= len(metric):
        return scale[:, None] * (vectors / np.sqrt(values)) @ vectors.T
    top, kept = vectors[:, -rank:], values[-rank:]
    rest = np.eye(len(metric)) - top @ top.T
    inverse_root = rest / np.sqrt(values[:-rank].mean()) + (top / np.sqrt(kept)) @ top.T
    return scale[:, None] * inverse_root


def spread_factor(variances, count):
    """Diagonal factor from the variances of ``count`` draws, shrunk as Stan does."""
    return np.sqrt((count * variances + 5e-3) / (count + 5))


def mean_metric(local, points):
    """Mean of the local metrics at ``points``."""
    total = 0.0
    for point in points:
        total = total + np.asarray(local(jnp.asarray(point, jnp.float64)))
    return total / len(points)
