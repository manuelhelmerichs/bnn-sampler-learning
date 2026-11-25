"""Local curvature of a UCI BNN posterior, and what a metric recovers of it.

A metric is given by a factor ``W`` with ``W W' = M^-1``. Under it the local
curvature ``G`` becomes ``W' G W``; the exact local metric makes that the
identity.
"""

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg


def target(model, layout, x, y):
    """Jitted pieces of the potential U = -log posterior on flat float64 vectors.

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

    def mean(flat, row):
        return model.apply(layout.unflatten_traced(flat), row[None])[0, 0]

    def metric(flat):
        jacobian = jax.vmap(jax.grad(mean), in_axes=(None, 0))(flat, x)
        fisher = jacobian.T @ jacobian / jnp.exp(2 * flat[scale])
        return fisher.at[scale, scale].add(2.0 * len(y)) + jnp.eye(dimension)

    def hessian(flat):
        gradient = jax.grad(potential)
        return jax.lax.map(
            lambda tangent: jax.jvp(gradient, (flat,), (tangent,))[1],
            jnp.eye(dimension),
            batch_size=64,
        )

    def reverse(point, step):
        local = metric(point)
        return jnp.linalg.slogdet(local)[1], step @ local @ step

    precision = -jax.hessian(prior)(jnp.zeros(dimension))
    if not np.allclose(precision, np.eye(dimension), atol=1e-8):
        raise ValueError("The Fisher metric here assumes a standard normal prior")
    return {
        "dimension": dimension,
        "observations": len(y),
        "potential": jax.jit(jax.vmap(potential)),
        "gradient": jax.jit(jax.value_and_grad(potential)),
        "metric": jax.jit(metric),
        "hessian": jax.jit(hessian),
        # Log determinant of the metric at each point and its quadratic form of
        # the step that led there: the reverse density of a local proposal.
        "reverse": jax.jit(
            lambda points, steps: jax.lax.map(
                lambda pair: reverse(*pair), (points, steps)
            )
        ),
    }


def eigensystem(matrix):
    """Eigenvalues and eigenvectors of a symmetric matrix, largest first."""
    values, vectors = np.linalg.eigh(matrix)
    return values[::-1], vectors[:, ::-1]


def local_factor(values, vectors):
    """Factor of the exact local metric from its eigensystem."""
    return vectors / np.sqrt(values)


def constant_factors(metrics, covariance):
    """Factors of the constant metrics a chain could adapt.

    ``metrics`` are local metrics at reference draws of the chain, in order,
    and ``covariance`` is the covariance of all its draws. ``mean_early``
    averages the first quarter of the reference draws only.
    """
    mean = np.mean(metrics, axis=0)
    early = np.mean(metrics[: max(1, len(metrics) // 4)], axis=0)
    dimension = len(mean)

    def inverse_factor(metric):
        return scipy.linalg.solve_triangular(
            np.linalg.cholesky(metric), np.eye(dimension), lower=True
        ).T

    # A chain's covariance can be numerically singular; keep its factor finite.
    values, vectors = np.linalg.eigh(covariance)
    values = np.maximum(values, 1e-12 * values[-1])
    return {
        "identity": np.eye(dimension),
        "mean_diagonal": np.diag(1 / np.sqrt(np.diag(mean))),
        "mean": inverse_factor(mean),
        "mean_early": inverse_factor(early),
        "covariance_diagonal": np.diag(np.sqrt(np.diag(covariance))),
        "covariance": vectors * np.sqrt(values),
    }


def spread(curvature, factor):
    """Condition number and 90%-to-10% quantile ratio of ``factor' curvature factor``."""
    values = np.linalg.eigvalsh(factor.T @ curvature @ factor)
    low, high = np.quantile(values, [0.1, 0.9])
    return values[-1] / values[0], high / low


def overlap(first, second, rank):
    """Shared fraction of two leading eigenspaces; ``rank / dimension`` if unrelated."""
    return float(np.sum((first[:, :rank].T @ second[:, :rank]) ** 2) / rank)


def spectrum(values):
    """Summary of a local metric's eigenvalues, in units of the prior precision."""
    return {
        "top": float(values[0]),
        # Number of well-determined parameters: sum of 1 - prior / posterior precision.
        "effective_dimension": float(np.sum(1 - 1 / values)),
        **{f"above_{level}": int(np.sum(values > level)) for level in (2, 10, 100)},
    }
