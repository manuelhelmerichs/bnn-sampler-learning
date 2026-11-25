"""Gaussian proposals fitted to a chain, and random-walk proposals shaped by a metric."""

import numpy as np


def fit(draws):
    """Mean, coordinate scales and correlation eigensystem of a corpus, largest first."""
    mean = draws.mean(axis=0, dtype=np.float64)
    covariance = np.cov(draws, rowvar=False, dtype=np.float64)
    scale = np.sqrt(np.diag(covariance))
    values, vectors = np.linalg.eigh(covariance / np.outer(scale, scale))
    values = np.maximum(values[::-1], 1e-12)
    return {
        "mean": mean,
        "scale": scale,
        "values": values,
        "vectors": vectors[:, ::-1],
        "covariance": covariance,
    }


def gaussian(model, rank):
    """The fitted Gaussian with a low-rank-plus-diagonal covariance.

    The leading ``rank`` eigendirections of the correlation matrix keep their
    variance and the others share the mean of the remaining eigenvalues. Rank 0
    is the diagonal Gaussian and full rank the moment-matched one.
    """
    values = model["values"].copy()
    if rank < len(values):
        values[rank:] = values[rank:].mean()
    return {**model, "values": values}


def sample(proposal, rng, count):
    noise = rng.standard_normal((count, len(proposal["values"])))
    whitened = (noise * np.sqrt(proposal["values"])) @ proposal["vectors"].T
    return proposal["mean"] + proposal["scale"] * whitened


def log_density(proposal, points):
    whitened = ((points - proposal["mean"]) / proposal["scale"]) @ proposal["vectors"]
    values = proposal["values"]
    return (
        -0.5 * np.sum(whitened**2 / values, axis=1)
        - 0.5 * np.sum(np.log(values))
        - np.sum(np.log(proposal["scale"]))
        - 0.5 * len(values) * np.log(2 * np.pi)
    )


def independence_scores(at_proposals, at_chain):
    """How a proposal does as an independence proposal, from log importance weights.

    ``at_proposals`` are log weights (unnormalized log posterior minus log
    proposal density) at draws of the proposal and ``at_chain`` at draws of the
    chain. The Jeffreys divergence is the sum of both KL divergences between
    the chain's law and the proposal; the normalizing constant cancels.
    """
    weights = np.exp(at_proposals - at_proposals.max())
    ratio = at_proposals[None, :] - at_chain[:, None]
    return {
        "ess_fraction": float(weights.sum() ** 2 / np.sum(weights**2) / len(weights)),
        "acceptance": float(np.mean(np.exp(np.minimum(ratio, 0.0)))),
        "jeffreys": float(at_chain.mean() - at_proposals.mean()),
    }


def local_gaussian(target, center, factor, scales, count, rng):
    """Draws of ``N(center, scale * metric^-1)`` as proposals around a chain draw.

    Returns per scale the mean and median drop of the log posterior, the drop
    ``scale * dimension / 2`` a posterior with this constant curvature would
    give, and the spread and effective sample size of the importance weights.
    """
    import jax.numpy as jnp

    dimension = len(center)
    start = float(target["potential"](jnp.asarray(center[None]))[0])
    rows = []
    for scale in scales:
        noise = rng.standard_normal((count, dimension))
        points = center + np.sqrt(scale) * noise @ factor.T
        drop = np.asarray(target["potential"](jnp.asarray(points))) - start
        logweight = -drop + 0.5 * np.sum(noise**2, axis=1)
        weights = np.exp(logweight - logweight.max())
        rows.append(
            {
                "drop_mean": float(drop.mean()),
                "drop_median": float(np.median(drop)),
                "quadratic": scale * dimension / 2,
                "logweight_sd": float(logweight.std()),
                "ess_fraction": float(weights.sum() ** 2 / np.sum(weights**2) / count),
            }
        )
    return rows


def walk(target, center, factor, steps, count, rng, local=None):
    """Random-walk Metropolis proposals of root mean square length ``steps``.

    The proposal covariance is ``factor factor'`` scaled to the step length. A
    constant metric gives a symmetric proposal. ``local`` is the log determinant
    of the exact local metric at ``center``; the proposal then depends on the
    position, and its acceptance includes the density of the reverse move under
    the metric at the proposed point.

    Returns per step length the mean and median drop of the log posterior, the
    mean acceptance and the expected squared jump distance.
    """
    import jax.numpy as jnp

    dimension = len(center)
    trace = np.sum(factor**2)
    start = float(target["potential"](jnp.asarray(center[None]))[0])
    rows = []
    for step in steps:
        scale = step / np.sqrt(trace)
        noise = rng.standard_normal((count, dimension))
        moves = scale * noise @ factor.T
        points = center + moves
        drop = np.asarray(target["potential"](jnp.asarray(points))) - start
        log_ratio = -drop
        if local is not None:
            determinant, quadratic = (
                np.asarray(value)
                for value in target["reverse"](jnp.asarray(points), jnp.asarray(moves))
            )
            log_ratio = (
                log_ratio
                + 0.5 * (determinant - local)
                - 0.5 * (quadratic / scale**2 - np.sum(noise**2, axis=1))
            )
        acceptance = np.exp(np.minimum(log_ratio, 0.0))
        rows.append(
            {
                "drop_mean": float(drop.mean()),
                "drop_median": float(np.median(drop)),
                "acceptance": float(acceptance.mean()),
                "esjd": float(np.mean(np.sum(moves**2, axis=1) * acceptance)),
            }
        )
    return rows
