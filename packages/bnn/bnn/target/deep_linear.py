"""Deep-linear coefficient map and independent induced-prior integration."""

from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from scipy.special import gammaln, logsumexp, ndtr, roots_genlaguerre


@dataclass(frozen=True)
class DeepLinear:
    depth: int
    width: int = 4
    data_seed: int = 28092026
    noise: float = 0.3

    def __post_init__(self):
        if self.depth not in (1, 2, 3) or self.width < 2 or self.noise <= 0:
            raise ValueError(
                "This small reference supports depths 1, 2, 3 and width >= 2"
            )

    @property
    def shapes(self):
        if self.depth == 1:
            return ((1, 2),)
        return (
            ((self.width, 2),)
            + ((self.width, self.width),) * (self.depth - 2)
            + ((1, self.width),)
        )

    @property
    def dimension(self):
        return sum(a * b for a, b in self.shapes)

    def coefficients(self, position):
        offset, product = 0, None
        for rows, cols in self.shapes:
            matrix = position[offset : offset + rows * cols].reshape(rows, cols)
            product = matrix if product is None else matrix @ product
            offset += rows * cols
        return product.reshape(2) / self.width ** ((self.depth - 1) / 2)

    def variance(self, position):
        """Prior variance of the coefficients given every layer but the first.

        With ``a`` the product of those layers the coefficients are ``a @ W1`` up
        to their scale, so given ``a`` their prior is ``N(0, v I)`` with ``v =
        |a|^2 / width^(depth - 1)``. Under the prior ``v`` is the product of
        ``depth - 1`` independent ``chi2_width / width`` factors, the variable the
        reference integrates by quadrature.
        """
        offset, product = sum(a * b for a, b in self.shapes[:1]), None
        for rows, cols in self.shapes[1:]:
            matrix = position[offset : offset + rows * cols].reshape(rows, cols)
            product = matrix if product is None else matrix @ product
            offset += rows * cols
        if product is None:
            return 1.0
        return (product @ product.T).reshape(()) / self.width ** (self.depth - 1)

    def data(self):
        x = np.linspace(-2, 2, 40)
        design = np.column_stack((np.ones(len(x)), x))
        y = design @ np.array([0.4, -0.7])
        y += np.random.default_rng(self.data_seed).normal(0, self.noise, len(x))
        return design, y

    def logdensity(self, position):
        import jax.numpy as jnp

        design, y = self.data()
        residual = jnp.asarray(y) - jnp.asarray(design) @ self.coefficients(position)
        return -0.5 * (position @ position + residual @ residual / self.noise**2)


def observable_map():
    locations = np.array([-2.5, -1.25, 0.0, 1.25, 2.5])
    return np.vstack((np.eye(2), np.column_stack((np.ones(5), locations))))


def reference(model, order):
    """Integrate beta|v analytically and v=product(chi2_h/h) by quadrature."""
    if model.depth == 1:
        variance, prior_logweights = np.ones(1), np.zeros(1)
    else:
        nodes, weights = roots_genlaguerre(order, model.width / 2 - 1)
        keep = weights > 0
        nodes, weights = nodes[keep], weights[keep]
        grids = np.meshgrid(
            *([2 * nodes / model.width] * (model.depth - 1)), indexing="ij"
        )
        variance = np.prod(grids, axis=0).ravel()
        logweights = np.log(weights) - gammaln(model.width / 2)
        grids = np.meshgrid(*([logweights] * (model.depth - 1)), indexing="ij")
        prior_logweights = np.sum(grids, axis=0).ravel()
    design, y = model.data()
    precision = (
        design.T @ design / model.noise**2 + np.eye(2)[None] / variance[:, None, None]
    )
    covariance = np.linalg.inv(precision)
    rhs = design.T @ y / model.noise**2
    mean = covariance @ rhs
    loglik = -0.5 * (
        len(y) * np.log(2 * np.pi * model.noise**2)
        + 2 * np.log(variance)
        + np.linalg.slogdet(precision)[1]
        + y @ y / model.noise**2
        - mean @ rhs
    )
    logz = logsumexp(prior_logweights + loglik)
    probability = np.exp(prior_logweights + loglik - logz)
    log_variance = probability @ np.log(variance)
    log_variance_sd = np.sqrt(probability @ (np.log(variance) - log_variance) ** 2)
    beta_mean = probability @ mean
    beta_cov = np.einsum("n,nij->ij", probability, covariance) + np.einsum(
        "n,ni,nj->ij", probability, mean - beta_mean, mean - beta_mean
    )
    mapping = observable_map()
    means, variances = (
        mean @ mapping.T,
        np.einsum("oi,nij,oj->no", mapping, covariance, mapping),
    )
    observable_mean = mapping @ beta_mean
    observable_sd = np.sqrt(np.diag(mapping @ beta_cov @ mapping.T))
    thresholds = observable_mean[:, None] + observable_sd[:, None] * np.array(
        [-2, -1, 0, 1, 2]
    )
    cdf = np.einsum(
        "n,nok->ok",
        probability,
        ndtr((thresholds[None] - means[:, :, None]) / np.sqrt(variances[:, :, None])),
    )
    return {
        "order": order,
        "mean": observable_mean.tolist(),
        "sd": observable_sd.tolist(),
        "coefficient_covariance": beta_cov.tolist(),
        # Posterior mean and standard deviation of the log of ``DeepLinear.variance``.
        "log_variance": [float(log_variance), float(log_variance_sd)],
        "cdf_thresholds": thresholds.tolist(),
        "cdf": cdf.tolist(),
        "log_evidence": float(logz),
    }


def checked_reference(model, orders):
    rows = [reference(model, order) for order in orders]
    checks = []
    for previous, current in pairwise(rows):
        mean_error = np.max(
            np.abs(np.array(previous["mean"]) - current["mean"]) / current["sd"]
        )
        cov_error = np.linalg.norm(
            np.array(previous["coefficient_covariance"])
            - current["coefficient_covariance"]
        ) / np.linalg.norm(current["coefficient_covariance"])
        cdf_error = np.max(np.abs(np.array(previous["cdf"]) - current["cdf"]))
        log_mean, log_sd = current["log_variance"]
        log_error = np.abs(np.array(previous["log_variance"]) - [log_mean, log_sd])
        checks.append(
            {
                "mean_sd_error": float(mean_error),
                "covariance_relative_error": float(cov_error),
                "cdf_absolute_error": float(cdf_error),
                # Zero at depth 1, where the variance is fixed.
                "log_variance_sd_error": float(log_error.max() / log_sd)
                if log_sd > 0
                else 0.0,
            }
        )
    return {
        **rows[-1],
        "refinement": checks,
        "qualified": bool(checks and max(checks[-1].values()) < 0.002),
    }
