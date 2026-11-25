"""Independent marginal-density, gradient and predictive-uncertainty checks."""

import jax
import jax.numpy as jnp
import numpy as np
from bnn.target.collapsed_bnn import Target, conditional, evaluate_population
from scipy.stats import multivariate_normal


def test_collapsed_density_matches_observation_space_and_joint_factorization():
    target = Target("bnn")
    rng = np.random.default_rng(3)
    with jax.enable_x64(True):
        likelihood = target.log_likelihood()
        for _ in range(4):
            q = rng.normal(size=target.dimension)
            x, y, _ = target.data()
            phi = target.features(q, x)
            sigma = 0.3 * np.exp(0.5 * q[-1])
            c = phi @ phi.T + sigma**2 * np.eye(len(x))
            expected = multivariate_normal.logpdf(y, cov=c)
            np.testing.assert_allclose(likelihood(q), expected, atol=2e-10)
            m, cov, _, _ = conditional(target, q)
            a = rng.multivariate_normal(m, cov)
            joint = multivariate_normal.logpdf(
                a, mean=np.zeros(len(a))
            ) + multivariate_normal.logpdf(y, phi @ a, sigma**2 * np.eye(len(x)))
            np.testing.assert_allclose(
                joint - multivariate_normal.logpdf(a, m, cov), expected, atol=2e-10
            )


def test_collapsed_gradient_and_predictive_covariance():
    target = Target("bnn")
    q = np.random.default_rng(1).normal(size=target.dimension)
    with jax.enable_x64(True):
        likelihood = target.log_likelihood()
        actual = np.asarray(jax.grad(likelihood)(jnp.asarray(q)))
        eps = 1e-5
        expected = np.array(
            [
                (likelihood(q + eps * e) - likelihood(q - eps * e)) / (2 * eps)
                for e in np.eye(len(q))
            ]
        )
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
        pop = evaluate_population(target, np.stack([q, q + 0.1]))
    means, covs = zip(*(conditional(target, v)[2:] for v in (q, q + 0.1)), strict=True)
    np.testing.assert_allclose(pop["mean"], np.mean(means, axis=0), atol=1e-11)
    np.testing.assert_allclose(
        pop["covariance"],
        np.mean(covs, axis=0) + np.cov(means, rowvar=False, ddof=0),
        atol=1e-11,
    )
