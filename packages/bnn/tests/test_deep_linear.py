"""Independent checks of the deep-linear quadrature reference."""

import numpy as np
import pytest
from bnn.target.deep_linear import DeepLinear, checked_reference, observable_map
from scipy.special import logsumexp
from scipy.stats import multivariate_normal


def test_depth_one_reference_is_the_conjugate_gaussian_posterior():
    model = DeepLinear(1)
    design, y = model.data()
    reference = checked_reference(model, [64, 128])
    # Observation-space identities, independent of the reference's precision form.
    evidence_cov = design @ design.T + model.noise**2 * np.eye(len(y))
    mean = design.T @ np.linalg.solve(evidence_cov, y)
    covariance = np.eye(2) - design.T @ np.linalg.solve(evidence_cov, design)
    mapping = observable_map()
    np.testing.assert_allclose(reference["mean"], mapping @ mean, atol=1e-12)
    np.testing.assert_allclose(
        reference["coefficient_covariance"], covariance, atol=1e-12
    )
    assert reference["log_evidence"] == pytest.approx(
        multivariate_normal.logpdf(y, cov=evidence_cov), abs=1e-10
    )


@pytest.mark.parametrize("depth,width", [(2, 4), (3, 8)])
def test_quadrature_matches_importance_sampling_of_prior_weights(depth, width):
    model = DeepLinear(depth, width)
    reference = checked_reference(model, [64, 128, 256])
    assert reference["qualified"]
    weights = np.random.default_rng(3).normal(size=(200_000, model.dimension))
    # Coefficients by explicit products of the prior weight matrices.
    product, offset = None, 0
    for rows, cols in model.shapes:
        matrix = weights[:, offset : offset + rows * cols].reshape(-1, rows, cols)
        product = matrix if product is None else matrix @ product
        offset += rows * cols
    beta = product.reshape(-1, 2) / width ** ((depth - 1) / 2)
    variance = [model.variance(row) for row in weights[:20_000]]
    # The explicit product of every layer but the first gives the same variance.
    later, offset = None, 2 * width
    for rows, cols in model.shapes[1:]:
        matrix = weights[:, offset : offset + rows * cols].reshape(-1, rows, cols)
        later = matrix if later is None else matrix @ later
        offset += rows * cols
    explicit = np.sum(later**2, axis=(1, 2)) / width ** (depth - 1)
    np.testing.assert_allclose(variance, explicit[:20_000], rtol=1e-12)
    explicit = np.log(explicit)
    design, y = model.data()
    loglik = multivariate_normal.logpdf(
        y[None] - beta @ design.T, cov=model.noise**2 * np.eye(len(y))
    )
    w = np.exp(loglik - logsumexp(loglik))
    ess = 1 / np.sum(w**2)
    observables = beta @ observable_map().T
    mean = w @ observables
    sd = np.sqrt(w @ (observables - mean) ** 2)
    error = np.abs(mean - reference["mean"]) / reference["sd"]
    assert np.all(error < 5 / np.sqrt(ess))
    np.testing.assert_allclose(sd, reference["sd"], rtol=10 / np.sqrt(ess))
    log_mean = w @ explicit
    log_sd = np.sqrt(w @ (explicit - log_mean) ** 2)
    reference_mean, reference_sd = reference["log_variance"]
    assert abs(log_mean - reference_mean) < 5 * reference_sd / np.sqrt(ess)
    assert log_sd == pytest.approx(reference_sd, rel=10 / np.sqrt(ess))
    log_evidence = logsumexp(loglik) - np.log(len(loglik))
    assert log_evidence == pytest.approx(
        reference["log_evidence"], abs=5 / np.sqrt(ess)
    )
