"""Independent checks of synthetic target densities and moments."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from bnn.target.synthetic import FAMILIES, SyntheticTarget, Task
from scipy.integrate import dblquad, quad
from scipy.special import ndtr
from scipy.stats import multivariate_normal, norm

# Shapes with every feature switched on; the funnel is kept moderate so that
# finite differences and sample moments stay accurate.
SHAPES = {"condition": 30, "bend": 0.4, "separation": 2, "tail": 1.5, "funnel": 1.5}


@pytest.mark.parametrize("hard", [None, 3])
@pytest.mark.parametrize("family", FAMILIES)
def test_normalized_density_force_and_jax_agree(family, hard):
    target = SyntheticTarget(
        Task(family, "screen", 0, 31, dimension=5, hard=hard, **SHAPES)
    )
    points = target.sample(5, 811)
    with jax.enable_x64():
        energy = target.potential()
        actual = jax.vmap(jax.value_and_grad(energy))(jnp.asarray(points))
    np.testing.assert_allclose(actual[0], -target.logpdf(points), atol=2e-12)
    np.testing.assert_allclose(actual[1], target.force(points), atol=2e-11)
    eps = 1e-5
    finite_difference = np.column_stack(
        [
            (-target.logpdf(points + eps * e) + target.logpdf(points - eps * e))
            / (2 * eps)
            for e in np.eye(target.dimension)
        ]
    )
    np.testing.assert_allclose(
        target.force(points), finite_difference, atol=1e-7, rtol=1e-6
    )
    # Positive potential gradient must move DOWN energy when subtracted.
    step = 1e-6 * target.force(points)
    assert np.all(target.logpdf(points - step) > target.logpdf(points))


@pytest.mark.parametrize("family", ["gaussian", "banana", "funnel", "blr"])
def test_change_of_variables_uses_full_jacobian(family):
    target = SyntheticTarget(
        Task(family, "screen", 0, 7, dimension=4, condition=5, funnel=2)
    )
    z = np.array([0.8, -0.3, 1.1, -0.6])
    eps = 1e-5
    jacobian = np.column_stack(
        [
            (target.forward(z + eps * e) - target.forward(z - eps * e)) / (2 * eps)
            for e in np.eye(4)
        ]
    )
    expected_logpdf = norm.logpdf(z).sum() - np.linalg.slogdet(jacobian)[1]
    np.testing.assert_allclose(
        target.logpdf(target.forward(z)), expected_logpdf, atol=1e-9
    )
    np.testing.assert_allclose(target.coordinates(target.forward(z)), z, atol=1e-12)
    if family == "banana":
        # Two pairs, each stretched by three along its first coordinate.
        assert abs(np.linalg.det(jacobian)) == pytest.approx(9, abs=1e-8)
    if family == "funnel":
        expected = 2 * np.exp(3 * 2 * z[0] / 2)
        assert abs(np.linalg.det(jacobian)) == pytest.approx(expected, rel=1e-8)


@pytest.mark.parametrize("family", ["gaussian", "banana", "funnel", "mixture"])
def test_hard_block_is_the_smaller_target_times_a_standard_normal(family):
    hard, dimension = 4, 9
    small = SyntheticTarget(Task(family, "screen", 0, 3, dimension=hard, **SHAPES))
    large = SyntheticTarget(
        Task(family, "screen", 0, 4, dimension=dimension, hard=hard, **SHAPES)
    )
    points = small.sample(16, 5)
    easy = np.random.default_rng(6).normal(size=(16, dimension - hard))
    aligned = np.hstack([(points - small.location) @ small.rotation, easy])
    np.testing.assert_allclose(
        large.logpdf(large.location + aligned @ large.rotation.T),
        small.logpdf(points) + norm.logpdf(easy).sum(-1),
        atol=1e-10,
    )
    covariance = np.eye(dimension)
    covariance[:hard, :hard] = small.rotation.T @ small.covariance @ small.rotation
    np.testing.assert_allclose(
        large.rotation.T @ large.covariance @ large.rotation, covariance, atol=1e-10
    )


@pytest.mark.parametrize("family,bend,funnel", [("banana", 0.4, 3), ("funnel", 0, 3)])
def test_two_dimensional_targets_integrate_to_one(family, bend, funnel):
    target = SyntheticTarget(
        Task(family, "screen", 0, 5, dimension=2, bend=bend, funnel=funnel)
    )

    def density(second, first):
        u = np.array([first, second])
        return float(np.exp(target.logpdf(target.location + u @ target.rotation.T)))

    if family == "banana":
        # The second coordinate is centred on the parabola b (u_0^2 - 9).
        mass = dblquad(
            density,
            -45,
            45,
            lambda a: bend * (a * a - 9) - 12,
            lambda a: bend * (a * a - 9) + 12,
        )[0]
    else:
        # Given v the second coordinate has standard deviation exp(v / 2).
        mass = dblquad(
            density,
            -12 * funnel,
            12 * funnel,
            lambda v: -14 * np.exp(v / 2),
            lambda v: 14 * np.exp(v / 2),
            epsabs=1e-12,
        )[0]
    assert mass == pytest.approx(1, abs=1e-8)


def test_gaussian_moments_density_and_curvature_are_exact():
    target = SyntheticTarget(Task("gaussian", "screen", 0, 9, condition=1000))
    expected_covariance = (
        target.rotation @ np.diag(np.geomspace(0.001, 1, 8)) @ target.rotation.T
    )
    np.testing.assert_allclose(target.covariance, expected_covariance, atol=1e-15)
    assert np.linalg.eigvalsh(target.covariance).min() > 0
    points = target.sample(17, 18)
    np.testing.assert_allclose(
        target.logpdf(points),
        multivariate_normal.logpdf(
            points, mean=target.location, cov=expected_covariance
        ),
        atol=1e-11,
    )
    with jax.enable_x64():
        np.testing.assert_allclose(
            jax.hessian(target.potential())(jnp.asarray(target.mean)),
            np.linalg.inv(expected_covariance),
            atol=1e-9,
        )
    # Numerical integration of one Gaussian coordinate verifies normalization,
    # first and second moments independently of the sampler.
    mu, variance = target.mean[0], target.covariance[0, 0]
    for order, expected in [(0, 1), (1, mu), (2, variance + mu**2)]:
        value = quad(
            lambda x, order=order: x**order * norm.pdf(x, mu, np.sqrt(variance)),
            -np.inf,
            np.inf,
        )[0]
        assert value == pytest.approx(expected, abs=1e-9)


def test_blr_density_equals_joint_divided_by_exact_evidence():
    target = SyntheticTarget(
        Task(
            "blr",
            "screen",
            0,
            13,
            dimension=3,
            observations=7,
            noise=0.7,
            prior=1.3,
            condition=10,
        )
    )
    prior_cov = target.task.prior**2 * np.eye(3)
    noise_cov = target.task.noise**2 * np.eye(7)
    evidence_cov = target.x @ prior_cov @ target.x.T + noise_cov
    log_evidence = multivariate_normal.logpdf(target.y, cov=evidence_cov)
    for beta in target.sample(4, 915):
        expected = (
            multivariate_normal.logpdf(beta, cov=prior_cov)
            + multivariate_normal.logpdf(target.y, mean=target.x @ beta, cov=noise_cov)
            - log_evidence
        )
        assert target.logpdf(beta) == pytest.approx(expected, abs=2e-12)
    # Independent observation-space posterior identity (not the normal equations).
    expected_mean = prior_cov @ target.x.T @ np.linalg.solve(evidence_cov, target.y)
    expected_cov = prior_cov - prior_cov @ target.x.T @ np.linalg.solve(
        evidence_cov, target.x @ prior_cov
    )
    np.testing.assert_allclose(target.mean, expected_mean, atol=2e-13)
    np.testing.assert_allclose(target.covariance, expected_cov, atol=2e-13)


@pytest.mark.parametrize(
    "family,options",
    [
        ("gaussian", {"hard": 3, "condition": 30}),
        ("banana", {}),
        ("banana", {"hard": 2}),
        ("funnel", {"funnel": 1.0}),
        ("funnel", {"funnel": 1.0, "hard": 3}),
        ("mixture", {}),
        ("mixture", {"tail": 1.5}),
        ("mixture", {"tail": 1.5, "hard": 3}),
    ],
)
def test_exact_draws_recover_independently_derived_moments(family, options):
    shape = {"bend": 0.4, "separation": 2} | options
    target = SyntheticTarget(Task(family, "screen", 0, 71, dimension=5, **shape))
    x = target.sample(131072, 270)
    blocks = x.reshape(32, -1, 5)
    means = blocks.mean(1)
    covs = np.array([np.cov(b, rowvar=False) for b in blocks])
    assert np.all(
        np.abs(means.mean(0) - target.mean) < 6 * means.std(0, ddof=1) / np.sqrt(32)
    )
    assert np.all(
        np.abs(covs.mean(0) - target.covariance) < 6 * covs.std(0, ddof=1) / np.sqrt(32)
    )


@pytest.mark.parametrize("separation,tail", [(2, 1), (5, 1), (2, 1.5)])
def test_mixture_occupancy_handles_overlap_and_weight_interventions(separation, tail):
    target = SyntheticTarget(
        Task("mixture", "screen", 0, 91, dimension=3, separation=separation, tail=tail)
    )
    first = (ndtr(separation) + ndtr(separation / tail)) / 2
    expected = 0.2 * first + 0.8 * ndtr(-separation)
    assert target.occupancy == pytest.approx(expected)
    x = target.sample(131072, 86)
    actual = np.mean(target.coordinates(x)[:, 0] < 0)
    assert abs(actual - expected) < 5 * np.sqrt(expected * (1 - expected) / len(x))
    if (separation, tail) == (2, 1):
        assert abs(expected - target.task.weight) > 0.01
    if tail > 1:
        # The wide half loses more mass across the boundary than the unit half.
        assert expected < 0.2 * ndtr(separation) + 0.8 * ndtr(-separation) - 0.005
    u = target.coordinates(x[:100])
    expected_pdf = (
        0.1 * multivariate_normal.pdf(u, mean=[-separation, 0, 0])
        + 0.1 * multivariate_normal.pdf(u, mean=[-separation, 0, 0], cov=tail**2)
        + 0.8 * multivariate_normal.pdf(u, mean=[separation, 0, 0])
    )
    np.testing.assert_allclose(np.exp(target.logpdf(x[:100])), expected_pdf, rtol=1e-12)
    # The bend couples coordinate 1 to the square of coordinate 2; undoing it
    # must leave the plain mixture, which has no such dependence.
    raw = (x - target.location) @ target.rotation
    base = target.coordinates(x)
    scale = tail**3 * np.sqrt(2 / len(x))
    # A coordinate off the axis has these second and fourth moments.
    second, fourth = 0.9 + 0.1 * tail**2, 3 * (0.9 + 0.1 * tail**4)
    assert np.mean(raw[:, 1] * raw[:, 2] ** 2) == pytest.approx(
        0.1 * (fourth - second), abs=6 * scale
    )
    assert abs(np.mean(base[:, 1] * base[:, 2] ** 2)) < 6 * scale
    w = 0.2 + 0.06 / (first - ndtr(-separation))
    altered = target.sample(131072, 87, weight=w)
    assert abs(np.mean(target.coordinates(altered)[:, 0] < 0) - expected - 0.06) < 0.006
