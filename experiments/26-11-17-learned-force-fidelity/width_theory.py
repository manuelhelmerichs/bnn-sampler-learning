"""Small analytical/numerical counterchecks for width-equivariant force studies."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm


def run():
    def density(s):
        return math.exp(-math.cosh(2 * s))

    normalization = quad(density, -8, 8, epsabs=1e-12)[0]
    variance = (
        quad(lambda s: s * s * density(s), -8, 8, epsabs=1e-12)[0] / normalization
    )
    near_mode = quad(density, -0.05, 0.05, epsabs=1e-12)[0] / normalization
    # An iid Gaussian perturbation of two ordered scalar neuron coordinates.
    h, difference = 0.01, 0.1
    crossing = float(norm.cdf(-difference / math.sqrt(4 * h)))
    rng = np.random.default_rng(141027)
    noise = rng.normal(size=(200000, 2))
    empirical = float(
        np.mean(difference + math.sqrt(2 * h) * (noise[:, 0] - noise[:, 1]) < 0)
    )
    # Two vector-valued blocks exchange on a collision-free continuous path.
    angle = np.linspace(0, np.pi, 1001)
    first = np.stack((np.cos(angle), np.sin(angle)), -1)
    second = -first
    finite_data = []
    for count in (1, 4, 16, 64, 256, 1024, 4096):
        # Integrate the identified product t analytically. The remaining 1D
        # quadrature is the exact conditional-u>0 posterior for n observations
        # x=y=1 with unit observation variance and independent Gaussian priors.
        def marginal(s, count=count):
            inverse = math.exp(-2 * s)
            return math.exp(
                -0.5 * math.exp(2 * s) - 0.5 * count * inverse / (count + inverse)
            ) / math.sqrt(1 + inverse / count)

        z = quad(marginal, -40, 8, epsabs=1e-11)[0]

        def expectation(function, marginal=marginal, z=z):
            return (
                quad(lambda s: function(s) * marginal(s), -40, 8, epsabs=1e-11)[0] / z
            )

        mean_s = expectation(lambda s: s)
        mean_product = expectation(
            lambda s, count=count: count / (count + math.exp(-2 * s))
        )
        second_product = expectation(
            lambda s, count=count: (
                1 / (count + math.exp(-2 * s))
                + (count / (count + math.exp(-2 * s))) ** 2
            )
        )
        finite_data.append(
            dict(
                observations=count,
                log_scale_mean=mean_s,
                log_scale_variance=expectation(lambda s: s * s) - mean_s**2,
                product_mean=mean_product,
                product_variance=second_product - mean_product**2,
            )
        )
    return {
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "identified_product_rescaling_posterior": {
            "model": "f(x)=a ReLU(u*x), u=exp(s), a=exp(-s), independent standard normal priors, identified product a*u=1",
            "log_scale_variance": variance,
            "probability_abs_log_scale_below_0_05": near_mode,
            "quadrature_interval": [-8, 8],
            "normalization": normalization,
            "interpretation": "A nondegenerate conditional distribution after the function is fully identified; minimum norm only locates its mode.",
        },
        "finite_data_exact_quadrature": finite_data,
        "ordered_scalar_noisy_crossing": {
            "step": h,
            "initial_difference": difference,
            "exact_probability": crossing,
            "empirical_probability": empirical,
        },
        "vector_block_exchange": {
            "minimum_block_distance": float(
                np.linalg.norm(first - second, axis=-1).min()
            ),
            "endpoint_swap_max_error": float(np.max(np.abs(first[-1] - second[0]))),
            "interpretation": "A continuous path can exchange two 2D blocks without a collision; this path is not asserted to stay on a function-equivalence manifold.",
        },
    }
