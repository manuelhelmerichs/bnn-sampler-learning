"""Metropolis-corrected HMC with a normalized separable kinetic energy.

The momentum has independent coordinates with a density ``rho`` proportional
to ``exp(-k(r))``, drawn exactly, and the leapfrog moves the position by ``eps * k'(r)``. Every
family leaves the target invariant, so a family can only change efficiency.

For a symmetric ``rho`` that falls in ``|r|``, the stationary flux through a
surface ``{z_0 = x}`` is ``pi_0(x) * E[k'(r)_+] = pi_0(x) * rho(0)`` per unit
time, with ``pi_0`` the marginal density of ``z_0``. Chains change mode at most
twice that often, which is ``2 * eps * pi_0(x) * rho(0)`` per gradient.
"""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import gammaln

JITTER = 0.2


def gaussian():
    return {
        "k": lambda r: 0.5 * r**2,
        "dk": lambda r: r,
        "sample": lambda key, shape, dtype: jax.random.normal(key, shape, dtype),
        "rest": 1 / np.sqrt(2 * np.pi),
    }


def power(p):
    """``rho`` proportional to ``exp(-|r|^p / p)``: Laplace at 1, Gaussian at 2."""

    def sample(key, shape, dtype):
        k1, k2 = jax.random.split(key)
        size = (p * jax.random.gamma(k1, 1 / p, shape, dtype)) ** (1 / p)
        return jnp.where(jax.random.bernoulli(k2, 0.5, shape), size, -size)

    return {
        "k": lambda r: jnp.abs(r) ** p / p,
        "dk": lambda r: jnp.sign(r) * jnp.abs(r) ** (p - 1),
        "sample": sample,
        "rest": float(np.exp(-np.log(2) - np.log(p) / p - gammaln(1 + 1 / p))),
    }


def student(nu):
    return {
        "k": lambda r: 0.5 * (nu + 1) * jnp.log1p(r**2 / nu),
        "dk": lambda r: (nu + 1) * r / (nu + r**2),
        "sample": lambda key, shape, dtype: jax.random.t(key, nu, shape, dtype),
        "rest": float(
            np.exp(gammaln((nu + 1) / 2) - gammaln(nu / 2)) / np.sqrt(nu * np.pi)
        ),
    }


def scale_mixture(weight, ratio):
    """``weight N[0, 1] + (1 - weight) N[0, ratio^2]``."""
    logs = np.log([weight, (1 - weight) / ratio])
    precisions = np.array([1.0, ratio**-2])

    def pieces(r):
        return logs - 0.5 * precisions * r[..., None] ** 2

    def sample(key, shape, dtype):
        k1, k2 = jax.random.split(key)
        wide = jax.random.bernoulli(k1, 1 - weight, shape)
        return jax.random.normal(k2, shape, dtype) * jnp.where(wide, ratio, 1.0)

    return {
        "k": lambda r: -jax.nn.logsumexp(pieces(r), axis=-1),
        "dk": lambda r: (
            r * jnp.sum(jax.nn.softmax(pieces(r), axis=-1) * precisions, -1)
        ),
        "sample": sample,
        "rest": float((weight + (1 - weight) / ratio) / np.sqrt(2 * np.pi)),
    }


# The Gaussian, and the families a learned separable kinetic energy is chosen
# from: lighter and heavier tails, and mixtures of two momentum scales.
FAMILIES = {
    "gaussian": gaussian(),
    "laplace": power(1.0),
    "power_1.5": power(1.5),
    "power_3": power(3.0),
    "student_3": student(3.0),
    "student_5": student(5.0),
    "student_20": student(20.0),
    "mixture_0.5_2": scale_mixture(0.5, 2.0),
    "mixture_0.5_4": scale_mixture(0.5, 4.0),
    "mixture_0.9_3": scale_mixture(0.9, 3.0),
    "mixture_0.9_6": scale_mixture(0.9, 6.0),
}


def hmc(family, energy, gradient, observe, label):
    """A batch of chains of one task.

    ``run(key, theta, step, task, transitions=, burn=, length=)`` draws 1 to
    ``length`` leapfrog steps per transition and returns, per chain: the sums of
    ``observe`` over the transitions after ``burn``, the acceptance rate, the
    gradients spent after ``burn`` and how often ``label`` changed.
    """

    def transition(key, theta, force, step, task, length):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        r = family["sample"](k1, theta.shape, theta.dtype)
        steps = jax.random.randint(k2, (), 1, length + 1)
        eps = step * jax.random.uniform(
            k3, minval=1 - JITTER, maxval=1 + JITTER, dtype=theta.dtype
        )
        start = energy(theta, task) + jnp.sum(family["k"](r))

        def leapfrog(carry, index):
            x, r, g = carry
            half = r - 0.5 * eps * g
            moved = x + eps * family["dk"](half)
            new = gradient(moved, task)
            carry = jax.tree.map(
                lambda a, b: jnp.where(index < steps, a, b),
                (moved, half - 0.5 * eps * new, new),
                carry,
            )
            return carry, None

        (x, r, g), _ = jax.lax.scan(leapfrog, (theta, r, force), jnp.arange(length))
        change = energy(x, task) + jnp.sum(family["k"](r)) - start
        probability = jnp.where(
            jnp.isfinite(change), jnp.exp(jnp.minimum(0.0, -change)), 0.0
        )
        accepted = jax.random.uniform(k4, dtype=theta.dtype) < probability
        return (
            jnp.where(accepted, x, theta),
            jnp.where(accepted, g, force),
            probability,
            steps,
        )

    def chain(key, theta, step, task, transitions, burn, length):
        def one(carry, inputs):
            theta, force, mode, sums, counts = carry
            key, index = inputs
            theta, force, probability, steps = transition(
                key, theta, force, step, task, length
            )
            keep = (index >= burn).astype(theta.dtype)
            new = label(theta, task, mode)
            counts = counts + keep * jnp.stack(
                [
                    probability,
                    steps.astype(theta.dtype),
                    ((new != mode) & (mode != 0)).astype(theta.dtype),
                ]
            )
            return (theta, force, new, sums + keep * observe(theta, task), counts), None

        mode = label(theta, task, jnp.zeros((), jnp.int32))
        carry = (
            theta,
            gradient(theta, task),
            mode,
            jnp.zeros_like(observe(theta, task)),
            jnp.zeros(3, theta.dtype),
        )
        (_, _, _, sums, counts), _ = jax.lax.scan(
            one, carry, (jax.random.split(key, transitions), jnp.arange(transitions))
        )
        kept = transitions - burn
        return sums / kept, counts[0] / kept, counts[1], counts[2]

    def run(key, theta, step, task, *, transitions, burn, length):
        return jax.vmap(
            lambda k, x: chain(k, x, step, task, transitions, burn, length)
        )(jax.random.split(key, len(theta)), theta)

    return run


def transitions_for(gradients, length):
    """Transitions that spend ``gradients`` on average."""
    return max(1, round(2 * gradients / (length + 1)))
