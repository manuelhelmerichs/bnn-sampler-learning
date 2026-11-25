"""Two-mode mixtures whose parameters are data.

In aligned coordinates ``z`` the target is ``w N[-a e_0, I] + (1 - w) N[a e_0, I]``.
A chain runs in ``theta = location + scale * rotation @ z``; the aligned frame
has neither location nor rotation, the shifted one a location only, the turned
one a rotation only, the rotated one both. Separation, weight, scale and frame
are arrays, so one compiled sampler serves every task of a cell, and a sampler
can be trained over a distribution of tasks. With unit scale this is the
``mixture`` family of ``bnn.target.synthetic`` without bend or tail.

The reference of a task is ``N[location, scale^2 I]``: the target with its two
modes merged at their midpoint, as a prior is to a posterior.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp
from scipy import integrate
from scipy.stats import norm

# Statistics a sampler accumulates over retained states, in aligned
# coordinates: the mass of the first mode, the barrier's occupancy, the
# second moment of every coordinate off the axis, and their fourth moment,
# tail mass and the squared distance of the axis coordinate from its mode
# centre, which no arm is trained on.
SCALARS = ("mass", "barrier", "fourth", "tail", "axis")
TAIL = 2.0
BARRIER = 1.0


def tasks(rng, count, dimension, separation, frame, weights=None, scale=1.0):
    """``count`` tasks as arrays; ``frame`` is aligned, shifted, turned or rotated."""
    if frame not in {"aligned", "shifted", "turned", "rotated"}:
        raise ValueError(f"Unknown frame: {frame}")
    weights = rng.uniform(0.15, 0.85, count) if weights is None else np.asarray(weights)
    task = {
        "separation": np.full(count, float(separation)),
        "logit": np.log((1 - weights) / weights),
        "scale": np.full(count, float(scale)),
        "location": np.zeros((count, dimension)),
    }
    if frame in {"turned", "rotated"}:
        q, r = np.linalg.qr(rng.normal(size=(count, dimension, dimension)))
        task["rotation"] = q * np.sign(np.diagonal(r, axis1=1, axis2=2))[:, None, :]
    if frame in {"shifted", "rotated"}:
        task["location"] = rng.normal(size=(count, dimension))
    return task


def weights(task):
    return 1 / (1 + np.exp(np.asarray(task["logit"])))


def truth(task):
    """Exact values of ``SCALARS`` and of the second moment, per task."""
    a, w = np.asarray(task["separation"]), weights(task)

    def axis(a, w):
        def density(x):
            return w * norm.pdf(x + a) + (1 - w) * norm.pdf(x - a)

        return integrate.quad(
            lambda x: (abs(x) - a) ** 2 * density(x), -a - 12, a + 12, points=[0.0]
        )[0]

    return {
        "mass": w * norm.cdf(a) + (1 - w) * norm.cdf(-a),
        "barrier": w * (norm.cdf(BARRIER + a) - norm.cdf(-BARRIER + a))
        + (1 - w) * (norm.cdf(BARRIER - a) - norm.cdf(-BARRIER - a)),
        "second": np.ones_like(a),
        "fourth": np.full_like(a, 3.0),
        "tail": np.full_like(a, 2 * norm.cdf(-TAIL)),
        "axis": np.array([axis(*pair) for pair in zip(a, w, strict=True)]),
    }


def dimension(task):
    return task["location"].shape[-1]


def rotated(vector, task, back=False):
    """Aligned to chain coordinates, or ``back``; only rotated frames store a rotation."""
    if "rotation" not in task:
        return vector
    return vector @ (task["rotation"] if back else task["rotation"].T)


def aligned(theta, task):
    return rotated(theta - task["location"], task, back=True) / task["scale"]


def embedded(z, task):
    return task["location"] + task["scale"] * rotated(z, task)


def force(theta, task):
    """Gradient of the potential at ``theta`` of shape ``(..., D)``."""
    z = aligned(theta, task)
    a = task["separation"]
    right = jax.nn.sigmoid(2 * a * z[..., 0] + task["logit"])
    gradient = z.at[..., 0].add(a - 2 * a * right)
    return rotated(gradient, task) / task["scale"]


def reference_force(theta, task):
    """Gradient of the reference's potential at ``theta`` of shape ``(..., D)``."""
    return (theta - task["location"]) / task["scale"] ** 2


def excess(theta, task):
    """Potential of the target over the reference's at one position, up to a constant."""
    z = aligned(theta, task)
    return potential(theta, task) - 0.5 * jnp.sum(z**2)


def potential(theta, task):
    """Normalized potential of one position."""
    z = aligned(theta, task)
    a, d = task["separation"], z.shape[-1]
    logs = jnp.stack(
        [
            jax.nn.log_sigmoid(-task["logit"]) - 0.5 * (z[0] + a) ** 2,
            jax.nn.log_sigmoid(task["logit"]) - 0.5 * (z[0] - a) ** 2,
        ]
    )
    return (
        0.5 * jnp.sum(z[1:] ** 2)
        - logsumexp(logs)
        + 0.5 * d * np.log(2 * np.pi)
        + d * jnp.log(task["scale"])
    )


def draws(key, task, count):
    """Exact draws in the chain's coordinates."""
    k1, k2 = jax.random.split(key)
    z = jax.random.normal(k1, (count, dimension(task)), task["location"].dtype)
    first = jax.random.uniform(k2, (count,)) < jax.nn.sigmoid(-task["logit"])
    z = z.at[:, 0].add(jnp.where(first, -1.0, 1.0) * task["separation"])
    return embedded(z, task)


def statistics(theta, task):
    """One row of statistics per state: ``SCALARS`` and then the second moments."""
    z = aligned(theta, task)
    off = z[..., 1:]
    scalars = jnp.stack(
        [
            (z[..., 0] < 0).astype(z.dtype),
            (jnp.abs(z[..., 0]) < BARRIER).astype(z.dtype),
            jnp.mean(off**4, axis=-1),
            jnp.mean((jnp.abs(off) > TAIL).astype(z.dtype), axis=-1),
            (jnp.abs(z[..., 0]) - task["separation"]) ** 2,
        ],
        axis=-1,
    )
    return jnp.concatenate([scalars, off**2], axis=-1)


def core(theta, task, previous):
    """The mode whose core a state is in (-1 or 1), or the one it was in last."""
    z = aligned(theta, task)[..., 0]
    half = task["separation"] / 2
    return jnp.where(z < -half, -1, jnp.where(z > half, 1, previous))


def barrier_density(separation, weight):
    """Smallest density of the axis coordinate between the modes."""
    x = np.linspace(-1.0, 1.0, 4001)
    return float(
        np.min(
            weight * norm.pdf(x + separation) + (1 - weight) * norm.pdf(x - separation)
        )
    )
