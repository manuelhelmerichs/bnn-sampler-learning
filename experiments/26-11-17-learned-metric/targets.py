"""Targets of the metric comparison in their aligned frame, with their exact maps.

Every family is a standard normal ``z`` pushed through a known map: ``inverse``
turns ``z`` into an exact draw ``u`` and ``forward`` Gaussianizes the target.
Chains run in the aligned frame ``u = (theta - location) @ rotation`` of
``bnn.target.synthetic``, where the map acts on single coordinates and
pairs. No arm of the comparison depends on the frame: dense metrics, estimated
or exact, and the maps are equivariant under rotations, so a chain in this frame
is the same experiment as one in the task's coordinates. A diagonal metric would
depend on the frame and has no arm here.
"""

from dataclasses import dataclass

import numpy as np
from bnn.target.synthetic import SyntheticTarget, Task
from scipy.special import ndtr

FAMILIES = ("gaussian", "banana", "funnel")
# Both estimands have the exact value 1 at every dimension. The first is the
# primary: a second moment along the softest direction, along the bent
# coordinate, and of the funnel's scale coordinate. The funnel's second is the
# probability of its neck, v < -s, over its exact value.
ESTIMANDS = {
    "gaussian": ("soft", "stiff"),
    "banana": ("bent", "wide"),
    "funnel": ("scale", "neck"),
}


@dataclass(frozen=True)
class Frame:
    task: Task
    # Exact covariance and exact mean Hessian; both are diagonal in this frame.
    variances: np.ndarray
    curvatures: np.ndarray


def frame(task):
    if task.family not in FAMILIES:
        raise ValueError(f"No Gaussianizing map for {task.family}")
    d, k = task.dimension, task.hard
    variances, curvatures = np.ones(d), np.ones(d)
    if task.family == "gaussian":
        variances = SyntheticTarget(task).eigenvalues.copy()
        curvatures = 1 / variances
    elif task.family == "banana":
        # u_a = 3 z_a and u_c = z_c + b (u_a^2 - 9). The Hessian of a pair has
        # d2U/da2 = 1/9 - 2 b r + 4 b^2 a^2 with r = z_c, and mean 1/9 + 36 b^2.
        pairs = k // 2
        variances[0 : 2 * pairs : 2] = 9
        variances[1 : 2 * pairs : 2] = 1 + 162 * task.bend**2
        curvatures[0 : 2 * pairs : 2] = 1 / 9 + 36 * task.bend**2
    else:
        # v ~ N(0, s^2) and u_i = z_i exp(v / 2): d2U/dv2 has the mean
        # 1/s^2 + (k - 1)/2 and d2U/du_i2 = exp(-v) the mean exp(s^2 / 2).
        variances[0] = task.funnel**2
        variances[1:k] = np.exp(task.funnel**2 / 2)
        curvatures[0] = 1 / task.funnel**2 + (k - 1) / 2
        curvatures[1:k] = np.exp(task.funnel**2 / 2)
    return Frame(task, variances, curvatures)


def potential(task):
    """``U(u)`` in the aligned frame, up to a constant for the Gaussian."""
    import jax.numpy as jnp

    if task.family != "gaussian":
        return SyntheticTarget(task).aligned_potential()
    variances = jnp.asarray(frame(task).variances)
    return lambda u: 0.5 * jnp.sum(u**2 / variances)


def maps(task):
    """``forward(u) = z`` and ``inverse(z) = u`` for one point, in JAX."""
    import jax.numpy as jnp

    k = task.hard
    if task.family == "gaussian":
        scales = jnp.asarray(np.sqrt(frame(task).variances))
        return (lambda u: u / scales), (lambda z: z * scales)
    if task.family == "banana":
        bend, pairs = task.bend, k // 2
        a, c = slice(0, 2 * pairs, 2), slice(1, 2 * pairs, 2)

        def forward(u):
            return u.at[a].set(u[a] / 3).at[c].set(u[c] - bend * (u[a] ** 2 - 9))

        def inverse(z):
            wide = 3 * z[a]
            return z.at[a].set(wide).at[c].set(z[c] + bend * (wide**2 - 9))

        return forward, inverse
    scale = task.funnel

    def forward(u):
        return u.at[0].set(u[0] / scale).at[1:k].set(u[1:k] * jnp.exp(-u[0] / 2))

    def inverse(z):
        v = scale * z[0]
        return z.at[0].set(v).at[1:k].set(z[1:k] * jnp.exp(v / 2))

    return forward, inverse


def estimands(task):
    """The two estimands of one aligned position."""
    import jax.numpy as jnp

    if task.family == "gaussian":
        picked = jnp.asarray([task.hard - 1, 0])
        scales = jnp.asarray(frame(task).variances)[picked]
        return lambda u: u[picked] ** 2 / scales
    if task.family == "banana":
        scales = jnp.asarray([1 + 162 * task.bend**2, 9.0])
        return lambda u: u[jnp.asarray([1, 0])] ** 2 / scales
    scale, neck = task.funnel, float(ndtr(-1.0))
    return lambda u: jnp.stack([(u[0] / scale) ** 2, (u[0] < -scale) / neck])


def sample(task, count, rng):
    """Exact draws in the aligned frame."""
    import jax

    z = rng.normal(size=(count, task.dimension))
    return np.asarray(jax.vmap(maps(task)[1])(z))
