"""Synthetic targets in the frame where their covariance is diagonal.

HMC with an identity mass, a scalar step size and isotropic momentum is
equivariant under rotations and translations, and every injected error field
has a rotation-invariant law. A chain run in this frame is therefore the same
experiment as one run in the task's own coordinates, at O(D) instead of O(D^2)
per gradient. The learned force is the exception: it is fitted and evaluated in
the task's coordinates, ``theta = origin + rotation @ u``.
"""

from dataclasses import dataclass

import numpy as np
from bnn.target.synthetic import SyntheticTarget, Task

FAMILIES = ("gaussian", "blr", "banana")
# Every estimand is a second moment along one aligned coordinate, divided by its
# exact value, so its truth is 1 at every dimension. The first is the primary.
ESTIMANDS = {
    "gaussian": ("soft", "stiff"),
    "blr": ("soft", "stiff"),
    "banana": ("bent", "wide"),
}


@dataclass(frozen=True)
class Frame:
    task: Task
    variances: np.ndarray
    rotation: np.ndarray
    origin: np.ndarray
    # The two coordinates of the estimands.
    indexes: tuple[int, int]
    # Largest Hessian eigenvalue on the typical set and E |grad U|^2.
    stiffness: float
    force_power: float

    @property
    def dimension(self):
        return self.task.dimension

    def sample(self, count, rng):
        """Exact draws in the aligned frame."""
        z = rng.normal(size=(count, self.dimension))
        if self.task.family != "banana":
            return z * np.sqrt(self.variances)
        u, pairs = z.copy(), self.dimension_pairs
        a, c = slice(0, 2 * pairs, 2), slice(1, 2 * pairs, 2)
        u[:, a] = 3 * z[:, a]
        u[:, c] = z[:, c] + self.task.bend * (u[:, a] ** 2 - 9)
        return u

    @property
    def dimension_pairs(self):
        return self.task.hard // 2

    def to_task(self, u):
        return self.origin + u @ self.rotation.T


def frame(task):
    target = SyntheticTarget(task)
    d, family = task.dimension, task.family
    if family == "gaussian":
        # The task's rotation diagonalizes it, with the eigenvalues ascending
        # over the hard block: coordinate 0 is the stiffest, hard - 1 the softest.
        variances, rotation, origin = target.eigenvalues, target.rotation, target.mean
        indexes = (task.hard - 1, 0)
    elif family == "blr":
        variances, rotation = np.linalg.eigh(target.covariance)
        origin, indexes = target.mean, (d - 1, 0)
    elif family == "banana":
        rotation, origin = target.rotation, target.location
        variances = np.diag(rotation.T @ target.covariance @ rotation)
        # The bent and the wide coordinate of the first pair.
        indexes = (1, 0)
    else:
        raise ValueError(f"No aligned frame for {family}")
    if family == "banana":
        b, pairs = task.bend, task.hard // 2
        # Per pair, E (dU/du_a)^2 = 1/9 + 36 b^2 and E (dU/du_c)^2 = 1.
        power = pairs * (1 / 9 + 36 * b**2 + 1) + d - 2 * pairs
        stiffness = _banana_stiffness(b)
    else:
        power, stiffness = float(np.sum(1 / variances)), float(1 / variances.min())
    return Frame(
        task,
        np.asarray(variances, float),
        np.asarray(rotation, float),
        np.asarray(origin, float),
        indexes,
        float(stiffness),
        float(power),
    )


def _banana_stiffness(bend, quantile=0.999, count=1 << 16):
    """A high quantile of the largest Hessian eigenvalue of one banana pair."""
    rng = np.random.default_rng(0)
    a = 3 * rng.normal(size=count)
    r = rng.normal(size=count)
    first = 1 / 9 - 2 * bend * r + 4 * bend**2 * a**2
    off = -2 * bend * a
    trace, determinant = first + 1, first - off**2
    largest = 0.5 * (trace + np.sqrt(trace**2 - 4 * determinant))
    return max(float(np.quantile(largest, quantile)), 1.0)


def potential(task):
    """``U(u, variances)`` in the aligned frame, up to a constant."""
    import jax.numpy as jnp

    if task.family != "banana":
        return lambda u, variances: 0.5 * jnp.sum(u**2 / variances)
    pairs, bend = task.hard // 2, task.bend

    def energy(u, variances):
        a, c = u[0 : 2 * pairs : 2], u[1 : 2 * pairs : 2]
        return 0.5 * (
            jnp.sum((a / 3) ** 2)
            + jnp.sum((c - bend * (a**2 - 9)) ** 2)
            + jnp.sum(u[2 * pairs :] ** 2)
        )

    return energy


def estimands(indexes):
    """Normalized second moments along the estimand coordinates."""
    import jax.numpy as jnp

    picked = jnp.asarray(indexes)
    return lambda u, variances: u[picked] ** 2 / variances[picked]
