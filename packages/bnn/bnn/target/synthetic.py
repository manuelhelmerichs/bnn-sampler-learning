"""Normalized, independently sampleable targets for sampler measurement controls.

Every family has exact draws, a normalized density, its force, and exact first
and second moments at any dimension.

The banana, funnel and mixture are defined in aligned coordinates
``u = (theta - location) @ rotation``. Their structure sits in the first
``hard`` aligned coordinates, and the remaining ones are independent standard
normal. A task with ``hard < dimension`` is therefore the ``hard``-dimensional
target times a standard normal, rotated. The Gaussian and BLR confine their
spread of eigenvalues to ``hard`` directions in the same way.

The NumPy density/force and JAX potential are separate implementations. All
randomness is explicit; constructing a target never changes global RNG state.
"""

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np
from scipy.special import logsumexp, ndtr

FAMILIES = ("gaussian", "banana", "funnel", "mixture", "blr")


def _digest(value, namespace):
    """SHA-256 of ``value`` as canonical JSON, in its own namespace."""
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(f"{namespace}\0{text}".encode()).hexdigest()


@dataclass(frozen=True)
class Task:
    family: str
    split: str
    index: int
    seed: int
    dimension: int = 8
    condition: float = 1.0
    bend: float = 0.1
    separation: float = 2.0
    weight: float = 0.2
    # Half of the first mixture component is wider by this factor.
    tail: float = 1.0
    observations: int = 32
    noise: float = 0.3
    prior: float = 1.0
    # Standard deviation of the funnel's log-variance coordinate.
    funnel: float = 3.0
    # Leading aligned coordinates that carry the family's structure; None means
    # all of them. The others are standard normal.
    hard: int | None = None

    def __post_init__(self):
        if self.family not in FAMILIES:
            raise ValueError("Unknown target family")
        if self.split not in {"screen", "confirmation"}:
            raise ValueError("Unknown task split")
        if self.dimension < 2 or self.observations < 1 or self.index < 0:
            raise ValueError("Invalid dimensions/counts")
        if self.hard is None:
            object.__setattr__(self, "hard", self.dimension)
        if not isinstance(self.hard, int) or not 2 <= self.hard <= self.dimension:
            raise ValueError("The hard block needs 2 to `dimension` coordinates")
        if not all(
            np.isfinite(v)
            for v in (
                self.condition,
                self.bend,
                self.separation,
                self.weight,
                self.tail,
                self.noise,
                self.prior,
                self.funnel,
            )
        ):
            raise ValueError("Nonfinite task parameters")
        if (
            self.condition < 1
            or min(self.noise, self.prior, self.separation, self.funnel) <= 0
        ):
            raise ValueError("Invalid target scales")
        if not 0 < self.weight < 1:
            raise ValueError("Mixture weight must be interior")
        if self.tail < 1:
            raise ValueError("The wide half of a component cannot be narrower")

    @property
    def identity(self):
        return _digest(asdict(self), "synthetic-task-v1")

    def stream(self, role, repeat=0):
        if repeat < 0 or not role:
            raise ValueError("A named, nonnegative random stream is required")
        return int(_digest([self.identity, role, repeat], "rng")[:16], 16)


class SyntheticTarget:
    """Numerical view of an immutable Task; all exposed constants are read-only."""

    def __init__(self, task):
        self.task = task
        self.dimension = d = task.dimension
        self.hard = k = task.hard
        # Banana pairs (0, 1), (2, 3), ... inside the hard block. An odd block
        # leaves its last coordinate standard normal.
        self.pairs = k // 2
        # Mixture coordinates 1, 3, ... are sheared by their right neighbours.
        self.bent = np.arange(1, k - 1, 2)
        rng = np.random.Generator(np.random.PCG64(task.seed))
        q, r = np.linalg.qr(rng.normal(size=(d, d)))
        self.rotation = q * np.sign(np.diag(r))
        self.location = rng.normal(size=d)
        self.eigenvalues = np.concatenate(
            [np.geomspace(1 / task.condition, 1, k), np.ones(d - k)]
        )
        self.x = (
            rng.normal(size=(task.observations, d))
            @ (self.rotation * np.sqrt(self.eigenvalues)).T
        )
        self.beta = rng.normal(scale=task.prior, size=d)
        self.y = self.x @ self.beta + rng.normal(scale=task.noise, size=len(self.x))
        if task.family == "blr":
            precision = np.eye(d) / task.prior**2 + self.x.T @ self.x / task.noise**2
            self.covariance = np.linalg.solve(precision, np.eye(d))
            self.mean = self.covariance @ self.x.T @ self.y / task.noise**2
        elif task.family == "gaussian":
            self.mean = self.location.copy()
            self.covariance = (self.rotation * self.eigenvalues) @ self.rotation.T
        else:
            if task.family == "mixture":
                # Gaussian pieces along the axis: the unit and the wide half of
                # the first component, then the second component.
                self.shares = np.array(
                    [task.weight / 2, task.weight / 2, 1 - task.weight]
                )
                self.centers = task.separation * np.array([-1.0, -1.0, 1.0])
                self.variances = np.array([1.0, task.tail**2, 1.0])
            mean, covariance = self._aligned_moments()
            self.mean = self.location + self.rotation @ mean
            self.covariance = self.rotation @ covariance @ self.rotation.T
        self.cholesky = np.linalg.cholesky(self.covariance)
        self.precision = np.linalg.solve(self.covariance, np.eye(d))
        for value in vars(self).values():
            if isinstance(value, np.ndarray):
                value.setflags(write=False)

    def _aligned_moments(self):
        """Exact mean and covariance of the aligned coordinates."""
        t, d, k = self.task, self.dimension, self.hard
        mean, variance = np.zeros(d), np.ones(d)
        if t.family == "banana":
            # u_a = 3 z_a and u_c = z_c + b (u_a^2 - 9): E u_c = 0, Var u_c =
            # 1 + b^2 Var u_a^2 = 1 + 162 b^2, and Cov(u_a, u_c) = b E u_a^3 = 0.
            variance[0 : 2 * self.pairs : 2] = 9
            variance[1 : 2 * self.pairs : 2] = 1 + 162 * t.bend**2
            return mean, np.diag(variance)
        if t.family == "funnel":
            # v ~ N(0, s^2); given v the scaled coordinates are N(0, e^v), so
            # their variance is E e^v = e^(s^2 / 2) and they are uncorrelated.
            variance[0] = t.funnel**2
            variance[1:k] = np.exp(t.funnel**2 / 2)
            return mean, np.diag(variance)
        # Second and fourth moments of an unbent hard coordinate off the axis.
        second = self.shares @ self.variances
        fourth = 3 * self.shares @ self.variances**2
        axis = self.shares @ self.centers
        mean[0] = axis
        mean[self.bent] = t.bend * (second - 1)
        variance[:k] = second
        covariance = np.diag(variance)
        covariance[0, 0] += self.shares @ self.centers**2 - axis**2
        # Bent coordinates share the scale of their piece, and through it
        # correlate with each other and with the axis.
        covariance[np.ix_(self.bent, self.bent)] += t.bend**2 * (fourth / 3 - second**2)
        covariance[self.bent, self.bent] += t.bend**2 * 2 * fourth / 3
        covariance[self.bent, 0] = covariance[0, self.bent] = t.bend * (
            self.shares @ (self.variances * self.centers) - second * axis
        )
        return mean, covariance

    def forward(self, z):
        """Map standard normal to Gaussian/BLR/banana/funnel; mixture needs a label."""
        t, k = self.task, self.hard
        z = np.asarray(z)
        if t.family in {"gaussian", "blr"}:
            return self.mean + z @ self.cholesky.T
        if t.family == "mixture":
            raise ValueError("A mixture is not a single deterministic Gaussian map")
        u = z.copy()
        if t.family == "banana":
            a, c = slice(0, 2 * self.pairs, 2), slice(1, 2 * self.pairs, 2)
            u[..., a] = 3 * z[..., a]
            u[..., c] = z[..., c] + t.bend * (u[..., a] ** 2 - 9)
        else:
            u[..., 0] = t.funnel * z[..., 0]
            u[..., 1:k] = z[..., 1:k] * np.exp(u[..., :1] / 2)
        return self.location + u @ self.rotation.T

    def coordinates(self, theta):
        """Gaussian base coordinates, or unbent mixture-aligned coordinates."""
        t, k = self.task, self.hard
        theta = np.asarray(theta)
        if t.family in {"gaussian", "blr"}:
            return (theta - self.mean) @ np.linalg.inv(self.cholesky).T
        u = (theta - self.location) @ self.rotation
        z = u.copy()
        if t.family == "banana":
            a, c = slice(0, 2 * self.pairs, 2), slice(1, 2 * self.pairs, 2)
            z[..., a] = u[..., a] / 3
            z[..., c] = u[..., c] - t.bend * (u[..., a] ** 2 - 9)
        elif t.family == "funnel":
            z[..., 0] = u[..., 0] / t.funnel
            z[..., 1:k] = u[..., 1:k] * np.exp(-u[..., :1] / 2)
        else:
            z[..., self.bent] -= t.bend * (u[..., self.bent + 1] ** 2 - 1)
        return z

    def sample(self, count, seed, *, weight=None):
        if count < 1:
            raise ValueError("Positive sample count required")
        rng = np.random.Generator(np.random.PCG64(seed))
        z = rng.normal(size=(count, self.dimension))
        if self.task.family != "mixture":
            if weight is not None:
                raise ValueError("Weight intervention only applies to mixtures")
            return self.forward(z)
        w = self.task.weight if weight is None else weight
        if not 0 <= w <= 1:
            raise ValueError("Invalid sampling weight")
        first = rng.random(count) < w
        z[first & (rng.random(count) < 0.5), : self.hard] *= self.task.tail
        z[:, 0] += np.where(first, -1, 1) * self.task.separation
        z[:, self.bent] += self.task.bend * (z[:, self.bent + 1] ** 2 - 1)
        return self.location + z @ self.rotation.T

    def _piece_logs(self, z):
        """Log densities of the weighted mixture pieces, up to the 2 pi constant."""
        k = self.hard
        energy = (
            (z[..., :1] - self.centers) ** 2
            + np.sum(z[..., 1:k] ** 2, axis=-1, keepdims=True)
        ) / self.variances + np.sum(z[..., k:] ** 2, axis=-1, keepdims=True)
        return np.log(self.shares) - 0.5 * k * np.log(self.variances) - 0.5 * energy

    def logpdf(self, theta):
        t = self.task
        z = self.coordinates(theta)
        constant = self.dimension * np.log(2 * np.pi)
        if t.family in {"gaussian", "blr"}:
            return (
                -0.5 * (np.sum(z**2, axis=-1) + constant)
                - np.log(np.diag(self.cholesky)).sum()
            )
        if t.family == "banana":
            # Each shear has determinant one; each pair's first scale is THREE.
            return -0.5 * (np.sum(z**2, axis=-1) + constant) - self.pairs * np.log(3)
        if t.family == "funnel":
            # The map from z has Jacobian determinant s exp((hard - 1) v / 2).
            v = t.funnel * z[..., 0]
            return (
                -0.5 * (np.sum(z**2, axis=-1) + constant)
                - np.log(t.funnel)
                - 0.5 * (self.hard - 1) * v
            )
        return logsumexp(self._piece_logs(z), axis=-1) - constant / 2

    def force(self, theta):
        """Positive potential gradient, to be SUBTRACTED by Langevin updates."""
        t, k = self.task, self.hard
        theta = np.asarray(theta)
        if t.family in {"gaussian", "blr"}:
            return (theta - self.mean) @ self.precision
        u = (theta - self.location) @ self.rotation
        if t.family == "banana":
            a, c = slice(0, 2 * self.pairs, 2), slice(1, 2 * self.pairs, 2)
            residual = u[..., c] - t.bend * (u[..., a] ** 2 - 9)
            grad = u.copy()
            grad[..., a] = u[..., a] / 9 - 2 * t.bend * u[..., a] * residual
            grad[..., c] = residual
        elif t.family == "funnel":
            v, scaled = u[..., 0], u[..., 1:k]
            shrink = np.exp(-v)
            grad = u.copy()
            grad[..., 1:k] = scaled * shrink[..., None]
            grad[..., 0] = (
                v / t.funnel**2
                + 0.5 * (k - 1)
                - 0.5 * shrink * np.sum(scaled**2, axis=-1)
            )
        else:
            z = self.coordinates(theta)
            logs = self._piece_logs(z)
            precision = (
                np.exp(logs - logsumexp(logs, axis=-1, keepdims=True)) / self.variances
            )
            grad = z.copy()
            grad[..., :k] *= precision.sum(-1, keepdims=True)
            grad[..., 0] -= precision @ self.centers
            grad[..., self.bent + 1] -= (
                2 * t.bend * u[..., self.bent + 1] * grad[..., self.bent]
            )
        return grad @ self.rotation.T

    def potential(self):
        """Return a scalar normalized JAX potential, without setting JAX precision."""
        import jax.numpy as jnp

        constant = self.dimension * np.log(2 * np.pi) / 2
        if self.task.family in {"gaussian", "blr"}:
            mean, precision = jnp.asarray(self.mean), jnp.asarray(self.precision)
            logdet = np.log(np.diag(self.cholesky)).sum()

            def energy(theta):
                delta = theta - mean
                return 0.5 * delta @ precision @ delta + logdet + constant

            return energy
        aligned = self.aligned_potential()
        rotation, location = jnp.asarray(self.rotation), jnp.asarray(self.location)
        return lambda theta: aligned((theta - location) @ rotation)

    def aligned_potential(self):
        """The banana, funnel or mixture potential of ``(theta - location) @ rotation``.

        The task seed enters these targets through that frame alone, so one
        compiled sampler serves every seed when the frame is passed as data.
        """
        import jax.numpy as jnp
        from jax.scipy.special import logsumexp as jlogsumexp

        t, k = self.task, self.hard
        if t.family in {"gaussian", "blr"}:
            raise ValueError("Gaussian targets have no seed-independent frame")
        constant = self.dimension * np.log(2 * np.pi) / 2
        if t.family == "banana":
            pairs = self.pairs

            def energy(u):
                a, c = u[0 : 2 * pairs : 2], u[1 : 2 * pairs : 2]
                return (
                    0.5
                    * (
                        jnp.sum((a / 3) ** 2)
                        + jnp.sum((c - t.bend * (a**2 - 9)) ** 2)
                        + jnp.sum(u[2 * pairs :] ** 2)
                    )
                    + constant
                    + pairs * np.log(3)
                )

            return energy
        if t.family == "funnel":

            def energy(u):
                v = u[0]
                return (
                    0.5 * (v / t.funnel) ** 2
                    + np.log(t.funnel)
                    + 0.5 * jnp.exp(-v) * jnp.sum(u[1:k] ** 2)
                    + 0.5 * (k - 1) * v
                    + 0.5 * jnp.sum(u[k:] ** 2)
                    + constant
                )

            return energy
        bent = jnp.asarray(self.bent)
        centers, variances = jnp.asarray(self.centers), jnp.asarray(self.variances)
        offsets = np.log(self.shares) - 0.5 * k * np.log(self.variances)

        def energy(u):
            z = u.at[bent].add(-t.bend * (u[bent + 1] ** 2 - 1))
            logs = (
                offsets
                - 0.5 * ((z[0] - centers) ** 2 + jnp.sum(z[1:k] ** 2)) / variances
            )
            return constant + 0.5 * jnp.sum(z[k:] ** 2) - jlogsumexp(logs)

        return energy

    @property
    def occupancy(self):
        if self.task.family != "mixture":
            raise ValueError("Occupancy is defined only for the mixture")
        t = self.task
        first = (ndtr(t.separation) + ndtr(t.separation / t.tail)) / 2
        return float(t.weight * first + (1 - t.weight) * ndtr(-t.separation))
