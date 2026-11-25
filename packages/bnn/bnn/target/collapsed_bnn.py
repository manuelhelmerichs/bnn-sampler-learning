"""Finite continuous BNN with an exactly marginalized Gaussian readout."""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_solve


@dataclass(frozen=True)
class Target:
    kind: str
    width: int = 8
    seed: int = 21092026

    @property
    def dimension(self):
        return 6 if self.kind == "gaussian" else 3 * self.width + 1

    def data(self):
        rng = np.random.default_rng(self.seed)
        if self.kind == "gaussian":
            x = rng.normal(size=(20, 6)) / np.sqrt(6)
            y = x @ np.linspace(-0.7, 0.8, 6) + rng.normal(0, 0.7, 20)
            return x, y, np.eye(6)
        x = rng.uniform(-2, 2, (32, 2))
        y = (
            np.tanh(1.5 * x[:, 0] + 0.3)
            + 0.6 * np.tanh(1.5 * x[:, 1] - 0.4)
            + rng.normal(0, 0.25, len(x))
        )
        test = np.random.default_rng(self.seed + 1).uniform(-2, 2, (32, 2))
        return x, y, test

    def features(self, position, x):
        h = np.asarray(position[: 3 * self.width]).reshape(self.width, 3)
        activation = np.tanh(x @ h[:, :2].T / np.sqrt(2) + h[:, 2])
        return np.column_stack([activation / np.sqrt(self.width), np.ones(len(x))])

    def log_likelihood(self):
        """Return a JAX scalar function; importing this module changes no JAX state."""
        import jax.numpy as jnp
        import jax.scipy.linalg as jl

        x, y, _ = (jnp.asarray(a) for a in self.data())
        if self.kind == "gaussian":

            def likelihood(position):
                return -0.5 * jnp.sum(((y - x @ position) / 0.7) ** 2) - len(
                    x
                ) * jnp.log(0.7 * jnp.sqrt(2 * jnp.pi))

            return likelihood

        def likelihood(position):
            h = position[:-1].reshape(self.width, 3)
            phi = jnp.concatenate(
                [
                    jnp.tanh(x @ h[:, :2].T / jnp.sqrt(2) + h[:, 2])
                    / jnp.sqrt(self.width),
                    jnp.ones((len(x), 1)),
                ],
                axis=1,
            )
            log_sigma = jnp.log(0.3) + 0.5 * position[-1]
            sigma = jnp.exp(log_sigma)
            scaled = phi / sigma
            chol = jnp.linalg.cholesky(jnp.eye(self.width + 1) + scaled.T @ scaled)
            mean = jl.cho_solve((chol, True), scaled.T @ (y / sigma))
            # Residual form avoids subtracting two large positive quadratics.
            return -0.5 * (
                len(x) * (jnp.log(2 * jnp.pi) + 2 * log_sigma)
                + 2 * jnp.log(jnp.diag(chol)).sum()
                + jnp.sum(((y - phi @ mean) / sigma) ** 2)
                + mean @ mean
            )

        return likelihood

    def gaussian_reference(self):
        if self.kind != "gaussian":
            raise ValueError("Only the linear target has an analytic reference")
        x, y, _ = self.data()
        chol = np.linalg.cholesky(np.eye(6) + x.T @ x / 0.7**2)
        cov = cho_solve((chol, True), np.eye(6))
        mean = cov @ x.T @ y / 0.7**2
        c = x @ x.T + 0.7**2 * np.eye(len(x))
        logz = -0.5 * (
            len(x) * np.log(2 * np.pi)
            + np.linalg.slogdet(c)[1]
            + y @ np.linalg.solve(c, y)
        )
        return mean, cov, float(logz)


def conditional(target, position):
    """Independent NumPy implementation for checking the differentiable target."""
    x, y, test = target.data()
    phi, phi_test = (target.features(position, a) for a in (x, test))
    sigma = 0.3 * np.exp(0.5 * position[-1])
    chol = np.linalg.cholesky(np.eye(target.width + 1) + phi.T @ phi / sigma**2)
    mean = cho_solve((chol, True), phi.T @ y / sigma**2)
    cov = cho_solve((chol, True), np.eye(len(mean)))
    return mean, cov, phi_test @ mean, phi_test @ cov @ phi_test.T


def observable_names(target):
    if target.kind == "gaussian":
        return [f"parameter_{i}" for i in range(6)]
    return [
        "noise_scale",
        "hidden_squared_radius",
        "marginal_log_likelihood",
        "kernel_trace",
        "kernel_label_alignment",
        "conditional_covariance_trace",
    ] + [f"conditional_mean_test_{i}" for i in range(0, 32, 4)]


def evaluate_population(target, positions, batch_size=256):
    """Rao–Blackwellized functions and invariant observables, without readout RNG."""
    import jax
    import jax.numpy as jnp
    import jax.scipy.linalg as jl

    positions = np.asarray(positions)
    if target.kind == "gaussian":
        return {
            "observables": positions,
            "mean": positions.mean(0),
            "covariance": np.cov(positions, rowvar=False, ddof=0),
            "mean_kernel": np.eye(6),
            "conditional_mean_trace": float(positions.var(0).sum()),
            "conditional_covariance_trace": 0.0,
        }
    x, y, test = (jnp.asarray(a) for a in target.data())
    loglik = target.log_likelihood()

    @jax.jit
    @jax.vmap
    def evaluate(position):
        h = position[:-1].reshape(target.width, 3)

        def features(z):
            return jnp.concatenate(
                [
                    jnp.tanh(z @ h[:, :2].T / jnp.sqrt(2) + h[:, 2])
                    / jnp.sqrt(target.width),
                    jnp.ones((len(z), 1)),
                ],
                axis=1,
            )

        phi, f = features(x), features(test)
        sigma = 0.3 * jnp.exp(0.5 * position[-1])
        chol = jnp.linalg.cholesky(jnp.eye(target.width + 1) + phi.T @ phi / sigma**2)
        m = jl.cho_solve((chol, True), phi.T @ y / sigma**2)
        a = jl.solve_triangular(chol, f.T, lower=True).T
        b, c = f @ m, a @ a.T
        kernel = phi @ phi.T
        centered = kernel - kernel.mean(0) - kernel.mean(1)[:, None] + kernel.mean()
        yc = y - y.mean()
        alignment = yc @ centered @ yc / (jnp.linalg.norm(centered) * (yc @ yc))
        obs = jnp.concatenate(
            [
                jnp.array(
                    [
                        sigma,
                        position[:-1] @ position[:-1],
                        loglik(position),
                        jnp.trace(kernel),
                        alignment,
                        jnp.trace(c),
                    ]
                ),
                b[::4],
            ]
        )
        return obs, b, c, kernel

    outputs = [[] for _ in range(4)]
    for start in range(0, len(positions), batch_size):
        for bucket, values in zip(
            outputs,
            jax.device_get(evaluate(positions[start : start + batch_size])),
            strict=True,
        ):
            bucket.append(np.asarray(values))
    obs, b, c, kernel = [np.concatenate(v) for v in outputs]
    return {
        "observables": obs,
        "mean": b.mean(0),
        "covariance": np.cov(b, rowvar=False, ddof=0) + c.mean(0),
        "mean_kernel": kernel.mean(0),
        "conditional_mean_trace": float(b.var(0).sum()),
        "conditional_covariance_trace": float(np.trace(c.mean(0))),
    }
