"""Adaptive tempered SMC with exact weights and full diagnostics.

The sampler targets ``pi_lmbda(x) i~ p0(x) * L(x)^lmbda`` and anneals ``lmbda``
from 0 to 1 adaptively (bisection on the incremental effective sample size).
Each bridge performs the standard reweight -> systematic resample -> MALA
rejuvenation cycle, so the final population at ``lmbda = 1`` is *equally
weighted by construction*; the result still carries explicit log-weights and
the run fails loudly if the final bridge is not reached (no silent partial
bridges).

MALA is preconditioned per bridge with the population standard deviation and
its step size is adapted toward a target acceptance within each bridge (two
half-rounds; parameters for a round are fixed before the round starts, so
every round is a valid MCMC kernel for the current bridge target).

Density contract: ``logprior_fn(x, params)`` and ``loglik_fn(x, params)`` are
scalar JAX functions of a single position ``x`` of shape ``(d,)``; ``params``
is a pytree of arrays threaded through jit so one compiled kernel serves a
whole (dataset, temperature, prior) grid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np


@dataclass
class SMCConfig:
    n_mcmc: int = 8
    target_ess: float = 0.8  # incremental ESS fraction targeted by bisection
    max_bridges: int = 300
    init_step: float = 0.5  # in preconditioned units
    acc_target: float = 0.65
    seed: int = 0


@dataclass
class SMCResult:
    positions: np.ndarray  # (n, d) float64 at lmbda = 1
    log_z: float  # log normalizing constant of L under p0
    bridges: list[dict] = field(default_factory=list)
    n_loglik_evals: int = 0

    def acceptance_summary(self) -> dict:
        accs = [b["acc"] for b in self.bridges]
        return {
            "mean": float(np.mean(accs)) if accs else float("nan"),
            "min": float(np.min(accs)) if accs else float("nan"),
            "max": float(np.max(accs)) if accs else float("nan"),
        }


def systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Systematic resampling indices for normalized ``weights``."""
    points = (rng.random() + np.arange(len(weights))) / len(weights)
    cdf = np.cumsum(weights)
    cdf[-1] = 1.0
    return np.searchsorted(cdf, points, side="right")


_KERNEL_CACHE: dict = {}


def _build_kernels(logprior_fn, loglik_fn):
    """Jitted batch evaluators and one MALA round, cached per density pair."""
    key = (logprior_fn, loglik_fn)
    if key in _KERNEL_CACHE:
        return _KERNEL_CACHE[key]

    def logpost(x, params, lmbda):
        return logprior_fn(x, params) + lmbda * loglik_fn(x, params)

    @jax.jit
    def batch_eval(pos, params, lmbda):
        lp = jax.vmap(lambda x: logprior_fn(x, params))(pos)
        ll = jax.vmap(lambda x: loglik_fn(x, params))(pos)
        return lp, ll, lp + lmbda * ll

    value_and_grad = jax.value_and_grad(logpost)

    def mala_step(carry, key):
        pos, logp, grad, params, lmbda, step, sigma = carry
        n = pos.shape[0]
        k1, k2 = jax.random.split(key)
        noise = jax.random.normal(k1, pos.shape)
        var = (step * sigma) ** 2
        drift = 0.5 * var * grad
        prop = pos + drift + step * sigma * noise
        logp_p, grad_p = jax.vmap(lambda x: value_and_grad(x, params, lmbda))(prop)
        drift_p = 0.5 * var * grad_p
        # log q(x|x') - log q(x'|x) with diagonal preconditioner
        fwd = jnp.sum((prop - pos - drift) ** 2 / var, axis=1)
        bwd = jnp.sum((pos - prop - drift_p) ** 2 / var, axis=1)
        log_alpha = logp_p - logp + 0.5 * (fwd - bwd)
        accept = jnp.log(jax.random.uniform(k2, (n,))) < log_alpha
        pos = jnp.where(accept[:, None], prop, pos)
        logp = jnp.where(accept, logp_p, logp)
        grad = jnp.where(accept[:, None], grad_p, grad)
        acc = jnp.mean(jnp.exp(jnp.minimum(log_alpha, 0.0)))
        return (pos, logp, grad, params, lmbda, step, sigma), acc

    @partial(jax.jit, static_argnames=("n_steps",))
    def mala_round(key, pos, params, lmbda, step, sigma, *, n_steps: int):
        logp, grad = jax.vmap(lambda x: value_and_grad(x, params, lmbda))(pos)
        carry = (pos, logp, grad, params, lmbda, step, sigma)
        keys = jax.random.split(key, n_steps)
        carry, accs = jax.lax.scan(mala_step, carry, keys)
        return carry[0], jnp.mean(accs)

    kernels = (batch_eval, mala_round)
    _KERNEL_CACHE[key] = kernels
    return kernels


def next_temperature(loglik, beta: float, fraction: float = 0.8) -> float:
    """Largest temperature <= 1 whose incremental ESS keeps ``fraction`` of particles.

    The incoming population has equal weights. Bisection always advances the
    temperature by at least 1e-12, so a sampler cannot stall on one bridge.
    """
    loglik = np.asarray(loglik, dtype=np.float64)

    def ess_fraction(candidate):
        logw = (candidate - beta) * loglik
        w = np.exp(logw - logw.max())
        return w.sum() ** 2 / (len(w) * np.sum(w**2))

    if ess_fraction(1.0) >= fraction:
        return 1.0
    lo, hi = beta, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if ess_fraction(mid) >= fraction:
            lo = mid
        else:
            hi = mid
    return max(lo, beta + 1e-12)


def tempered_smc(
    logprior_fn,
    loglik_fn,
    params,
    init_positions: np.ndarray,
    config: SMCConfig,
    *,
    bridge_stats_fn=None,
) -> SMCResult:
    """Run adaptive tempered SMC from exact prior draws to ``lmbda = 1``."""
    # Float64 numerics are scoped to the run; global JAX state is unchanged.
    with jax.enable_x64(True):
        batch_eval, mala_round = _build_kernels(logprior_fn, loglik_fn)
        rng = np.random.default_rng(config.seed)
        key = jax.random.key(config.seed)
        pos = jnp.asarray(init_positions, jnp.float64)
        n, _ = pos.shape
        n_half = max(config.n_mcmc // 2, 1)
        n_rest = max(config.n_mcmc - n_half, 1)

        lmbda = 0.0
        step = config.init_step
        log_z = 0.0
        bridges: list[dict] = []
        n_evals = n  # initial batch evaluation

        _, ll, _ = batch_eval(pos, params, lmbda)
        ll = np.asarray(ll)
        ll = np.where(np.isfinite(ll), ll, -np.inf)
        if not np.any(np.isfinite(ll)):
            raise RuntimeError("no particle has finite log-likelihood at lmbda=0")

        for _ in range(config.max_bridges):
            new_lmbda = next_temperature(ll, lmbda, config.target_ess)
            dl = new_lmbda - lmbda

            # incremental weights and evidence under uniform incoming weights
            logw = dl * ll
            m = logw.max()
            w = np.exp(logw - m)
            log_z_inc = float(m + np.log(np.mean(w)))
            log_z += log_z_inc
            w /= w.sum()
            ess_frac = float(1.0 / (n * np.sum(w**2)))

            ancestors = systematic_resample(w, rng)
            pos = pos[jnp.asarray(ancestors)]

            # preconditioner from the resampled population
            sig = np.asarray(jnp.std(pos, axis=0))
            sig = np.maximum(sig, 1e-6 + 1e-3 * sig.max())
            sigma = jnp.asarray(sig)

            key, k1, k2 = jax.random.split(key, 3)
            pos, acc1 = mala_round(
                k1, pos, params, new_lmbda, step, sigma, n_steps=n_half
            )
            acc1 = float(acc1)
            step = float(np.clip(step * np.exp(acc1 - config.acc_target), 1e-4, 50.0))
            pos, acc2 = mala_round(
                k2, pos, params, new_lmbda, step, sigma, n_steps=n_rest
            )
            acc2 = float(acc2)
            step = float(np.clip(step * np.exp(acc2 - config.acc_target), 1e-4, 50.0))
            n_evals += n * (config.n_mcmc + 2)

            _, ll_j, _ = batch_eval(pos, params, new_lmbda)
            ll = np.asarray(ll_j)
            ll = np.where(np.isfinite(ll), ll, -np.inf)

            bridge = {
                "lmbda": float(new_lmbda),
                "dl": float(dl),
                "ess_frac": ess_frac,
                "log_z_inc": log_z_inc,
                "acc": 0.5 * (acc1 + acc2),
                "acc_rounds": [acc1, acc2],
                "step": step,
                "n_unique_ancestors": int(np.unique(ancestors).size),
            }
            if bridge_stats_fn is not None:
                bridge.update(bridge_stats_fn(np.asarray(pos)))
            bridges.append(bridge)
            lmbda = new_lmbda
            if lmbda >= 1.0:
                break

        if abs(lmbda - 1.0) > 1e-7:
            raise RuntimeError(
                f"tempered SMC stopped at lmbda={lmbda:.6f} after "
                f"{config.max_bridges} bridges; refusing to return a partial bridge"
            )

        return SMCResult(
            positions=np.asarray(pos, np.float64),
            log_z=float(log_z),
            bridges=bridges,
            n_loglik_evals=int(n_evals),
        )
