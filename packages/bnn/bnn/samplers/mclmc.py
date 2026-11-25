"""Microcanonical Langevin Monte Carlo with the MILE adaptation.

The adaptation follows BlackJAX 1.2.2 with changes for Bayesian neural
networks: a scheduled energy-error target, weighted Welford moments, and an
effective-sample-size estimate of the trajectory length from a bounded trace.
Each stage of the adaptation is one phase, so warmup pauses and resumes like
sampling does.
"""

from __future__ import annotations

from typing import Literal, NamedTuple

import blackjax.mcmc as mcmc
import jax
import jax.numpy as jnp
from blackjax.adaptation.mclmc_adaptation import MCLMCAdaptationState
from blackjax.diagnostics import effective_sample_size
from blackjax.mcmc.integrators import IntegratorState
from blackjax.util import pytree_size
from jax.flatten_util import ravel_pytree
from pydantic import Field, PositiveFloat, PositiveInt, model_validator

from bnn.engine import Phase
from bnn.samplers.base import SAMPLES, SamplerConfig
from bnn.trees import tree_all_finite

TRACE_COORDINATES = 2000
TRACE_SAMPLES = 10000
L_FACTOR = 0.4


def handle_nans(previous_state, next_state, step_size, step_size_max, kinetic_change):
    """Reject a nonfinite transition and bound the step size below the failed one."""
    finite = tree_all_finite(next_state) & jnp.isfinite(kinetic_change)
    state = jax.tree.map(
        lambda new, old: jnp.where(finite, new, old), next_state, previous_state
    )
    maximum = jnp.where(finite, step_size_max, step_size * 0.8)
    return finite, state, maximum, jnp.where(finite, kinetic_change, 0.0)


class Tuning(NamedTuple):
    """Carry of the step-size and covariance stages."""

    state: IntegratorState
    params: MCLMCAdaptationState
    log_time: jax.Array
    log_x_average: jax.Array
    step_size_max: jax.Array
    weight: jax.Array
    moments: jax.Array


class Tuned(NamedTuple):
    """A chain state with the parameters its kernel uses next."""

    state: IntegratorState
    params: MCLMCAdaptationState
    final_L: jax.Array


class Trace(NamedTuple):
    """Carry of the trajectory-length stage: a bounded coordinate trace."""

    state: IntegratorState
    params: MCLMCAdaptationState
    coordinates: jax.Array
    samples: jax.Array


class Sampling(NamedTuple):
    """Carry of the production chain."""

    state: IntegratorState
    params: MCLMCAdaptationState


@SamplerConfig.member("mclmc")
class MCLMCConfig(SamplerConfig):
    """Unadjusted microcanonical Langevin dynamics on the full data."""

    name: Literal["mclmc"] = "mclmc"
    step_size_init: PositiveFloat = 0.005
    desired_energy_var_start: PositiveFloat = Field(
        default=5e-4, description="Energy variance target at the start of warmup."
    )
    desired_energy_var_end: PositiveFloat = Field(
        default=1e-4, description="Energy variance target at the end of warmup."
    )
    trust_in_estimate: PositiveFloat = 1.5
    num_effective_samples: PositiveInt = 100
    diagonal_preconditioning: bool = False

    @model_validator(mode="after")
    def _validate(self) -> MCLMCConfig:
        if self.warmup < 20:
            raise ValueError("mclmc needs at least 20 warmup transitions.")
        if self.keep_warmup:
            raise ValueError("mclmc does not store warmup positions.")
        return self

    def phases(self, target, *, step_size_init=None) -> tuple[Phase, ...]:
        """Step size and covariance, optional readjustment, length, sampling.

        ``step_size_init`` replaces the configured initial step size and may be
        a traced value.
        """
        if step_size_init is None:
            step_size_init = self.step_size_init
        covariance_steps = max(2, self.warmup // 10)
        trace_steps = max(2, self.warmup // 10)
        tuning_steps = self.warmup - covariance_steps - trace_steps
        total = tuning_steps + covariance_steps + 1
        decay = (self.num_effective_samples - 1.0) / (self.num_effective_samples + 1.0)

        def kernel(constants, sqrt_diag_cov):
            return mcmc.mclmc.build_kernel(
                logdensity_fn=lambda position: target.log_density(position, constants),
                integrator=mcmc.integrators.isokinetic_mclachlan,
                sqrt_diag_cov=sqrt_diag_cov,
            )

        def energy_target(step):
            progress = jnp.minimum(step / total, 1.0)
            return self.desired_energy_var_start - progress * (
                self.desired_energy_var_start - self.desired_energy_var_end
            )

        def tuning_step(carry: Tuning, x, constants, averaging):
            dim = carry.params.sqrt_diag_cov.shape[0]
            proposal, info = kernel(constants, carry.params.sqrt_diag_cov)(
                rng_key=x["key"],
                state=carry.state,
                L=carry.params.L,
                step_size=carry.params.step_size,
            )
            success, state, step_size_max, energy_change = handle_nans(
                carry.state,
                proposal,
                carry.params.step_size,
                carry.step_size_max,
                info.energy_change,
            )
            log_xi = jnp.logaddexp(
                2 * jnp.log(jnp.abs(energy_change))
                - jnp.log(dim * energy_target(x["index"])),
                jnp.log(1e-8),
            )
            # Both weighted sums stay in log space: h**6 underflows in float32.
            log_weight = -0.5 * jnp.square(log_xi / (6.0 * self.trust_in_estimate))
            log_x_average = jnp.logaddexp(
                jnp.log(decay) + carry.log_x_average,
                log_weight + log_xi - 6 * jnp.log(carry.params.step_size),
            )
            log_time = jnp.logaddexp(jnp.log(decay) + carry.log_time, log_weight)
            step_size = jnp.exp((log_time - log_x_average) / 6.0)
            step_size = jnp.minimum(step_size, step_size_max)
            step_size = jnp.maximum(step_size, jnp.finfo(step_size.dtype).tiny)
            params = carry.params._replace(step_size=step_size)

            # Weighted Welford accumulation avoids cancellation in E[x^2] - E[x]^2.
            flat = ravel_pytree(state.position)[0]
            weight = averaging * success * step_size
            total_weight = carry.weight + weight
            delta = flat - carry.moments[0]
            mean = carry.moments[0] + delta * weight / jnp.maximum(total_weight, 1e-30)
            second = carry.moments[1] + weight * delta * (flat - mean)
            return Tuning(
                state,
                params,
                log_time,
                log_x_average,
                step_size_max,
                total_weight,
                jnp.stack((mean, second)),
            ), {}

        def fresh(state, params) -> Tuning:
            dim = params.sqrt_diag_cov.shape[0]
            dtype = params.sqrt_diag_cov.dtype
            return Tuning(
                state,
                params,
                jnp.asarray(-jnp.inf, dtype),
                jnp.asarray(-jnp.inf, dtype),
                jnp.asarray(jnp.inf, dtype),
                jnp.zeros((), dtype),
                jnp.zeros((2, dim), dtype),
            )

        def start_tuning(position, key, constants) -> Tuning:
            state = mcmc.mclmc.init(
                position=position,
                logdensity_fn=lambda value: target.log_density(value, constants),
                rng_key=key,
            )
            dim = pytree_size(position)
            dtype = ravel_pytree(position)[0].dtype
            params = MCLMCAdaptationState(
                L=jnp.maximum(jnp.sqrt(dim), 15.0).astype(dtype),
                step_size=jnp.asarray(step_size_init, dtype),
                sqrt_diag_cov=jnp.ones((dim,), dtype),
            )
            return fresh(state, params)

        def finish_tuning(carry: Tuning, constants) -> Tuned:
            dim = carry.params.sqrt_diag_cov.shape[0]
            variances = jnp.maximum(
                carry.moments[1] / jnp.maximum(carry.weight, 1e-30), 1e-8
            )
            if self.diagonal_preconditioning:
                params = carry.params._replace(sqrt_diag_cov=jnp.sqrt(variances))
                final_L = jnp.sqrt(dim).astype(variances.dtype)
            else:
                params = carry.params
                final_L = jnp.sqrt(jnp.sum(variances))
            return Tuned(carry.state, params, final_L)

        phases = [
            Phase(
                "step_size",
                tuning_steps + covariance_steps,
                start_tuning,
                lambda carry, x, constants: tuning_step(
                    carry,
                    x,
                    constants,
                    (x["index"] >= tuning_steps).astype(jnp.float32),
                ),
                finish_tuning,
                check=lambda carry: carry.state,
            )
        ]
        if self.diagonal_preconditioning:
            # The step size is readjusted to the preconditioned dynamics.
            phases.append(
                Phase(
                    "readjust",
                    covariance_steps // 3,
                    lambda tuned, key, constants: (
                        fresh(tuned.state, tuned.params),
                        tuned.final_L,
                    ),
                    lambda carry, x, constants: (
                        (tuning_step(carry[0], x, constants, 0.0)[0], carry[1]),
                        {},
                    ),
                    lambda carry, constants: Tuned(
                        carry[0].state, carry[0].params, carry[1]
                    ),
                    check=lambda carry: carry[0].state,
                )
            )

        stride = max(1, (trace_steps + TRACE_SAMPLES - 1) // TRACE_SAMPLES)
        stored = (trace_steps + stride - 1) // stride

        def start_trace(tuned: Tuned, key, constants) -> Trace:
            flat = ravel_pytree(tuned.state.position)[0]
            count = min(flat.shape[0], TRACE_COORDINATES)
            offset = jax.random.randint(key, (), 0, flat.shape[0])
            coordinates = (
                jnp.linspace(0, flat.shape[0] - 1, count).astype(jnp.int32) + offset
            ) % flat.shape[0]
            return Trace(
                tuned.state,
                tuned.params._replace(L=tuned.final_L),
                coordinates,
                jnp.zeros((stored + 1, count), flat.dtype),
            )

        def trace_step(carry: Trace, x, constants):
            state, _ = kernel(constants, carry.params.sqrt_diag_cov)(
                rng_key=x["key"],
                state=carry.state,
                L=carry.params.L,
                step_size=carry.params.step_size,
            )
            # Transitions between retained times write to the spare last row.
            row = jnp.where(x["index"] % stride == 0, x["index"] // stride, stored)
            values = ravel_pytree(state.position)[0][carry.coordinates]
            return carry._replace(
                state=state, samples=carry.samples.at[row].set(values)
            ), {}

        def finish_trace(carry: Trace, constants) -> Sampling:
            samples = carry.samples[:stored]
            ess = effective_sample_size(samples[None, ...])
            valid = jnp.isfinite(ess) & (ess > 0)
            span = (stored - 1) * stride + 1
            estimate = (
                L_FACTOR
                * carry.params.step_size
                * jnp.sum(jnp.where(valid, span / ess, 0))
                / jnp.maximum(jnp.sum(valid), 1)
            )
            length = jnp.where(
                jnp.any(valid) & (estimate > 0), estimate, carry.params.L
            )
            return Sampling(carry.state, carry.params._replace(L=length))

        def sample_step(carry: Sampling, x, constants):
            state, info = kernel(constants, carry.params.sqrt_diag_cov)(
                rng_key=x["key"],
                state=carry.state,
                L=carry.params.L,
                step_size=carry.params.step_size,
            )
            return Sampling(state, carry.params), {
                "energy_change": info.energy_change,
                "is_divergent": ~tree_all_finite(state.position),
            }

        phases += [
            Phase(
                "trajectory_length",
                trace_steps,
                start_trace,
                trace_step,
                finish_trace,
                check=lambda carry: carry.state,
            ),
            Phase(
                "sample",
                self.steps,
                lambda sampling, key, constants: sampling,
                sample_step,
                position=lambda carry: carry.state.position,
                keep=self.keep(),
                store=SAMPLES,
                must_move=True,
                check=lambda carry: carry.state,
            ),
        ]
        return tuple(phases)

    @staticmethod
    def parameters(sampling: Sampling) -> dict:
        """The adapted kernel parameters of a chain."""
        return {
            "L": sampling.params.L,
            "step_size": sampling.params.step_size,
            "sqrt_diag_cov": sampling.params.sqrt_diag_cov,
        }


def warmup(logdensity_fn, position, key, steps: int, *, step_size_init=None, **options):
    """Adapt MCLMC to an arbitrary log density, outside of a run.

    ``options`` are the adaptation options of :class:`MCLMCConfig`;
    ``step_size_init`` may be a traced value. Returns the adapted chain state
    and its ``L``, ``step_size`` and ``sqrt_diag_cov``.
    """
    from types import SimpleNamespace

    from bnn.engine import simulate

    config = MCLMCConfig(warmup=steps, steps=1, **options)
    target = SimpleNamespace(log_density=lambda value, constants: logdensity_fn(value))
    phases = config.phases(target, step_size_init=step_size_init)[:-1]
    sampling = simulate(phases, position, key)
    return sampling.state, sampling.params
