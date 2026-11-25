"""Compiled HMC chains in the coordinates of their arm.

1 to ``L`` leapfrog steps with unit mass in the chain's own coordinates and a
Metropolis step. A constant metric and a learned map both enter through the
coordinates: the chain runs ``y`` and its potential is the target's, pulled back.

All functions act on one chain. ``group`` holds what the chains of one group
share; batches have axes ``(group, chain)``.
"""

from functools import partial

import jax
import jax.numpy as jnp

# Dual averaging of the step size (Hoffman and Gelman, 2014).
SHRINKAGE, OFFSET, DECAY = 0.05, 10.0, 0.75


def batch(function, shared=()):
    """Map a one-chain function over (group, chain); ``shared`` args are per group."""

    def inner_axes(count):
        return tuple(None if i in shared else 0 for i in range(count))

    def mapped(*args):
        inner = jax.vmap(function, in_axes=inner_axes(len(args)))
        return jax.vmap(inner)(*args)

    return mapped


def compiled(function, shared, static):
    """``batch`` under ``jit``, with the ``static`` keywords compiled in."""

    @partial(jax.jit, static_argnames=static)
    def call(*args, **keywords):
        return batch(partial(function, **keywords), shared)(*args)

    return call


def chain(init, transition, estimands):
    """The batched kernels of one transition: start, tune and run."""

    def tune(key, state, initial, acceptance, group, *, steps, thin):
        """Dual averaging towards ``acceptance`` over ``steps`` transitions.

        Returns the state, the averaged step size and the position after every
        ``thin`` transitions.
        """

        def one(carry, args):
            state, log_step, log_mean, gap = carry
            count, subkey = args
            state, probability, _ = transition(subkey, state, jnp.exp(log_step), group)
            weight = 1 / (count + OFFSET)
            gap = (1 - weight) * gap + weight * (acceptance - probability)
            log_step = jnp.log(10 * initial) - jnp.sqrt(count) / SHRINKAGE * gap
            power = count**-DECAY
            log_mean = power * log_step + (1 - power) * log_mean
            return (state, log_step, log_mean, gap), None

        def block(carry, args):
            carry, _ = jax.lax.scan(one, carry, args)
            return carry, carry[0][0]

        zero = jnp.zeros((), initial.dtype)
        shape = (steps // thin, thin)
        (state, _, log_mean, _), positions = jax.lax.scan(
            block,
            (state, jnp.log(initial), jnp.log(initial), zero),
            (
                jnp.arange(1, steps + 1, dtype=initial.dtype).reshape(shape),
                jax.random.split(key, shape),
            ),
        )
        return state, jnp.exp(log_mean), positions

    def run(key, state, step, group, *, steps):
        """Sums of the estimands over ``steps`` transitions, and accepted moves."""

        def one(carry, subkey):
            state, sums, accepts = carry
            state, _, accepted = transition(subkey, state, step, group)
            return (state, sums + estimands(state[0], group), accepts + accepted), None

        zero = jnp.zeros_like(estimands(state[0], group))
        (state, sums, accepts), _ = jax.lax.scan(
            one, (state, zero, jnp.zeros((), jnp.int64)), jax.random.split(key, steps)
        )
        return state, sums, accepts

    return {
        "init": jax.jit(batch(init, shared=(1,))),
        "tune": compiled(tune, shared=(4,), static=("steps", "thin")),
        "run": compiled(run, shared=(3,), static=("steps",)),
        "transition": transition,
    }


def step_size(key, step, jitter, dtype):
    return step * jax.random.uniform(
        key, minval=1 - jitter, maxval=1 + jitter, dtype=dtype
    )


def decide(key, state, proposed, change):
    """Metropolis step for an energy change; a change that is not finite rejects."""
    probability = jnp.where(
        jnp.isfinite(change), jnp.exp(jnp.minimum(0.0, -change)), 0.0
    )
    accepted = jax.random.uniform(key, dtype=change.dtype) < probability
    state = jax.tree.map(
        lambda new, old: jnp.where(accepted, new, old), proposed, state
    )
    return state, probability, accepted


def hmc(potential, estimands, length, jitter):
    """HMC with unit mass for ``potential(y, group)`` and ``estimands(y, group)``.

    A transition takes 1 to ``length`` leapfrog steps, drawn uniformly, so that
    no trajectory length resonates with the target. It computes ``length`` steps
    and keeps the drawn number; only those count as gradients.
    """
    value_and_grad = jax.value_and_grad(potential)

    def init(y, group):
        return y, *value_and_grad(y, group)

    def transition(key, state, step, group):
        position, energy, gradient = state
        momentum_key, jitter_key, count_key, accept_key = jax.random.split(key, 4)
        start = jax.random.normal(momentum_key, position.shape, position.dtype)
        size = step_size(jitter_key, step, jitter, position.dtype)
        count = jax.random.randint(count_key, (), 1, length + 1)

        def leap(carry, index):
            y, p, _, _ = carry
            y = y + size * p
            value, slope = value_and_grad(y, group)
            moved = (y, p - size * slope, value, slope)
            return jax.tree.map(
                lambda new, old: jnp.where(index < count, new, old), moved, carry
            ), None

        (y, p, value, slope), _ = jax.lax.scan(
            leap,
            (position, start - 0.5 * size * gradient, energy, gradient),
            jnp.arange(length),
        )
        p = p + 0.5 * size * slope
        change = value + 0.5 * p @ p - energy - 0.5 * start @ start
        return decide(accept_key, state, (y, value, slope), change)

    return chain(init, transition, estimands)
