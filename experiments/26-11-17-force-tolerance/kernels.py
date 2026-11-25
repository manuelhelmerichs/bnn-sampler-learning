"""Compiled chains that move with a surrogate force.

HMC: ``L`` leapfrog steps with the surrogate force, then a Metropolis step with
the exact potential. Leapfrog with any position-dependent force is a
volume-preserving involution after a momentum flip, so the chain keeps pi
exactly for every frozen field. With fresh noise the state also carries the
noisy force at its position, whose stationary law is the noise law; the move is
then an involution on positions, momenta and noises, and pi is kept as well. A
force error therefore costs acceptance, never bias.

ULA: the draft's unadjusted kinetic Langevin update
``v' = (1 - gamma) v - h F(q) + sqrt(2 gamma) xi``, ``q' = q + h v'``.

All functions act on one chain. ``group`` holds what the chains of one group
share (target and force parameters); batches have axes ``(group, chain)``.
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


def compiled(function, shared):
    """``batch`` under ``jit``, with the number of steps as a static keyword."""

    @partial(jax.jit, static_argnames=("steps",))
    def call(*args, steps):
        return batch(partial(function, steps=steps), shared)(*args)

    return call


def keys(key, shape):
    return jax.random.split(key, shape)


def hmc(potential, force, estimands, length, jitter):
    """Kernels of HMC with ``length`` leapfrog steps and jittered step sizes.

    ``potential(u, group)``, ``force(u, key, group)`` and ``estimands(u, group)``
    act on one chain.
    """

    def init(u, key, group):
        return u, potential(u, group), force(u, key, group)

    def transition(key, state, step, group):
        position, energy, current = state
        momentum_key, jitter_key, accept_key, force_key = jax.random.split(key, 4)
        start = jax.random.normal(momentum_key, position.shape, position.dtype)
        size = step * jax.random.uniform(
            jitter_key, minval=1 - jitter, maxval=1 + jitter, dtype=position.dtype
        )

        def leap(carry, subkey):
            u, p, _ = carry
            u = u + size * p
            f = force(u, subkey, group)
            return (u, p - size * f, f), None

        (u, p, f), _ = jax.lax.scan(
            leap,
            (position, start - 0.5 * size * current, current),
            jax.random.split(force_key, length),
        )
        p = p + 0.5 * size * f
        proposed = potential(u, group)
        change = proposed + 0.5 * p @ p - energy - 0.5 * start @ start
        probability = jnp.where(
            jnp.isfinite(change), jnp.exp(jnp.minimum(0.0, -change)), 0.0
        )
        accepted = jax.random.uniform(accept_key, dtype=position.dtype) < probability
        state = jax.tree.map(
            lambda new, old: jnp.where(accepted, new, old),
            (u, proposed, f),
            state,
        )
        return state, probability, accepted

    def adapt(key, state, initial, acceptance, group, *, steps):
        """Dual averaging towards ``acceptance``; returns the averaged step size."""

        def step(carry, args):
            state, log_step, log_mean, gap = carry
            count, subkey = args
            state, probability, _ = transition(subkey, state, jnp.exp(log_step), group)
            weight = 1 / (count + OFFSET)
            gap = (1 - weight) * gap + weight * (acceptance - probability)
            log_step = jnp.log(10 * initial) - jnp.sqrt(count) / SHRINKAGE * gap
            power = count**-DECAY
            log_mean = power * log_step + (1 - power) * log_mean
            return (state, log_step, log_mean, gap), None

        zero = jnp.zeros((), initial.dtype)
        (state, _, log_mean, _), _ = jax.lax.scan(
            step,
            (state, jnp.log(initial), jnp.log(initial), zero),
            (
                jnp.arange(1, steps + 1, dtype=initial.dtype),
                jax.random.split(key, steps),
            ),
        )
        return state, jnp.exp(log_mean)

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

    def trace(key, state, step, group, *, steps):
        """The estimands after every transition."""

        def one(state, subkey):
            state, _, _ = transition(subkey, state, step, group)
            return state, estimands(state[0], group)

        return jax.lax.scan(one, state, jax.random.split(key, steps))

    def record(key, state, step, group, *, steps):
        """Every leapfrog position and its force, for a corpus.

        Also returns whether each point's trajectory was accepted. A rejected
        trajectory can leave the target far behind, so a corpus keeps only the
        accepted ones.
        """

        def one(state, subkey):
            position, energy, current = state
            momentum_key, jitter_key, accept_key, force_key = jax.random.split(
                subkey, 4
            )
            start = jax.random.normal(momentum_key, position.shape, position.dtype)
            size = step * jax.random.uniform(
                jitter_key, minval=1 - jitter, maxval=1 + jitter, dtype=position.dtype
            )

            def leap(carry, k):
                u, p, _ = carry
                u = u + size * p
                f = force(u, k, group)
                return (u, p - size * f, f), (u, f)

            (u, p, f), visited = jax.lax.scan(
                leap,
                (position, start - 0.5 * size * current, current),
                jax.random.split(force_key, length),
            )
            p = p + 0.5 * size * f
            proposed = potential(u, group)
            change = proposed + 0.5 * p @ p - energy - 0.5 * start @ start
            accepted = jnp.log(jax.random.uniform(accept_key)) < -change
            state = jax.tree.map(
                lambda new, old: jnp.where(accepted, new, old), (u, proposed, f), state
            )
            return state, (*visited, jnp.broadcast_to(accepted, (length,)))

        state, (positions, forces, kept) = jax.lax.scan(
            one, state, jax.random.split(key, steps)
        )
        return (
            state,
            positions.reshape(-1, positions.shape[-1]),
            forces.reshape(-1, forces.shape[-1]),
            kept.reshape(-1),
        )

    return {
        "init": jax.jit(batch(init, shared=(2,))),
        "adapt": compiled(adapt, shared=(4,)),
        "run": compiled(run, shared=(3,)),
        "trace": compiled(trace, shared=(3,)),
        "record": compiled(record, shared=(3,)),
    }


def ula(force, estimands):
    """Kernels of the draft's unadjusted kinetic Langevin update."""

    def run(key, state, step, friction, group, *, steps):
        def one(carry, subkey):
            (q, v), sums = carry
            noise_key, force_key = jax.random.split(subkey)
            noise = jax.random.normal(noise_key, q.shape, q.dtype)
            v = (1 - friction) * v - step * force(q, force_key, group)
            v = v + jnp.sqrt(2 * friction) * noise
            q = q + step * v
            return ((q, v), sums + estimands(q, group)), None

        zero = jnp.zeros_like(estimands(state[0], group))
        (state, sums), _ = jax.lax.scan(
            one, (state, zero), jax.random.split(key, steps)
        )
        return state, sums

    return {"run": compiled(run, shared=(4,))}


def oracle_force(potential, fields):
    """``grad U + noise * xi + scale * B (u - c)``; ``fields`` adds the linear part."""
    from .errors import linear

    gradient = jax.grad(potential)

    def force(u, key, group):
        f = gradient(u, group) + group["noise"] * jax.random.normal(
            key, u.shape, u.dtype
        )
        if fields:
            f = f + group["scale"] * linear(u, group, jnp)
        return f

    return force


def oracle_potential(potential):
    return lambda u, group: potential(u, group["variances"])


def oracle_estimands(estimands):
    return lambda u, group: estimands(u, group["variances"])
