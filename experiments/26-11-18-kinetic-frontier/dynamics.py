"""Unadjusted samplers of one trial, and the kinetic fields a learner may change.

A trial runs ``chains`` chains of one task from standard normal positions in
aligned coordinates, which put half of them on either side of the barrier
whatever the mode weights are. Every sampler returns the same row of results.

The learned sampler is the update of Kim et al. (2024),

    r     <- r - eps [grad U + alpha + C beta(theta, r)] + sqrt(2 C eps T) xi
    theta <- theta + eps beta(theta, r),

with coordinatewise fields ``alpha`` and ``beta`` of the position, the
momentum, the gradient and running averages of the gradient. ``alpha = 0`` and
``beta = r`` is SGHMC by the symplectic Euler method.
"""

import blackjax.mcmc as mcmc
import jax
import jax.numpy as jnp
import numpy as np

from . import targets

DECAYS = (0.1, 0.5, 0.9, 0.99, 0.999, 0.9999)
FEATURES = 3 + len(DECAYS)
HIDDEN = 32
# Results of a trial: the scalars of ``targets``, the mean squared error and the
# mean of the second moments off the axis, mode changes per chain and step, and
# whether every number was finite.
RESULTS = (*targets.SCALARS, "local", "second", "switches", "finite")
# Errors of a trial are clipped here, so that a diverged trial stays in.
MASS_CLIP, LOCAL_CLIP = 1.0, 10.0


# ----------------------------------------------------------------------------
# Fields


def network(key, inputs=FEATURES):
    """Two hidden layers of 32 units; the output layer starts at zero."""
    k1, k2 = jax.random.split(key)
    return {
        "w1": jax.random.normal(k1, (inputs, HIDDEN)) / np.sqrt(inputs),
        "b1": jnp.zeros(HIDDEN),
        "w2": jax.random.normal(k2, (HIDDEN, HIDDEN)) / np.sqrt(HIDDEN),
        "b2": jnp.zeros(HIDDEN),
        "w3": jnp.zeros((HIDDEN, 2)),
        "b3": jnp.zeros(2),
    }


def apply(net, features):
    hidden = jax.nn.relu(features @ net["w1"] + net["b1"])
    hidden = jax.nn.relu(hidden @ net["w2"] + net["b2"])
    return hidden @ net["w3"] + net["b3"]


def features(theta, r, g, averages, normalized):
    stack = jnp.concatenate(
        [g[..., None], theta[..., None], r[..., None], averages], axis=-1
    )
    if normalized:
        # As published: every feature has a unit second moment over coordinates.
        stack = stack * jax.lax.rsqrt(1e-5 + jnp.mean(stack**2, axis=-2, keepdims=True))
    return stack


def fields(form, parameters, theta, r, g, averages):
    """``alpha`` and ``beta`` of one form.

    - classical: SGHMC.
    - residual: SGHMC plus a network of the raw features.
    - published: a network of normalized features times learned scales, without
      a term that passes the momentum through.
    - linear: SGHMC plus a linear map of the raw features.
    """
    if form == "classical":
        return jnp.zeros_like(r), r
    if form == "linear":
        out = features(theta, r, g, averages, False) @ parameters["weight"]
        out = out + parameters["bias"]
        return out[..., 0], r + out[..., 1]
    out = apply(parameters["net"], features(theta, r, g, averages, form == "published"))
    if form == "published":
        scales = jnp.exp(parameters["scales"])
        return scales[0] * out[..., 0], scales[1] * out[..., 1]
    return out[..., 0], r + out[..., 1]


# ----------------------------------------------------------------------------
# Samplers: ``start(key, theta, task)`` and ``step(key, state, task)`` of all
# chains of a trial, with the position first in the state.


def euler(form, parameters, force=targets.force):
    step_size, friction, temperature = jnp.exp(parameters["hyper"])
    noise = jnp.sqrt(2 * friction * step_size * temperature)

    def start(key, theta, task):
        r = jnp.sqrt(temperature) * jax.random.normal(key, theta.shape, theta.dtype)
        return theta, r, jnp.zeros((*theta.shape, len(DECAYS)), theta.dtype)

    def step(key, state, task):
        theta, r, averages = state
        g = force(theta, task)
        decays = jnp.asarray(DECAYS, theta.dtype)
        averages = decays * averages + (1 - decays) * g[..., None]
        alpha, beta = fields(form, parameters, theta, r, g, averages)
        r = r - step_size * (g + alpha + friction * beta)
        r = r + noise * jax.random.normal(key, r.shape, r.dtype)
        _, beta = fields(form, parameters, theta, r, g, averages)
        return theta + step_size * beta, r, averages

    return start, step


def baoab(parameters, force=targets.force):
    """Kinetic Langevin dynamics by the BAOAB splitting."""
    step_size, friction, temperature = jnp.exp(parameters["hyper"])
    decay = jnp.exp(-friction * step_size)
    noise = jnp.sqrt(temperature * (1 - decay**2))

    def start(key, theta, task):
        r = jnp.sqrt(temperature) * jax.random.normal(key, theta.shape, theta.dtype)
        return theta, r, force(theta, task)

    def step(key, state, task):
        theta, r, g = state
        r = r - 0.5 * step_size * g
        theta = theta + 0.5 * step_size * r
        r = decay * r + noise * jax.random.normal(key, r.shape, r.dtype)
        theta = theta + 0.5 * step_size * r
        g = force(theta, task)
        return theta, r - 0.5 * step_size * g, g

    return start, step


def bridge(parameters, force=targets.force, reference=targets.reference_force):
    """BAOAB on ``pi^(1 - c) pi_0^c`` between the target and its reference.

    The fourth number is the share ``c``. The reference is known wherever a
    prior is, and its force replaces that share of the target's: the modes move
    towards each other and the barrier sinks, in whatever frame.
    """
    share = jnp.exp(parameters["hyper"][3])
    return baoab(
        {"hyper": parameters["hyper"][:3]},
        lambda theta, task: (
            (1 - share) * force(theta, task) + share * reference(theta, task)
        ),
    )


def mclmc(parameters):
    """Unadjusted microcanonical Langevin dynamics as MILE runs it.

    ``hyper`` holds the logarithms of the step size and of the decoherence
    length. A step of the minimal-norm integrator takes two gradients.
    """
    step_size, length = jnp.exp(parameters["hyper"])

    def kernel(task):
        return mcmc.mclmc.build_kernel(
            logdensity_fn=lambda x: -targets.potential(x, task),
            integrator=mcmc.integrators.isokinetic_mclachlan,
            sqrt_diag_cov=jnp.ones(targets.dimension(task), task["location"].dtype),
        )

    def start(key, theta, task):
        return jax.vmap(
            lambda k, x: mcmc.mclmc.init(x, lambda y: -targets.potential(y, task), k)
        )(jax.random.split(key, len(theta)), theta)

    def step(key, state, task):
        return jax.vmap(lambda k, s: kernel(task)(k, s, length, step_size)[0])(
            jax.random.split(key, len(state.position)), state
        )

    return start, step


def trial(sampler, task, key, *, chains, steps, burn):
    """Run one trial and reduce it to ``RESULTS``."""
    start, step = sampler
    d = targets.dimension(task)
    k1, k2, k3 = jax.random.split(key, 3)
    z = jax.random.normal(k1, (chains, d), task["location"].dtype)
    state = start(k2, targets.embedded(z, task), task)

    def position(state):
        return state[0]

    def one(carry, inputs):
        state, mode, sums, switches = carry
        key, keep = inputs
        state = step(key, state, task)
        theta = position(state)
        new = targets.core(theta, task, mode)
        switches = switches + keep * jnp.mean((new != mode) & (mode != 0))
        sums = sums + keep * jnp.mean(targets.statistics(theta, task), axis=0)
        return (state, new, sums, switches), None

    mode = targets.core(position(state), task, jnp.zeros(chains, jnp.int32))
    count = len(targets.SCALARS)
    keep = (jnp.arange(steps) >= burn).astype(z.dtype)
    (_, _, sums, switches), _ = jax.lax.scan(
        one,
        (state, mode, jnp.zeros(count + d - 1, z.dtype), jnp.zeros((), z.dtype)),
        (jax.random.split(k3, steps), keep),
    )
    kept = steps - burn
    return row(sums / kept, switches / kept)


def row(means, switches):
    """``RESULTS`` from the means of ``targets.statistics`` and the mode changes."""
    count = len(targets.SCALARS)
    second = means[count:]
    values = jnp.concatenate(
        [
            means[:count],
            jnp.stack([jnp.mean((second - 1) ** 2), jnp.mean(second), switches]),
        ]
    )
    finite = jnp.all(jnp.isfinite(values))
    return jnp.concatenate(
        [jnp.where(finite, values, 0.0), finite[None].astype(values.dtype)]
    )


def errors(rows, truth):
    """Clipped squared errors of the mass and of the second moments, per trial."""
    finite = rows[..., RESULTS.index("finite")] > 0
    mass = (rows[..., RESULTS.index("mass")] - truth) ** 2
    local = rows[..., RESULTS.index("local")]
    return (
        jnp.where(finite, jnp.minimum(mass, MASS_CLIP), MASS_CLIP),
        jnp.where(finite, jnp.minimum(local, LOCAL_CLIP), LOCAL_CLIP),
    )
