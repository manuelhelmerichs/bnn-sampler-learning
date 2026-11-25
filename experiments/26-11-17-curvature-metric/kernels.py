"""MCLMC and NUTS under a constant metric, as samplers of whitened coordinates.

Both run ``z`` with ``theta = center + W z`` and no metric of their own, which
is the sampler of ``theta`` under the metric ``(W W')^-1``. Work is counted in
gradients of the potential.
"""

import jax
import jax.numpy as jnp
import numpy as np

from .posterior import apply


def _logdensity(potential, center, factor):
    return lambda z: -potential(center + apply(factor, z))


def mclmc(potential, settings):
    """Unadjusted MCLMC as the campaigns run it, for a batch of chains.

    ``tune`` is the warmup of ``sample``: step size from the energy error,
    momentum decoherence length from the autocorrelation. ``block`` advances
    every chain by ``thin`` steps and returns its position, its potential, the
    mean squared energy error and the first two moments of ``z`` over the steps.
    Records keep that error per dimension as ``energy`` and the number of
    undone steps as ``failures``.
    """
    import blackjax.mcmc as mcmc
    from bnn.samplers.mclmc import warmup

    def tune(center, factor, key, step_size):
        state, parameters = warmup(
            _logdensity(potential, center, factor),
            jnp.zeros_like(center),
            key,
            settings["tuning_steps"],
            diagonal_preconditioning=False,
            desired_energy_var_start=settings["energy_variance"][0],
            desired_energy_var_end=settings["energy_variance"][1],
            trust_in_estimate=1.5,
            num_effective_samples=100,
            step_size_init=step_size,
        )
        return state, parameters.L, parameters.step_size

    def block(center, factor, state, length, step_size, key):
        kernel = mcmc.mclmc.build_kernel(
            logdensity_fn=_logdensity(potential, center, factor),
            integrator=mcmc.integrators.isokinetic_mclachlan,
            sqrt_diag_cov=jnp.ones_like(center),
        )

        def step(carry, key):
            state, first, second = carry
            moved, info = kernel(key, state, L=length, step_size=step_size)
            # A step that leaves the finite numbers is undone, as in the warmup.
            finite = jnp.isfinite(info.energy_change) & jnp.all(
                jnp.isfinite(moved.position)
            )
            state = jax.tree.map(
                lambda new, old: jnp.where(finite, new, old), moved, state
            )
            carry = (state, first + state.position, second + state.position**2)
            return carry, (jnp.where(finite, info.energy_change**2, 0.0), ~finite)

        zeros = jnp.zeros_like(center)
        (state, first, second), (errors, failed) = jax.lax.scan(
            step, (state, zeros, zeros), jax.random.split(key, settings["thin"])
        )
        position = center + apply(factor, state.position)
        readings = (-state.logdensity, jnp.mean(errors), jnp.sum(failed))
        return state, position, readings, first, second

    return jax.jit(jax.vmap(tune)), jax.jit(jax.vmap(block))


def run_mclmc(functions, centers, factors, step_sizes, budget, key, settings):
    """One segment of every chain: tune, then sample until ``budget`` gradients.

    Returns one row per chain: the positions and readings at each record, the
    tuned parameters, the variance of the positions over all sampled steps
    (diagonal factors only) and the gradients used.
    """
    tune, block = functions
    chains = len(centers)
    tune_key, key = jax.random.split(key)
    state, length, step_size = tune(
        centers, factors, jax.random.split(tune_key, chains), step_sizes
    )
    used = 2 * settings["tuning_steps"]
    blocks = max(1, (int(np.min(budget)) - used) // (2 * settings["thin"]))
    names = ("position", "potential", "gradients", "energy", "failures")
    records = {name: [] for name in names}
    first, second = 0.0, 0.0
    for block_key in jax.random.split(key, blocks):
        state, position, readings, block_first, block_second = block(
            centers,
            factors,
            state,
            length,
            step_size,
            jax.random.split(block_key, chains),
        )
        potential, error, failed = readings
        used += 2 * settings["thin"]
        records["position"].append(np.asarray(position))
        records["potential"].append(np.asarray(potential))
        records["gradients"].append(np.full(chains, used))
        records["energy"].append(np.asarray(error) / centers.shape[1])
        records["failures"].append(np.asarray(failed))
        first, second = (
            first + np.asarray(block_first),
            second + np.asarray(block_second),
        )
    steps = blocks * settings["thin"]
    spread = second / steps - (first / steps) ** 2
    factors = np.asarray(factors)
    stacked = {name: np.stack(rows, axis=1) for name, rows in records.items()}
    return [
        {
            **{name: values[chain] for name, values in stacked.items()},
            "step_size": float(step_size[chain]),
            "length": float(length[chain]),
            "energy_variance": float(stacked["energy"][chain].mean()),
            "variance": factors[chain] ** 2 * spread[chain]
            if factors.ndim == 2
            else None,
            "count": steps,
            "used": used,
        }
        for chain in range(chains)
    ]


def nuts(potential, settings):
    """NUTS transitions and the dual averaging of their step size.

    ``tune`` starts dual averaging towards the target acceptance on trees of at
    most ``tuning_depth`` doublings, ``transition`` is one NUTS transition at a
    given step size and ``update`` continues the averaging with its acceptance.
    """
    import blackjax
    from blackjax.adaptation.step_size import dual_averaging_adaptation

    kernel = blackjax.nuts.build_kernel()
    start, update, _ = dual_averaging_adaptation(settings["acceptance"])

    def tune(center, factor, position, step_size, key):
        logdensity = _logdensity(potential, center, factor)
        ones = jnp.ones_like(center)

        def step(carry, key):
            state, adaptation = carry
            state, info = kernel(
                key,
                state,
                logdensity,
                jnp.exp(adaptation.log_step_size),
                ones,
                max_num_doublings=settings["tuning_depth"],
            )
            adaptation = update(adaptation, info.acceptance_rate)
            return (state, adaptation), info.num_integration_steps

        (state, adaptation), leapfrogs = jax.lax.scan(
            step,
            (blackjax.nuts.init(position, logdensity), start(step_size)),
            jax.random.split(key, settings["tuning_transitions"]),
        )
        return state.position, adaptation, jnp.sum(leapfrogs) + 1

    def transition(center, factor, position, step_size, key):
        logdensity = _logdensity(potential, center, factor)
        state, info = kernel(
            key,
            blackjax.nuts.init(position, logdensity),
            logdensity,
            step_size,
            jnp.ones_like(center),
            max_num_doublings=settings["depth"],
        )
        return (
            state.position,
            center + apply(factor, state.position),
            -state.logdensity,
            info.num_integration_steps + 1,
            info.acceptance_rate,
            info.is_divergent,
        )

    return jax.jit(tune), jax.jit(transition), jax.jit(update)


def run_nuts(functions, centers, factors, step_sizes, budget, key, settings):
    """One segment of every chain, one chain after the other.

    Chains use different numbers of gradients per transition, so each runs
    until its own count reaches ``budget``. The step size keeps adapting over
    the first third of the gradients and is frozen at its average afterwards;
    only then is the kernel exact. Records are per transition.
    """
    tune, transition, update = functions
    rows = []
    for center, factor, step_size, limit, chain_key in zip(
        centers,
        factors,
        step_sizes,
        np.asarray(budget),
        jax.random.split(key, len(centers)),
        strict=True,
    ):
        tune_key, chain_key = jax.random.split(chain_key)
        position, adaptation, used = tune(
            center, factor, jnp.zeros_like(center), step_size, tune_key
        )
        used = int(used)
        adapting_until = used + (limit - used) / 3
        names = ("position", "potential", "gradients", "acceptance", "frozen")
        row = {name: [] for name in names}
        divergences, leapfrogs = 0, []
        while used < limit or not row["position"]:
            frozen = used >= adapting_until
            step_size = jnp.exp(
                adaptation.log_step_size_avg if frozen else adaptation.log_step_size
            )
            chain_key, step_key = jax.random.split(chain_key)
            position, theta, potential, count, accepted, divergent = transition(
                center, factor, position, step_size, step_key
            )
            if not frozen:
                adaptation = update(adaptation, accepted)
            used += int(count)
            row["position"].append(np.asarray(theta))
            row["potential"].append(float(potential))
            row["gradients"].append(used)
            row["acceptance"].append(float(accepted))
            row["frozen"].append(int(frozen))
            leapfrogs.append(int(count) - 1)
            divergences += int(divergent)
        row = {name: np.array(values) for name, values in row.items()}
        kept = row["frozen"] == 1 if row["frozen"].any() else row["frozen"] == 0
        rows.append(
            {
                **row,
                "step_size": float(jnp.exp(adaptation.log_step_size_avg)),
                # Mean acceptance and tree size at the frozen step size.
                "mean_acceptance": float(row["acceptance"][kept].mean()),
                "leapfrogs": float(np.asarray(leapfrogs)[kept].mean()),
                "divergences": divergences,
                "variance": row["position"].var(axis=0),
                "count": len(row["position"]),
                "used": used,
            }
        )
    return rows
