"""What a learned sampler's lead is made of, read across budgets.

A learned cell trains at one budget. Here its sampler runs from a quarter to
eight times that budget, next to two classical arms that are tuned again at
every budget:

- BAOAB over its whole grid, and restricted to temperature 1.
- BAOAB on the bridge ``pi^(1 - c) pi_0^c`` between the target and its
  reference ``pi_0``, which is the two modes merged at their midpoint. It is
  told the reference and nothing about the axis. Its draws are read raw, and
  reweighted to the target by ``(pi / pi_0)^c``, which makes them consistent.

A sampler whose error stops falling with the budget has bought its lead with a
bias.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from runner import CellSpec, Design, ResourceHint, repository_root

from . import dynamics, frontier, learning, targets

QUESTION = (
    "Does the lead of a learned sampler over classical ones persist as the "
    "budget grows, and does a classical sampler on the bridge to the target's "
    "reference match it?"
)
ROLES = frontier.ROLES
CONDITIONS = ((3.5, 8), (3.5, 32))
# The bridge's share of the reference, at step sizes of BAOAB.
BRIDGE = {
    "step": (0.7, 1.0, 1.4),
    "friction": (1.0,),
    "temperature": (1.0,),
    "share": (0.1, 0.2, 0.3, 0.4, 0.5),
}
ARMS = ("baoab", "baoab_unit", "bridge_raw", "bridge_reweighted", "learned")
EXPERIMENT = "26-11-18-kinetic-frontier"


def factors(profile, separation, dimension):
    smoke = profile == "smoke"
    return frontier.shared(profile, separation, dimension) | {
        "frame": "aligned",
        "form": "residual",
        "multiples": [0.5, 1] if smoke else [0.25, 0.5, 1, 2, 4, 8],
        "baoab": {k: list(v[:2] if smoke else v) for k, v in frontier.BAOAB.items()},
        "bridge": {k: list(v[:2] if smoke else v) for k, v in BRIDGE.items()},
        "transfer": [],
    }


def design():
    cells = []
    for profile in ("smoke", "pilot"):
        smoke = profile == "smoke"
        for separation, dimension in [(3.5, 8)] if smoke else CONDITIONS:
            seeds = {r: frontier.seed(profile, separation, dimension, r) for r in ROLES}
            inputs = {}
            if not smoke:
                learned = CellSpec(
                    "learned",
                    frontier.learned_factors(
                        profile, "residual", separation, dimension, "aligned"
                    ),
                    seeds,
                    role="pilot",
                )
                inputs["learned"] = {
                    "root": f"results/{EXPERIMENT}/{profile}/cells/{learned.cell_id}"
                }
            cells.append(
                CellSpec(
                    "budget",
                    factors(profile, separation, dimension),
                    seeds,
                    inputs=inputs,
                    role="pilot",
                    expected_schema=("metrics", "estimands"),
                    resources=ResourceHint(
                        memory_gb=4, walltime_minutes=frontier.WALLTIME[dimension]
                    ),
                )
            )
    return Design(QUESTION, ("smoke", "pilot"), ROLES, tuple(cells))


def bridged_trial(parameters, task, key, *, chains, steps, burn):
    """``RESULTS`` of the bridge's draws, raw and reweighted to the target."""
    start, step = dynamics.bridge(parameters)
    share = jnp.exp(parameters["hyper"][3])
    d = targets.dimension(task)
    k1, k2, k3 = jax.random.split(key, 3)
    z = jax.random.normal(k1, (chains, d), task["location"].dtype)
    state = start(k2, targets.embedded(z, task), task)
    excess = jax.vmap(targets.excess, in_axes=(0, None))
    # Relative to the midpoint, near the largest excess: weights of order one.
    top = targets.excess(task["location"], task)

    def one(carry, inputs):
        state, mode, sums, switches, weighted, total = carry
        key, keep = inputs
        state = step(key, state, task)
        theta = state[0]
        new = targets.core(theta, task, mode)
        switches = switches + keep * jnp.mean((new != mode) & (mode != 0))
        statistics = targets.statistics(theta, task)
        sums = sums + keep * jnp.mean(statistics, axis=0)
        weight = jnp.exp(share * (top - excess(theta, task)))
        weighted = weighted + keep * jnp.sum(weight[:, None] * statistics, axis=0)
        total = total + keep * jnp.sum(weight)
        return (state, new, sums, switches, weighted, total), None

    mode = targets.core(state[0], task, jnp.zeros(chains, jnp.int32))
    count = len(targets.SCALARS) + d - 1
    keep = (jnp.arange(steps) >= burn).astype(z.dtype)
    zero = jnp.zeros((), z.dtype)
    (_, _, sums, switches, weighted, total), _ = jax.lax.scan(
        one,
        (state, mode, jnp.zeros(count, z.dtype), zero, jnp.zeros(count, z.dtype), zero),
        (jax.random.split(k3, steps), keep),
    )
    kept = steps - burn
    return jnp.stack(
        [
            dynamics.row(sums / kept, switches / kept),
            dynamics.row(weighted / total, switches / kept),
        ]
    )


def measure(cell):
    spec, seeds = cell.factors, cell.seeds
    d = spec["dimension"]
    sets = frontier.task_sets(spec, seeds, spec["frame"])
    screen = frontier.as_arrays(sets["screen"])
    confirmation = frontier.as_arrays(sets["confirmation"])
    shape = (spec["targets"], spec["trials"])

    template = learning.initial(spec["form"], [1.0, 1.0, 1.0], jax.random.key(0))
    if "learned" in cell.inputs:
        stored = np.load(
            repository_root() / cell.inputs["learned"]["root"] / "learned.npz"
        )
        learned = ravel_pytree(template)[1](jnp.asarray(stored["parameters"]))
    else:
        # Smoke cells read no other cell: an untrained network at a mild setting.
        learned = template | {"hyper": jnp.log(jnp.asarray([0.3, 1.0, 1.0]))}

    names, settings = frontier.grid("baoab", spec["baoab"])
    hypers = jnp.log(jnp.asarray(settings, jnp.float32))
    unit = np.flatnonzero(settings[:, 2] == 1.0)
    bridge_names, bridge_settings = frontier.grid("bridge", spec["bridge"])
    bridge_hypers = jnp.log(jnp.asarray(bridge_settings, jnp.float32))
    budgets = []
    for m, multiple in enumerate(spec["multiples"]):
        steps = int(spec["steps"] * multiple)
        at = spec | {"steps": steps, "burn": steps // 4}
        entry = {"multiple": multiple, "steps": steps, "budget": spec["chains"] * steps}

        def keys(role, salt, shape_, m=m):
            return frontier.keys_for(seeds[role], 60, m, salt, shape=shape_)

        def read(rows, salt, m=m):
            return frontier.reduce(
                rows,
                sets["confirmation"],
                [seeds["bootstrap"], 60, m, salt],
                spec["bootstrap"],
            )

        # Classical BAOAB, tuned on the screen tasks at this budget.
        run = frontier.unadjusted("baoab", at)
        screened = run(hypers, screen, keys("screen", 0, (spec["screen_tasks"], 1)))
        mass, local = dynamics.errors(screened[:, :, 0], screen["mass"])
        loss = np.asarray(jnp.mean(mass + local, axis=1))
        for name, candidates in (
            ("baoab", np.arange(len(settings))),
            ("baoab_unit", unit),
        ):
            best = int(candidates[np.argmin(loss[candidates])])
            rows = np.asarray(
                run(hypers[best : best + 1], confirmation, keys("chain", 0, shape))
            )[0]
            entry[name] = {
                "setting": dict(zip(names, map(float, settings[best]), strict=True))
            }
            entry[name] |= read(rows, len(entry))

        # The same on the bridge to the reference.
        bridged = jax.jit(
            lambda hyper, tasks, keys_, at=at: jax.vmap(
                lambda task, row: jax.vmap(
                    lambda key: bridged_trial(
                        {"hyper": hyper},
                        task,
                        key,
                        chains=at["chains"],
                        steps=at["steps"],
                        burn=at["burn"],
                    )
                )(row)
            )(tasks, keys_)
        )
        screened = np.stack(
            [
                np.asarray(
                    bridged(h, screen, keys("screen", 1, (spec["screen_tasks"], 1)))
                )
                for h in bridge_hypers
            ]
        )
        for kind, name in enumerate(("bridge_raw", "bridge_reweighted")):
            mass, local = dynamics.errors(
                jnp.asarray(screened[:, :, 0, kind]), screen["mass"]
            )
            best = int(np.argmin(np.asarray(jnp.mean(mass + local, axis=1))))
            rows = np.asarray(
                bridged(bridge_hypers[best], confirmation, keys("chain", 1, shape))
            )
            entry[name] = {
                "setting": dict(
                    zip(bridge_names, map(float, bridge_settings[best]), strict=True)
                )
            } | read(rows[:, :, kind], len(entry))

        # The learned sampler, as trained at the first cell's budget.
        rows = jax.jit(
            lambda p, tasks, keys_, at=at: jax.vmap(
                lambda task, row: jax.vmap(
                    lambda key: dynamics.trial(
                        dynamics.euler(spec["form"], p),
                        task,
                        key,
                        chains=at["chains"],
                        steps=at["steps"],
                        burn=at["burn"],
                    )
                )(row)
            )(tasks, keys_)
        )(learned, confirmation, keys("chain", 2, shape))
        entry["learned"] = read(np.asarray(rows), len(entry))
        budgets.append(entry)
        print(
            f"a={spec['separation']} D={d} x{multiple}: "
            + " ".join(
                f"{name} {entry[name]['mass_rmse']:.3f}/{entry[name]['local_rmse']:.3f}"
                for name in ARMS
            ),
            flush=True,
        )
    return {
        "metrics": {
            "separation": spec["separation"],
            "dimension": d,
            "frame": spec["frame"],
            "budgets": budgets,
        },
        "estimands": ["mode_mass_rmse", "second_moment_rmse"],
    }
