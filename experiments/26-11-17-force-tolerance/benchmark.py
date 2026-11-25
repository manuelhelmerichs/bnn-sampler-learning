"""How accurate and how cheap must a gradient surrogate be to save compute?"""

from dataclasses import asdict
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from bnn.evaluation.cost_error import (
    break_even,
    cost_at,
    ladder,
    nested_resamples,
    rmse,
)
from bnn.target.synthetic import Task
from runner import CellSpec, Design, ResourceHint, cell_folder, finite

from . import errors, frames, kernels, learners

QUESTION = (
    "At which force error and relative cost does a gradient surrogate inside a "
    "sampler save compute?"
)
# Task parameters by family: a Gaussian with condition 100; Bayesian linear
# regression with 32 observations, whose noise of 3 keeps the stiffest direction
# below precision 90 up to D = 512; a banana with bend 0.1.
SHAPES = {
    "gaussian": {"condition": 100.0},
    "blr": {"noise": 3.0},
    "banana": {"bend": 0.1},
}
DIMENSIONS = (8, 32, 128)
# Family, dimension and hard block (None is the dense path).
CONDITIONS = (
    *(("gaussian", d, None) for d in DIMENSIONS),
    *(("gaussian", d, 8) for d in DIMENSIONS[1:]),
    *(("banana", d, None) for d in DIMENSIONS),
    *(("banana", d, 8) for d in DIMENSIONS[1:]),
    *(("blr", d, None) for d in DIMENSIONS),
)
LENGTHS = (1, 8, 64)
DELTAS = (0.01, 0.03, 0.1, 0.3)
ACCEPTANCES = (0.25, 0.45, 0.65, 0.85)
# Gradients per chain after warmup at which estimates are read.
RUNGS = (512, 2048, 8192, 32768)
TARGETS = 8
# A normalized second moment off by more than this left the target.
DIVERGED = 1e6
# The learned force is fitted up to this dimension.
LEARNED = 128
SCREEN_SEED = 520014
CONFIRMATION_SEED = 520100
ROLES = (
    "fields",
    "starts",
    "screen",
    "chain",
    "ula",
    "corpus",
    "fit",
    "evaluation",
    "bootstrap",
)
WALLTIME = {8: 20, 32: 30, 128: 60}


def arm_names(spec):
    return [
        kind if kind == "exact" else f"{kind}_{delta:g}"
        for kind, delta in errors.arms(spec["types"], spec["deltas"])
    ]


def samplers(spec):
    return [f"hmc{length}" for length in spec["lengths"]] + ["ula"]


def factors(profile, family, dimension, hard):
    smoke = profile == "smoke"
    shape = {**SHAPES[family], "hard": hard}
    tasks = [
        Task(family, "screen", 0, SCREEN_SEED, dimension, **shape),
        *(
            Task(family, "confirmation", i, CONFIRMATION_SEED + i, dimension, **shape)
            for i in range(2 if smoke else TARGETS)
        ),
    ]
    learned = None
    if dimension <= LEARNED:
        learned = {
            "chains": 8,
            "warmup": 50 if smoke else 200,
            "transitions": 32 if smoke else 512,
            "length": 8,
            "width": max(64, min(256, 2 * dimension)),
            "steps": 200 if smoke else 4000,
            "batch": 256 if smoke else 512,
            "rate": 1e-3,
            "holdout": 0.1,
            "interval": 100,
            "evaluation_draws": 512 if smoke else 4096,
        }
    return {
        "profile": profile,
        "family": family,
        "dimension": dimension,
        "path": "dense" if hard is None else "sparse",
        "screen": asdict(tasks[0]),
        "confirmation": [asdict(task) for task in tasks[1:]],
        "lengths": [1, 8] if smoke else list(LENGTHS),
        "types": list(errors.TYPES),
        "deltas": [0.1] if smoke else list(DELTAS),
        "chains": 4 if smoke else 8,
        "screen_chains": 4 if smoke else 64,
        "rungs": [128, 512] if smoke else list(RUNGS),
        "screen_budget": 256 if smoke else 4096,
        "warmup": 50 if smoke else 200,
        "acceptances": [0.45, 0.65] if smoke else list(ACCEPTANCES),
        "jitter": 0.2,
        # Initial step size times stiffness^(1/2) D^(1/4).
        "initial_step": 0.5,
        # ULA step size times stiffness^(1/2), friction per unit time, burn-in.
        "ula": {"step": 0.5, "friction": 1.0, "burnin": 128 if smoke else 1024},
        "learned": learned,
        # Cost of a potential value relative to a gradient. A BNN's forward pass
        # costs about a third of its forward and backward pass.
        "value_cost": 1 / 3,
        "bootstrap": 200 if smoke else 1000,
    }


def placement_factors(profile, width=64):
    smoke = profile == "smoke"
    return {
        "profile": profile,
        "chains": 4 if smoke else 8,
        "length": 8,
        "warmup": 50 if smoke else 500,
        "corpus_transitions": 32 if smoke else 1024,
        "test_transitions": 16 if smoke else 256,
        "realized_transitions": 64 if smoke else 2048,
        "acceptance": 0.65,
        "initial_step": 0.05,
        "jitter": 0.2,
        "fit": {
            "width": width,
            "steps": 200 if smoke else 4000,
            "batch": 256 if smoke else 512,
            "rate": 1e-3,
            "holdout": 0.1,
            "interval": 100,
        },
        "observations": [32, 128, 512, 2048, 8192, 32768],
        "whole_force_widths": [5, 20, 100],
        "whole_force_observations": [64, 1024, 8192],
        "student_observations": [64, 1024],
        "value_cost": 1 / 3,
    }


def design():
    cells = []
    for profile in ("smoke", "pilot"):
        conditions = [("gaussian", 8, None)] if profile == "smoke" else CONDITIONS
        for family, dimension, hard in conditions:
            spec = factors(profile, family, dimension, hard)
            task = Task(**spec["screen"])
            cells.append(
                CellSpec(
                    "oracle",
                    spec,
                    {
                        role: task.stream(f"force-tolerance:{profile}:{role}")
                        for role in ROLES
                    },
                    role="pilot",
                    expected_schema=("metrics", "estimands"),
                    resources=ResourceHint(
                        memory_gb=4, walltime_minutes=WALLTIME[dimension]
                    ),
                )
            )
        for width in (64,) if profile == "smoke" else (64, 256):
            cells.append(
                CellSpec(
                    "placement",
                    placement_factors(profile, width),
                    {"placement": 530000 + (profile == "pilot")},
                    role="pilot",
                    expected_schema=("metrics", "estimands"),
                    resources=ResourceHint(memory_gb=2, walltime_minutes=30),
                )
            )
    return Design(QUESTION, ("smoke", "pilot"), (*ROLES, "placement"), tuple(cells))


def measure(cell):
    if cell.stage == "placement":
        return placement(cell)
    return oracle(cell)


# ----------------------------------------------------------------------------
# Running batches of chains


def paired_keys(key, pairing, chains):
    """One key per (group, chain); groups with the same pairing index share keys."""
    return jax.random.split(key, (int(np.max(pairing)) + 1, chains))[
        jnp.asarray(pairing)
    ]


def listed(values):
    """A list for a summary, with undefined values as None."""
    return [float(v) if np.isfinite(v) else None for v in values]


def stacked(groups):
    return jax.tree.map(lambda *leaves: jnp.stack(leaves), *groups)


def hmc_rungs(kernel, group, starts, pairing, initial, acceptance, key, spec, rungs):
    """Warm up, then read the estimands after every rung of transitions.

    Returns estimates ``(group, chain, rung, estimand)``, the acceptance rate,
    the adapted step sizes and the seconds spent.
    """
    chains = starts.shape[1]
    init_key, adapt_key, run_key = jax.random.split(key, 3)
    start = perf_counter()
    state = kernel["init"](starts, paired_keys(init_key, pairing, chains), group)
    state, step = kernel["adapt"](
        paired_keys(adapt_key, pairing, chains),
        state,
        initial,
        acceptance,
        group,
        steps=spec["warmup"],
    )

    def block(key, state, steps):
        return kernel["run"](
            paired_keys(key, pairing, chains), state, step, group, steps=steps
        )

    totals, accepted, estimates = 0.0, 0.0, []
    for rung, (_, (sums, accepts), _) in zip(
        rungs, ladder(block, state, run_key, rungs), strict=True
    ):
        totals = totals + np.asarray(sums)
        accepted = accepted + np.asarray(accepts)
        estimates.append(totals / rung)
    return (
        np.stack(estimates, 2),
        accepted / rungs[-1],
        np.asarray(step),
        perf_counter() - start,
    )


def ula_rungs(kernel, group, starts, pairing, step, friction, key, burnin, rungs):
    chains = starts.shape[1]
    velocity_key, burn_key, run_key = jax.random.split(key, 3)
    start = perf_counter()
    velocities = jax.random.normal(
        velocity_key, (int(np.max(pairing)) + 1, *starts.shape[1:]), starts.dtype
    )[jnp.asarray(pairing)]
    state = (starts, velocities)
    state, _ = kernel["run"](
        paired_keys(burn_key, pairing, chains),
        state,
        step,
        friction,
        group,
        steps=burnin,
    )

    def block(key, state, steps):
        return kernel["run"](
            paired_keys(key, pairing, chains), state, step, friction, group, steps=steps
        )

    totals, estimates = 0.0, []
    for rung, (_, (sums,), _) in zip(
        rungs, ladder(block, state, run_key, rungs), strict=True
    ):
        totals = totals + np.asarray(sums)
        estimates.append(totals / rung)
    return np.stack(estimates, 2), perf_counter() - start


def select(estimates, acceptances):
    """Per arm, the target acceptance with the smallest screen RMSE (primary).

    Returns the choices and the screen RMSE of every (arm, acceptance).
    """
    error = rmse(estimates[..., -1, 0] - 1, axis=-1)
    error = np.where(np.isfinite(error), error, np.inf)
    return [acceptances[i] for i in np.argmin(error, axis=1)], error


def run_hmc(kernel, groups, fs, starts, length, acceptances, spec, keys):
    """Screen and confirmation runs of one HMC sampler for a set of arms.

    ``groups[a][t]`` is the force group of arm ``a`` on target ``t`` (0 is the
    screen target). Returns confirmation estimates ``(arm, target, chain, rung,
    estimand)``, acceptance and step sizes ``(arm, target, chain)``, selected
    acceptances, the screen RMSE ``(arm, acceptance)`` and seconds.
    """
    arms, targets = len(groups), len(fs) - 1
    chains, screen_chains = spec["chains"], spec["screen_chains"]
    d = fs[0].dimension
    initial = [
        spec["initial_step"] / np.sqrt(frame.stiffness) * d**-0.25 for frame in fs
    ]
    # Screen: every arm at every target acceptance, on the screen target.
    screen_groups = stacked([g[0] for g in groups for _ in acceptances])
    shape = (arms * len(acceptances), screen_chains)
    screen, _, _, screen_seconds = hmc_rungs(
        kernel,
        screen_groups,
        jnp.broadcast_to(starts[0][:screen_chains], (*shape, d)),
        np.zeros(shape[0], int),
        jnp.full(shape, initial[0]),
        jnp.tile(jnp.asarray(acceptances), arms)[:, None] * jnp.ones(shape),
        keys["screen"],
        spec,
        [spec["screen_budget"] // length],
    )
    chosen, screened = select(
        screen.reshape(arms, len(acceptances), *screen.shape[1:]), acceptances
    )
    # Confirmation: every arm on every confirmation target at its acceptance.
    confirmation_groups = stacked([g[t] for g in groups for t in range(1, targets + 1)])
    shape = (arms * targets, chains)
    estimates, acceptance, step, seconds = hmc_rungs(
        kernel,
        confirmation_groups,
        jnp.broadcast_to(
            jnp.stack([starts[t][:chains] for t in range(1, targets + 1)]),
            (arms, targets, chains, d),
        ).reshape(*shape, d),
        np.tile(np.arange(targets), arms),
        jnp.broadcast_to(
            jnp.asarray(initial[1:])[None, :, None], (arms, targets, chains)
        ).reshape(shape),
        jnp.repeat(jnp.asarray(chosen), targets)[:, None] * jnp.ones(shape),
        keys["chain"],
        spec,
        [rung // length for rung in spec["rungs"]],
    )
    unflat = (arms, targets, chains)
    return (
        estimates.reshape(*unflat, *estimates.shape[2:]),
        acceptance.reshape(unflat),
        step.reshape(unflat),
        chosen,
        screened,
        screen_seconds + seconds,
    )


def run_ula(kernel, groups, fs, starts, spec, key):
    arms, targets, chains = len(groups), len(fs) - 1, spec["chains"]
    d, ula = fs[0].dimension, spec["ula"]
    steps = np.array([ula["step"] / np.sqrt(frame.stiffness) for frame in fs[1:]])
    shape = (arms * targets, chains)
    step = jnp.broadcast_to(jnp.asarray(steps)[None, :, None], (arms, targets, chains))
    estimates, seconds = ula_rungs(
        kernel,
        stacked([g[t] for g in groups for t in range(1, targets + 1)]),
        jnp.broadcast_to(
            jnp.stack([starts[t][:chains] for t in range(1, targets + 1)]),
            (arms, targets, chains, d),
        ).reshape(*shape, d),
        np.tile(np.arange(targets), arms),
        step.reshape(shape),
        (ula["friction"] * step).reshape(shape),
        key,
        ula["burnin"],
        spec["rungs"],
    )
    return estimates.reshape(arms, targets, chains, *estimates.shape[2:]), seconds


# ----------------------------------------------------------------------------
# The oracle cell


def oracle(cell):
    spec, seeds = cell.factors, cell.seeds
    tasks = [Task(**spec["screen"]), *(Task(**task) for task in spec["confirmation"])]
    fs = [frames.frame(task) for task in tasks]
    arm_list = errors.arms(spec["types"], spec["deltas"])
    names, kinds = arm_names(spec), [kind for kind, _ in arm_list]

    def rng(role, *index):
        return np.random.default_rng([seeds[role], *index])

    def key(role, *index):
        return jax.random.key(int(rng(role, *index).integers(2**32)))

    count = max(spec["chains"], spec["screen_chains"])
    with jax.enable_x64():
        energy = frames.potential(tasks[0])
        potential = kernels.oracle_potential(energy)
        estimand = kernels.oracle_estimands(frames.estimands(fs[0].indexes))
        starts = [
            jnp.asarray(frame.sample(count, rng("starts", t)))
            for t, frame in enumerate(fs)
        ]
        fields, noise, scale = errors.stack(
            fs, arm_list, lambda t, kind: rng("fields", t, errors.TYPES.index(kind))
        )
        groups = [
            [
                {name: jnp.asarray(fields[name][a, t]) for name in errors.NAMES}
                | {
                    "noise": jnp.asarray(noise[a, t]),
                    "scale": jnp.asarray(scale[a, t]),
                    "variances": jnp.asarray(frame.variances),
                }
                for t, frame in enumerate(fs)
            ]
            for a in range(len(arm_list))
        ]
        # Arms without a linear field skip its rotations.
        subsets = {
            False: [a for a, k in enumerate(kinds) if k in ("exact", "noise")],
            True: [a for a, k in enumerate(kinds) if k in ("conservative", "curl")],
        }
        arrays, seconds, chosen, screens = {}, {}, {}, {}
        for length in spec["lengths"]:
            name = f"hmc{length}"
            parts, choice, screened = {}, {}, {}
            for linear, subset in subsets.items():
                if not subset:
                    continue
                kernel = kernels.hmc(
                    potential,
                    kernels.oracle_force(potential, linear),
                    estimand,
                    length,
                    spec["jitter"],
                )
                parts[linear] = run_hmc(
                    kernel,
                    [groups[a] for a in subset],
                    fs,
                    starts,
                    length,
                    spec["acceptances"],
                    spec,
                    {"screen": key("screen", length), "chain": key("chain", length)},
                )
                choice |= dict(zip(subset, parts[linear][3], strict=True))
                screened |= dict(
                    zip(subset, map(listed, parts[linear][4]), strict=True)
                )
            arrays[name] = merge(parts, subsets, len(arm_list))
            seconds[name] = sum(part[-1] for part in parts.values())
            chosen[name] = {names[a]: choice[a] for a in range(len(arm_list))}
            screens[name] = {names[a]: screened[a] for a in range(len(arm_list))}
        ula_parts = {}
        for linear, subset in subsets.items():
            if not subset:
                continue
            kernel = kernels.ula(kernels.oracle_force(potential, linear), estimand)
            ula_parts[linear] = run_ula(
                kernel, [groups[a] for a in subset], fs, starts, spec, key("ula")
            )
        ula = np.zeros((len(arm_list), *next(iter(ula_parts.values()))[0].shape[1:]))
        for linear, subset in subsets.items():
            if subset:
                ula[subset] = ula_parts[linear][0]
        arrays["ula"] = (ula, None, None)
        seconds["ula"] = sum(part[1] for part in ula_parts.values())
        learned = None
        if spec["learned"] is not None:
            learned = learn(spec, fs, energy, estimand, starts, rng, key)
            seconds["learned"] = learned.pop("seconds")
    curves = reduce_cell(spec, arrays, learned, seeds["bootstrap"])
    metrics = {
        "family": spec["family"],
        "path": spec["path"],
        "dimension": spec["dimension"],
        "hard": fs[0].task.hard,
        "stiffness": fs[0].stiffness,
        "force_power": fs[0].force_power,
        "value_cost": spec["value_cost"],
        "targets": len(fs) - 1,
        "chains": spec["chains"],
        "selected_acceptance": chosen,
        "acceptances": spec["acceptances"],
        "screen_rmse": screens,
        "curves": curves["oracle"],
        "step_size": {
            name: {
                arm: float(np.median(arrays[name][2][a])) for a, arm in enumerate(names)
            }
            for name in arrays
            if name != "ula"
        },
        "ula_step_size": float(spec["ula"]["step"] / np.sqrt(fs[1].stiffness)),
        "seconds": seconds,
        "repeat_file": "repeats.npz",
    }
    if spec["family"] != "banana" and spec["dimension"] <= LEARNED:
        metrics["ula_prediction"] = ula_prediction(spec, fs, groups, names)
    saved = (
        {f"{name}_estimates": arrays[name][0] for name in arrays}
        | {f"{name}_acceptance": arrays[name][1] for name in arrays if name != "ula"}
        | {f"{name}_step": arrays[name][2] for name in arrays if name != "ula"}
    )
    if learned is not None:
        metrics["learned_results"] = {
            "force_error": learned["force_error"],
            "fits": learned["fits"],
            "selected_acceptance": learned["chosen"],
            "screen_rmse": learned["screened"],
            "step_size": {
                name: float(np.median(value[2]))
                for name, value in learned["arrays"].items()
                if name != "ula"
            },
            "curves": curves["learned"],
        }
        saved |= {
            f"learned_{name}_estimates": value[0]
            for name, value in learned["arrays"].items()
        }
    np.savez_compressed(
        cell_folder() / "repeats.npz",
        arms=np.array(names),
        samplers=np.array(samplers(spec)),
        **saved,
    )
    return {
        "metrics": metrics,
        "estimands": ["normalized_second_moment_rmse", "break_even_relative_cost"],
    }


def merge(parts, subsets, arms):
    """Put the arms of the two batches back in arm order."""
    first = next(iter(parts.values()))
    out = [np.zeros((arms, *array.shape[1:])) for array in first[:3]]
    for linear, subset in subsets.items():
        if subset:
            for target, array in zip(out, parts[linear][:3], strict=True):
                target[subset] = array
    return tuple(out)


def learn(spec, fs, energy, estimand, starts, rng, key):
    """Fit the MLP force per target, then run every sampler with it."""
    options = spec["learned"]
    start = perf_counter()
    exact = errors.stack(fs, [("exact", 0.0)], None)[0]
    exact_groups = [
        {name: jnp.asarray(exact[name][0, t]) for name in errors.NAMES}
        | {
            "noise": jnp.asarray(0.0),
            "scale": jnp.asarray(0.0),
            "variances": jnp.asarray(frame.variances),
        }
        for t, frame in enumerate(fs)
    ]
    potential = kernels.oracle_potential(energy)
    corpus_kernel = kernels.hmc(
        potential,
        kernels.oracle_force(potential, False),
        estimand,
        options["length"],
        spec["jitter"],
    )
    gradient = jax.jit(jax.vmap(jax.grad(energy), in_axes=(0, None)))
    fitted, records, force_error = [], [], []
    chains, d = options["chains"], fs[0].dimension
    for t, frame in enumerate(fs):
        group = stacked([exact_groups[t]])
        u0 = jnp.asarray(frame.sample(chains, rng("corpus", t)))[None]
        pairing = np.zeros(1, int)
        init_key, adapt_key, record_key = jax.random.split(key("corpus", t), 3)
        state = corpus_kernel["init"](u0, paired_keys(init_key, pairing, chains), group)
        initial = spec["initial_step"] / np.sqrt(frame.stiffness) * d**-0.25
        state, step = corpus_kernel["adapt"](
            paired_keys(adapt_key, pairing, chains),
            state,
            jnp.full((1, chains), initial),
            jnp.full((1, chains), 0.65),
            group,
            steps=options["warmup"],
        )
        _, positions, forces, kept = corpus_kernel["record"](
            paired_keys(record_key, pairing, chains),
            state,
            step,
            group,
            steps=options["transitions"],
        )
        kept = np.asarray(kept).reshape(-1)
        u = np.asarray(positions).reshape(-1, d)[kept]
        g = np.asarray(forces).reshape(-1, d)[kept]
        params, record = learners.fit(
            frame.to_task(u),
            g @ frame.rotation.T,
            options,
            int(rng("fit", t).integers(2**31)),
        )
        draws = frame.sample(options["evaluation_draws"], rng("evaluation", t))
        exact_force = np.asarray(
            gradient(jnp.asarray(draws), jnp.asarray(frame.variances))
        )
        record["force_error"] = learners.relative_error(
            params,
            jnp.asarray(frame.to_task(draws)),
            jnp.asarray(exact_force @ frame.rotation.T),
        )
        record["corpus_gradients"] = float(
            chains
            * (1 + options["length"] * (options["warmup"] + options["transitions"]))
        )
        fitted.append(
            {
                "mlp": params,
                "rotation": jnp.asarray(frame.rotation),
                "origin": jnp.asarray(frame.origin),
                "variances": jnp.asarray(frame.variances),
            }
        )
        records.append(record)
        force_error.append(record["force_error"])

    def force(u, key, group):
        theta = group["origin"] + group["rotation"] @ u
        return group["rotation"].T @ learners.apply(group["mlp"], theta)

    arrays, chosen, screened = {}, {}, {}
    groups = [fitted]
    for length in spec["lengths"]:
        kernel = kernels.hmc(potential, force, estimand, length, spec["jitter"])
        *result, choice, screen, _ = run_hmc(
            kernel,
            groups,
            fs,
            starts,
            length,
            spec["acceptances"],
            spec,
            {
                "screen": key("screen", length, 1),
                "chain": key("chain", length, 1),
            },
        )
        arrays[f"hmc{length}"] = tuple(array[0] for array in result)
        chosen[f"hmc{length}"] = choice[0]
        screened[f"hmc{length}"] = listed(screen[0])
    estimates, _ = run_ula(
        kernels.ula(force, estimand), groups, fs, starts, spec, key("ula", 1)
    )
    arrays["ula"] = (estimates[0], None, None)
    return {
        "arrays": arrays,
        "fits": records[1:],
        "screen_fit": records[0],
        "force_error": force_error[1:],
        "chosen": chosen,
        "screened": screened,
        "seconds": perf_counter() - start,
    }


def ula_prediction(spec, fs, groups, names):
    """Exact stationary estimands of ULA on a Gaussian with a linear error field.

    The update is linear in (q, v), so its stationary law solves a discrete
    Lyapunov equation. Averaged over the confirmation targets on which the
    recursion is stable; ``unstable`` counts the others, which have no
    stationary law.
    """
    from scipy.linalg import solve_discrete_lyapunov

    d, ula = fs[0].dimension, spec["ula"]
    identity = np.eye(d)
    out = {}
    for a, name in enumerate(names):
        values, unstable = [], 0
        for t, frame in enumerate(fs[1:], start=1):
            group = {k: np.asarray(v) for k, v in groups[a][t].items()}
            h = ula["step"] / np.sqrt(frame.stiffness)
            gamma = ula["friction"] * h
            field = errors.matrix(group) if float(group["scale"]) else np.zeros((d, d))
            precision = np.diag(1 / frame.variances) + float(group["scale"]) * field
            shift = float(group["scale"]) * field @ group["offset"]
            transition = np.block(
                [
                    [identity - h * h * precision, h * (1 - gamma) * identity],
                    [-h * precision, (1 - gamma) * identity],
                ]
            )
            if np.max(np.abs(np.linalg.eigvals(transition))) >= 1:
                unstable += 1
                continue
            noise = 2 * gamma + h * h * float(group["noise"]) ** 2
            covariance = solve_discrete_lyapunov(
                transition,
                noise
                * np.block(
                    [[h * h * identity, h * identity], [h * identity, identity]]
                ),
            )
            mean = np.linalg.solve(
                np.eye(2 * d) - transition, np.concatenate([h * h * shift, h * shift])
            )
            index = list(frame.indexes)
            values.append(
                (np.diag(covariance)[index] + mean[index] ** 2) / frame.variances[index]
            )
        out[name] = {
            "estimands": np.mean(values, 0).tolist() if values else None,
            "unstable": unstable,
        }
    return out


# ----------------------------------------------------------------------------
# Costs and the reduction of one oracle cell


def counts(spec, sampler, surrogate, rung):
    """Value-and-gradient, value-only and surrogate evaluations per chain."""
    if sampler == "ula":
        steps = spec["ula"]["burnin"] + rung
        return (0.0, 0.0, float(steps)) if surrogate else (float(steps), 0.0, 0.0)
    length = int(sampler[3:])
    transitions = spec["warmup"] + rung // length
    if surrogate:
        return 0.0, 1.0 + transitions, 1.0 + length * transitions
    return 1.0 + length * transitions, 0.0, 0.0


def reduce_cell(spec, arrays, learned, seed):
    """RMSE curves and break-even costs at every rung, with paired intervals.

    The reference for an arm's RMSE is the cheapest exact HMC sampler, by
    interpolation along its curve. HMC arms are also compared with the same
    sampler with the exact force. Intervals resample targets, then chains.
    """
    names = arm_names(spec)
    order = samplers(spec)
    hmcs = [s for s in order if s != "ula"]
    c, rungs = spec["value_cost"], spec["rungs"]
    targets, chains = len(spec["confirmation"]), spec["chains"]
    groups = np.repeat(np.arange(targets), chains)
    ids = nested_resamples(groups, spec["bootstrap"], np.random.default_rng(seed))

    def trials(estimates):
        # (..., target, chain, rung, estimand) -> (trial, ..., rung, estimand).
        # A chain that diverged has an infinite error.
        error = np.where(np.abs(estimates - 1) < DIVERGED, estimates - 1, np.inf)
        moved = np.moveaxis(error, (-4, -3), (0, 1))
        return moved.reshape(targets * chains, *moved.shape[2:])

    oracle = {s: trials(arrays[s][0]) for s in order}
    cost = {
        (s, surrogate): np.array([counts(spec, s, surrogate, r) for r in rungs])
        for s in order
        for surrogate in (False, True)
    }
    work = {key: vg + c * v for key, (vg, v, _) in ((k, t.T) for k, t in cost.items())}
    # The exact HMC curves, and their bootstrap replicates (sample, rung).
    curve = {h: rmse(oracle[h][:, 0, :, 0]) for h in hmcs}
    replicates = {h: rmse(oracle[h][ids][:, :, 0, :, 0], axis=1) for h in hmcs}

    def needed(h, errors_by_rung, reference):
        with np.errstate(divide="ignore", invalid="ignore"):
            return cost_at(work[h, False], reference, errors_by_rung)[0]

    def best(r, references):
        return np.min([needed(h, r, references[h]) for h in hmcs], axis=0)

    def entries(s, error, surrogate, extra=(0.0, 0.0)):
        vg, v, evaluations = cost[s, surrogate].T
        sampling = vg + c * v
        own = sampling + extra[0]
        evaluations = evaluations + extra[1]
        r = rmse(error[..., 0])
        reference = best(r, curve)
        boot = rmse(error[ids][..., 0], axis=1)
        resampled = np.stack(
            [
                best(boot[b], {h: replicates[h][b] for h in hmcs})
                for b in range(len(ids))
            ]
        )
        same = needed(s, r, curve[s]) if s in curve else None
        rows = []
        for j, rung in enumerate(rungs):
            diverged = int(np.count_nonzero(~np.isfinite(error[:, j, 0])))
            entry = {
                "budget": rung,
                "diverged": diverged,
                "transitions": rung if s == "ula" else rung // int(s[3:]),
                "rmse": float(r[j]),
                "rmse_secondary": float(rmse(error[:, j, 1])),
                "bias": float(np.mean(error[:, j, 0])),
                "bias_secondary": float(np.mean(error[:, j, 1])),
                "value_gradients": float(vg[j] + extra[0]),
                "values": float(v[j]),
                "surrogates": float(cost[s, surrogate][j, 2]),
                "fit_evaluations": float(extra[1]),
            }
            if surrogate:
                entry["rho_star"] = float(
                    break_even(reference[j], own[j], evaluations[j])
                )
                entry["rho_star_interval"] = np.quantile(
                    break_even(resampled[:, j], own[j], evaluations[j]), [0.025, 0.975]
                ).tolist()
                # The saving of a free surrogate, against the best exact sampler.
                entry["headroom"] = float(reference[j] / own[j])
                if extra[0] or extra[1]:
                    # Without the corpus and the fit: the sampling run alone.
                    entry["rho_star_sampling"] = float(
                        break_even(reference[j], sampling[j], cost[s, surrogate][j, 2])
                    )
                if same is not None:
                    entry["rho_star_same"] = float(
                        break_even(same[j], own[j], evaluations[j])
                    )
                    # Exact transitions that buy this RMSE, per transition here.
                    entry["efficiency"] = float(same[j] / work[s, False][j])
            rows.append(finite(entry))
        return rows

    curves = {
        "oracle": {
            s: {
                name: entries(s, oracle[s][:, a], a > 0) for a, name in enumerate(names)
            }
            for s in order
        }
    }
    if learned:
        # A target's corpus and fit serve its chains: each chain pays its share.
        extra = (
            np.mean([fit["corpus_gradients"] for fit in learned["fits"]]) / chains,
            np.mean([fit["evaluations"] for fit in learned["fits"]]) / chains,
        )
        curves["learned"] = {
            s: entries(s, trials(learned["arrays"][s][0]), True, extra) for s in order
        }
    return curves


# ----------------------------------------------------------------------------
# Placement: the collapsed BNN's MLP force and the costs of the draft's learners


def placement(cell):
    from bnn.evaluation.diagnostics import multichain_ess

    spec = cell.factors
    rng = np.random.default_rng(cell.seeds["placement"])
    with jax.enable_x64():
        target, energy = learners.collapsed_bnn_potential()
        d, chains, length = target.dimension, spec["chains"], spec["length"]
        gradient = jax.grad(energy)

        def potential(u, group):
            return energy(u)

        def exact_force(u, key, group):
            return gradient(u)

        def learned_force(u, key, group):
            return learners.apply(group["mlp"], u)

        def observables(u, group):
            # The log-noise coordinate and the hidden squared radius.
            return jnp.stack([u[-1], jnp.sum(u[:-1] ** 2)])

        exact = kernels.hmc(potential, exact_force, observables, length, spec["jitter"])
        learned = kernels.hmc(
            potential, learned_force, observables, length, spec["jitter"]
        )
        none = stacked([{"none": jnp.zeros(())}])
        pairing = np.zeros(1, int)

        def warm(kernel, group, u0, key):
            init_key, adapt_key = jax.random.split(key)
            state = kernel["init"](u0, paired_keys(init_key, pairing, chains), group)
            return kernel["adapt"](
                paired_keys(adapt_key, pairing, chains),
                state,
                jnp.full((1, chains), spec["initial_step"]),
                jnp.full((1, chains), spec["acceptance"]),
                group,
                steps=spec["warmup"],
            )

        def recorded(key, transitions):
            """Leapfrog positions and forces of exact chains from prior draws."""
            u0 = jnp.asarray(rng.normal(size=(1, chains, d)))
            warm_key, record_key = jax.random.split(key)
            state, step = warm(exact, none, u0, warm_key)
            state, positions, forces, kept = exact["record"](
                paired_keys(record_key, pairing, chains),
                state,
                step,
                none,
                steps=transitions,
            )
            kept = np.asarray(kept).reshape(-1)
            return (
                state,
                np.asarray(positions).reshape(-1, d)[kept],
                np.asarray(forces).reshape(-1, d)[kept],
            )

        keys = jax.random.split(jax.random.key(cell.seeds["placement"]), 4)
        _, corpus, corpus_forces = recorded(keys[0], spec["corpus_transitions"])
        test_state, test, test_forces = recorded(keys[1], spec["test_transitions"])
        params, fit = learners.fit(
            corpus, corpus_forces, spec["fit"], int(rng.integers(2**31))
        )
        fit["force_error"] = learners.relative_error(
            params, jnp.asarray(test), jnp.asarray(test_forces)
        )
        fit["corpus_gradients"] = float(
            chains * (1 + length * (spec["warmup"] + spec["corpus_transitions"]))
        )
        # Realized: exact and learned HMC from the test chains' last states.
        realized = {}
        for name, kernel, group in (
            ("exact", exact, none),
            ("learned", learned, stacked([{"mlp": params}])),
        ):
            warm_key, trace_key = jax.random.split(keys[2])
            state, step = warm(kernel, group, test_state[0], warm_key)
            _, trace = kernel["trace"](
                paired_keys(trace_key, pairing, chains),
                state,
                step,
                group,
                steps=spec["realized_transitions"],
            )
            values = np.asarray(trace)[0]
            realized[name] = {
                "step_size": float(np.median(step)),
                "ess_per_transition": (
                    multichain_ess(values) / values.shape[0] / values.shape[1]
                ).tolist(),
            }
        efficiency = np.array(realized["learned"]["ess_per_transition"]) / np.array(
            realized["exact"]["ess_per_transition"]
        )
        realized["efficiency"] = efficiency.tolist()
        realized["rho_star"] = (efficiency - spec["value_cost"] / length).tolist()
        gradients = [
            learners.gradient_flops(learners.collapsed_bnn_potential(count)[1], d)
            for count in spec["observations"]
        ]
        metrics = {
            "collapsed_bnn": {
                "dimension": d,
                "observations": len(target.data()[0]),
                "fit": fit,
                "realized": realized,
                "mlp_flops": learners.mlp_flops(d, spec["fit"]["width"]),
                "gradient_flops": dict(
                    zip(map(str, spec["observations"]), gradients, strict=True)
                ),
            },
            "whole_force": learners.whole_force_costs(
                spec["whole_force_widths"], spec["whole_force_observations"]
            ),
            "student": learners.student_costs(spec["student_observations"]),
        }
    return {"metrics": metrics, "estimands": ["relative_force_error", "relative_cost"]}
