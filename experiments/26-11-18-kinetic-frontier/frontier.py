"""Exploration against bias at a fixed gradient budget, classical and learned.

Every trial spends ``chains * steps`` gradients on one two-mode mixture and
estimates the mass of the first mode and the second moments off the axis. The
classical cell runs tuned grids of unadjusted samplers, exact HMC and tempered
SMC. A learned cell trains the kinetic fields of Kim et al. (2024) on that
evaluation loss itself, which is the best case for the learner, and then reads
statistics and tasks it was not trained on.
"""

from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from bnn.evaluation.cost_error import nested_resamples
from bnn.samplers.tempered_smc import SMCConfig, tempered_smc
from jax.flatten_util import ravel_pytree
from runner import CellSpec, Design, ResourceHint, cell_folder, fingerprint

from . import dynamics, kinetic, learning, targets

SEPARATIONS = (2.5, 3.5, 5.0)
DIMENSIONS = (8, 32, 128)
FRAMES = ("aligned", "rotated")
# Learned cells: the residual form everywhere up to D = 32; at the separation
# where the budget is marginal also the published form, SGHMC itself with only
# its three numbers trained, which no frame changes, and the residual form in
# the turned frame, which keeps the reference at the origin and hides the axis.
LEARNED = (
    *(
        ("residual", a, d, frame)
        for a in SEPARATIONS
        for d in DIMENSIONS[:2]
        for frame in FRAMES
    ),
    *(("published", 3.5, d, frame) for d in DIMENSIONS[:2] for frame in FRAMES),
    *(("classical", 3.5, d, "aligned") for d in DIMENSIONS[:2]),
    *(("residual", 3.5, d, "turned") for d in DIMENSIONS[:2]),
)
ROLES = ("screen", "confirmation", "transfer", "chain", "train", "smc", "bootstrap")
SEED = 630018
# Step size, friction and temperature of the two kinetic Langevin integrators.
EULER = {
    "step": (0.1, 0.2, 0.35, 0.5, 0.7, 1.0),
    "friction": (0.3, 1.0, 3.0),
    "temperature": (1.0, 1.3, 1.7, 2.2, 3.0),
}
BAOAB = {
    "step": (0.35, 0.5, 0.7, 1.0, 1.4, 1.8),
    "friction": (0.1, 0.3, 1.0, 3.0),
    "temperature": (1.0, 1.3, 1.7, 2.2, 3.0),
}
# Step size and decoherence length of MCLMC, in units of sqrt(D).
MCLMC = {"step": (0.1, 0.2, 0.35, 0.5, 0.7, 1.0), "length": (0.3, 1.0, 3.0, 10.0, 30.0)}
# Step size of exact HMC in units of D^(-1/4), and its largest number of
# leapfrog steps.
HMC = {"step": (0.8, 1.2, 1.7, 2.4), "length": (2, 4, 8, 16)}
WALLTIME = {8: 60, 32: 150, 128: 240}


def seed(profile, separation, dimension, role):
    """Seeds follow from the condition, so its cells see the same tasks."""
    identity = [SEED, profile, separation, dimension, role]
    return int(fingerprint(identity, namespace="kinetic-frontier")[:16], 16)


def shared(profile, separation, dimension):
    smoke = profile == "smoke"
    return {
        "profile": profile,
        "separation": separation,
        "dimension": dimension,
        "chains": 4 if smoke else 8,
        "steps": 128 if smoke else 2048,
        "burn": 32 if smoke else 512,
        "screen_tasks": 8 if smoke else 64,
        "targets": 2 if smoke else 8,
        "trials": 2 if smoke else 32,
        "weights": [0.15, 0.85],
        "euler": {k: list(v[:2] if smoke else v) for k, v in EULER.items()},
        "bootstrap": 50 if smoke else 1000,
    }


def changes(separation, dimension, frame=None):
    """Tasks a sampler tuned on one condition is also read on, unchanged."""
    out = [
        {"separation": separation - 0.5},
        {"separation": separation + 0.5},
        {"scale": 2.0},
    ]
    if dimension < DIMENSIONS[-1]:
        out.append({"dimension": 4 * dimension})
    if frame is not None:
        out += [{"frame": other} for other in ("aligned", "shifted", "rotated")]
        out = [change for change in out if change != {"frame": frame}]
    return out


def classical_factors(profile, separation, dimension):
    smoke = profile == "smoke"
    return shared(profile, separation, dimension) | {
        "frame": "aligned",
        "baoab": {k: list(v[:2] if smoke else v) for k, v in BAOAB.items()},
        "mclmc": {k: list(v[:2] if smoke else v) for k, v in MCLMC.items()},
        "hmc": {k: list(v[:2] if smoke else v) for k, v in HMC.items()},
        "smc": {
            "particles": [32, 64] if smoke else [32, 64, 128, 256, 512],
            "moves": 8,
            "base_sd": 3.0,
            "trials": 2 if smoke else 16,
        },
        "transfer": [] if smoke else changes(separation, dimension),
    }


def learned_factors(profile, form, separation, dimension, frame):
    smoke = profile == "smoke"
    return shared(profile, separation, dimension) | {
        "frame": frame,
        "form": form,
        "training": {
            "iterations": 4 if smoke else 300 if dimension <= 8 else 200,
            "pairs": 2 if smoke else 16,
            "tasks": 2 if smoke else 8,
            "sigma": 0.03,
            "rate": 0.01,
            "weight": 1.0,
            "interval": 2 if smoke else 25,
        },
        "distill": {"steps": 64 if smoke else 512, "every": 16},
        "transfer": [] if smoke else changes(separation, dimension, frame),
    }


def design():
    cells = []
    for profile in ("smoke", "pilot"):
        smoke = profile == "smoke"
        conditions = (
            [(3.5, 8)] if smoke else [(a, d) for a in SEPARATIONS for d in DIMENSIONS]
        )
        for separation, dimension in conditions:
            cells.append(
                CellSpec(
                    "classical",
                    classical_factors(profile, separation, dimension),
                    {r: seed(profile, separation, dimension, r) for r in ROLES},
                    role="pilot",
                    expected_schema=("metrics", "estimands"),
                    resources=ResourceHint(
                        memory_gb=4, walltime_minutes=WALLTIME[dimension]
                    ),
                )
            )
        learned = [("residual", 3.5, 8, "aligned")] if smoke else LEARNED
        for form, separation, dimension, frame in learned:
            cells.append(
                CellSpec(
                    "learned",
                    learned_factors(profile, form, separation, dimension, frame),
                    {r: seed(profile, separation, dimension, r) for r in ROLES},
                    role="pilot",
                    expected_schema=("metrics", "estimands"),
                    resources=ResourceHint(
                        memory_gb=4, walltime_minutes=2 * WALLTIME[dimension]
                    ),
                )
            )
    return Design(QUESTION, ("smoke", "pilot"), ROLES, tuple(cells))


QUESTION = (
    "Do learned kinetic fields move the trade-off between crossing modes and "
    "local bias at a fixed gradient budget, or pick a point on the classical one?"
)

# ----------------------------------------------------------------------------
# Tasks


def as_arrays(task, dtype=jnp.float32):
    return {
        name: jnp.asarray(value, dtype)
        for name, value in (task | {"mass": targets.truth(task)["mass"]}).items()
    }


def task_sets(spec, seeds, frame):
    """Screen tasks with random weights; confirmation and transfer targets."""
    a, d = spec["separation"], spec["dimension"]
    low, high = spec["weights"]
    spread = np.linspace(low, high, spec["targets"])
    sets = {
        "screen": targets.tasks(
            np.random.default_rng([seeds["screen"], 0]),
            spec["screen_tasks"],
            d,
            a,
            frame,
        ),
        "confirmation": targets.tasks(
            np.random.default_rng([seeds["confirmation"], 0]),
            spec["targets"],
            d,
            a,
            frame,
            weights=spread,
        ),
    }
    for index, change in enumerate(spec["transfer"]):
        sets[label(change)] = targets.tasks(
            np.random.default_rng([seeds["transfer"], index]),
            spec["targets"],
            change.get("dimension", d),
            change.get("separation", a),
            change.get("frame", frame),
            weights=spread,
            scale=change.get("scale", 1.0),
        )
    return sets


def label(change):
    ((name, value),) = change.items()
    return f"{name}={value}"


def draw_training(spec, seeds, frame):
    def draw(iteration):
        rng = np.random.default_rng([seeds["train"], iteration])
        return as_arrays(
            targets.tasks(
                rng,
                spec["training"]["tasks"],
                spec["dimension"],
                spec["separation"],
                frame,
            )
        )

    return draw


# ----------------------------------------------------------------------------
# Reduction of trials


def reduce(rows, task, seed_, count):
    """Errors of ``rows[target, trial]`` against the exact values of ``task``."""
    rows = np.asarray(rows, np.float64)
    exact = targets.truth(task)
    index = {name: i for i, name in enumerate(dynamics.RESULTS)}
    finite = rows[..., index["finite"]] > 0
    mass = np.where(
        finite,
        np.minimum(
            (rows[..., index["mass"]] - exact["mass"][:, None]) ** 2, dynamics.MASS_CLIP
        ),
        dynamics.MASS_CLIP,
    )
    local = np.where(
        finite,
        np.minimum(rows[..., index["local"]], dynamics.LOCAL_CLIP),
        dynamics.LOCAL_CLIP,
    )
    groups = np.repeat(np.arange(rows.shape[0]), rows.shape[1])
    samples = nested_resamples(groups, count, np.random.default_rng(seed_))
    flat_mass, flat_local = mass.reshape(-1), local.reshape(-1)

    def interval(values):
        return [float(v) for v in np.quantile(values, [0.025, 0.975])]

    out = {
        "mass_rmse": float(np.sqrt(mass.mean())),
        "mass_rmse_interval": interval(np.sqrt(flat_mass[samples].mean(1))),
        "local_rmse": float(np.sqrt(local.mean())),
        "local_rmse_interval": interval(np.sqrt(flat_local[samples].mean(1))),
        "loss": float(mass.mean() + local.mean()),
        "loss_interval": interval(
            flat_mass[samples].mean(1) + flat_local[samples].mean(1)
        ),
        "mass_bias": float(
            np.mean(
                np.where(finite, rows[..., index["mass"]] - exact["mass"][:, None], 0.0)
            )
        ),
        "second": float(np.mean(rows[..., index["second"]][finite]))
        if finite.any()
        else None,
        "switches": float(np.mean(rows[..., index["switches"]])),
        "diverged": float(1 - finite.mean()),
    }
    # Statistics no arm is trained on, as the mean ratio to their exact value
    # over trials and its standard error over targets.
    for name in ("barrier", "fourth", "tail", "axis"):
        ratio = rows[..., index[name]] / exact[name][:, None]
        # Targets on which every trial diverged have no ratio to report.
        per_target = np.array(
            [r[ok].mean() for r, ok in zip(ratio, finite, strict=True) if ok.any()]
        )
        out[name] = float(per_target.mean()) if len(per_target) else None
        out[f"{name}_se"] = (
            float(per_target.std() / np.sqrt(len(per_target)))
            if len(per_target)
            else None
        )
    return out


# ----------------------------------------------------------------------------
# Classical cell


# The order in which a sampler reads its numbers; a design's factors lose theirs.
ORDER = {
    "euler": ("step", "friction", "temperature"),
    "baoab": ("step", "friction", "temperature"),
    "bridge": ("step", "friction", "temperature", "share"),
    "mclmc": ("step", "length"),
}


def grid(kind, axes):
    """Every setting of a sampler, one row each, in the order it reads them."""
    names = ORDER[kind]
    mesh = np.stack(
        np.meshgrid(*(axes[name] for name in names), indexing="ij"), axis=-1
    )
    return names, mesh.reshape(-1, len(names))


def unadjusted(kind, spec, steps_per_gradient=1):
    """``(hypers, tasks, keys[task, trial]) -> rows[setting, task, trial]``."""
    steps = spec["steps"] // steps_per_gradient
    burn = spec["burn"] // steps_per_gradient

    def sampler(hyper):
        parameters = {"hyper": hyper}
        if kind == "euler":
            return dynamics.euler("classical", parameters)
        return {"baoab": dynamics.baoab, "mclmc": dynamics.mclmc}[kind](parameters)

    def one(hyper, tasks, keys):
        return jax.vmap(
            lambda task, row: jax.vmap(
                lambda key: dynamics.trial(
                    sampler(hyper),
                    task,
                    key,
                    chains=spec["chains"],
                    steps=steps,
                    burn=burn,
                )
            )(row)
        )(tasks, keys)

    return jax.jit(
        lambda hypers, tasks, keys: jax.lax.map(lambda h: one(h, tasks, keys), hypers)
    )


def exact_hmc(spec, length):
    """``(step, tasks, keys[task, trial]) -> rows[task, trial]`` of Gaussian HMC."""
    run = kinetic.hmc(
        kinetic.FAMILIES["gaussian"],
        targets.potential,
        targets.force,
        targets.statistics,
        targets.core,
    )
    transitions = kinetic.transitions_for(spec["steps"], length)
    burn = transitions * spec["burn"] // spec["steps"]

    def trial(step, task, key):
        k1, k2 = jax.random.split(key)
        d = targets.dimension(task)
        z = jax.random.normal(k1, (spec["chains"], d), task["location"].dtype)
        sums, _, gradients, switches = run(
            k2,
            targets.embedded(z, task),
            step,
            task,
            transitions=transitions,
            burn=burn,
            length=length,
        )
        return dynamics.row(sums.mean(0), switches.sum() / gradients.sum())

    return jax.jit(
        lambda step, tasks, keys: jax.vmap(
            lambda task, row: jax.vmap(lambda key: trial(step, task, key))(row)
        )(tasks, keys)
    )


def _smc_prior(x, params):
    return -0.5 * jnp.sum((x - params["location"]) ** 2) / params["base"] ** 2


def _smc_likelihood(x, params):
    return -targets.potential(x, params) - _smc_prior(x, params)


def smc(spec, tasks, seed_):
    """Tempered SMC from a wide Gaussian: rows and costs ``[size, target, trial]``."""
    settings = spec["smc"]
    d = spec["dimension"]
    count, trials = spec["targets"], settings["trials"]
    rows = np.zeros((len(settings["particles"]), count, trials, len(dynamics.RESULTS)))
    costs = np.zeros(rows.shape[:3])
    observe = jax.jit(lambda x, task: targets.statistics(x, task).mean(0))
    for s, particles in enumerate(settings["particles"]):
        for t in range(count):
            task = {
                name: jnp.asarray(value[t], jnp.float64)
                for name, value in tasks.items()
            }
            params = task | {"base": jnp.asarray(settings["base_sd"] * task["scale"])}
            for trial in range(trials):
                rng = np.random.default_rng([seed_, s, t, trial])
                start = np.asarray(task["location"]) + settings["base_sd"] * float(
                    task["scale"]
                ) * rng.normal(size=(particles, d))
                result = tempered_smc(
                    _smc_prior,
                    _smc_likelihood,
                    params,
                    start,
                    SMCConfig(n_mcmc=settings["moves"], seed=int(rng.integers(2**31))),
                )
                with jax.enable_x64():
                    means = observe(jnp.asarray(result.positions), task)
                    rows[s, t, trial] = np.asarray(dynamics.row(means, jnp.zeros(())))
                costs[s, t, trial] = result.n_loglik_evals
    return rows, costs


def select(names, settings, rows, tasks):
    """The setting with the smallest loss on the screen tasks."""
    mass, local = dynamics.errors(rows[:, :, 0], tasks["mass"])
    loss = np.asarray(jnp.mean(mass + local, axis=1))
    best = int(np.argmin(loss))
    return best, dict(zip(names, (float(v) for v in settings[best]), strict=True)), loss


def keys_for(seed_, *index, shape):
    key = jax.random.key(int(np.random.default_rng([seed_, *index]).integers(2**32)))
    return jax.random.split(key, shape)


def measure_classical(cell):
    spec, seeds = cell.factors, cell.seeds
    d = spec["dimension"]
    sets = task_sets(spec, seeds, spec["frame"])
    budget = spec["chains"] * spec["steps"]
    arms, arrays, seconds = {}, {}, {}

    def record(name, family, setting, rows, task, salt):
        arms[name] = {"family": family, "setting": setting} | reduce(
            rows, task, [seeds["bootstrap"], salt], spec["bootstrap"]
        )

    families = {
        "euler": (spec["euler"], 1, lambda v: v),
        "baoab": (spec["baoab"], 1, lambda v: v),
        "mclmc": (spec["mclmc"], 2, lambda v: v * np.sqrt(d)),
    }
    selected = {}
    for f, (kind, (axes, cost, scale)) in enumerate(families.items()):
        start = perf_counter()
        names, settings = grid(kind, axes)
        hypers = jnp.log(jnp.asarray(scale(settings), jnp.float32))
        run = unadjusted(kind, spec, cost)
        screen = as_arrays(sets["screen"])
        screened = run(
            hypers,
            screen,
            keys_for(seeds["screen"], f, shape=(spec["screen_tasks"], 1)),
        )
        best, setting, loss = select(names, settings, screened, screen)
        confirmed = np.asarray(
            run(
                hypers,
                as_arrays(sets["confirmation"]),
                keys_for(seeds["chain"], f, shape=(spec["targets"], spec["trials"])),
            )
        )
        arrays[kind] = confirmed
        selected[kind] = (run, hypers[best : best + 1], setting)
        record(kind, kind, setting, confirmed[best], sets["confirmation"], f)
        # Every setting on the confirmation targets, for the frontier.
        arms[kind]["grid"] = [
            dict(zip(names, (float(v) for v in settings[i]), strict=True))
            | {"screen_loss": float(loss[i])}
            | {
                key: value
                for key, value in reduce(
                    confirmed[i], sets["confirmation"], [seeds["bootstrap"], f, i], 2
                ).items()
                if key in {"mass_rmse", "local_rmse", "loss", "second", "diverged"}
            }
            for i in range(len(settings))
        ]
        seconds[kind] = perf_counter() - start

    # Exact comparators at the same budget.
    start = perf_counter()
    with jax.enable_x64():
        screen = as_arrays(sets["screen"], jnp.float64)
        confirmation = as_arrays(sets["confirmation"], jnp.float64)
        candidates = []
        for i, length in enumerate(spec["hmc"]["length"]):
            run = exact_hmc(spec, length)
            for j, factor in enumerate(spec["hmc"]["step"]):
                step = jnp.asarray(factor * d**-0.25)
                rows = run(
                    step,
                    screen,
                    keys_for(
                        seeds["screen"], 10, i, j, shape=(spec["screen_tasks"], 1)
                    ),
                )
                mass, local = dynamics.errors(rows[:, 0], screen["mass"])
                candidates.append((float(jnp.mean(mass + local)), i, j))
        _, i, j = min(candidates)
        length, factor = spec["hmc"]["length"][i], spec["hmc"]["step"][j]
        run = exact_hmc(spec, length)
        step = jnp.asarray(factor * d**-0.25)
        confirmed = np.asarray(
            run(
                step,
                confirmation,
                keys_for(seeds["chain"], 10, shape=(spec["targets"], spec["trials"])),
            )
        )
        arrays["hmc"] = confirmed
        hmc_setting = {"step": float(step), "length": length}
        record("hmc", "hmc", hmc_setting, confirmed, sets["confirmation"], 10)
        hmc_transfer = {}
        for change in spec["transfer"]:
            name = label(change)
            task = as_arrays(sets[name], jnp.float64)
            # The step follows the dimension and the scale; nothing else is retuned.
            moved = step * task["scale"][0] * (targets.dimension(task) / d) ** -0.25
            hmc_transfer[name] = np.asarray(
                run(
                    moved,
                    task,
                    keys_for(
                        seeds["chain"], 11, shape=(spec["targets"], spec["trials"])
                    ),
                )
            )
    seconds["hmc"] = perf_counter() - start

    start = perf_counter()
    rows, costs = smc(spec, sets["confirmation"], seeds["smc"])
    arrays["smc"], arrays["smc_cost"] = rows, costs
    sizes = []
    for s, particles in enumerate(spec["smc"]["particles"]):
        sizes.append(
            {"particles": particles, "cost": float(costs[s].mean())}
            | reduce(
                rows[s],
                sets["confirmation"],
                [seeds["bootstrap"], 20, s],
                spec["bootstrap"],
            )
        )
    cost = np.array([size["cost"] for size in sizes])
    arms["smc"] = {"family": "smc", "sizes": sizes, "budget": budget}
    for name in ("mass_rmse", "local_rmse"):
        error = np.array([size[name] for size in sizes])
        # The error at the budget, read off the curve of cost against error.
        order = np.argsort(cost)
        arms["smc"][name] = float(
            np.exp(np.interp(np.log(budget), np.log(cost[order]), np.log(error[order])))
        )
        arms["smc"][f"{name}_extrapolated"] = bool(
            budget < cost.min() or budget > cost.max()
        )
    seconds["smc"] = perf_counter() - start

    # The selected classical settings on changed tasks, without retuning.
    transfer = {}
    for c, change in enumerate(spec["transfer"]):
        name = label(change)
        task = as_arrays(sets[name])
        transfer[name] = {}
        for f, (kind, (run, hyper, _)) in enumerate(selected.items()):
            moved = hyper
            if kind == "mclmc" and "dimension" in change:
                moved = hyper + 0.5 * np.log(change["dimension"] / d)
            rows = np.asarray(
                run(
                    moved,
                    task,
                    keys_for(
                        seeds["chain"],
                        30,
                        c,
                        f,
                        shape=(spec["targets"], spec["trials"]),
                    ),
                )
            )[0]
            transfer[name][kind] = reduce(
                rows, sets[name], [seeds["bootstrap"], 30, c, f], spec["bootstrap"]
            )
        transfer[name]["hmc"] = reduce(
            hmc_transfer[name],
            sets[name],
            [seeds["bootstrap"], 31, c],
            spec["bootstrap"],
        )
    np.savez_compressed(cell_folder() / "rows.npz", **arrays)
    return {
        "metrics": {
            "separation": spec["separation"],
            "dimension": d,
            "frame": spec["frame"],
            "budget": budget,
            "arms": arms,
            "transfer_results": transfer,
            "seconds": seconds,
            "row_file": "rows.npz",
        },
        "estimands": ["mode_mass_rmse", "second_moment_rmse"],
    }


# ----------------------------------------------------------------------------
# Learned cell


def measure_learned(cell):
    spec, seeds = cell.factors, cell.seeds
    d, form, frame = spec["dimension"], spec["form"], spec["frame"]
    training = spec["training"] | {k: spec[k] for k in ("chains", "steps", "burn")}
    sets = task_sets(spec, seeds, frame)
    screen = as_arrays(sets["screen"])
    shape = (spec["targets"], spec["trials"])

    # The learner starts from the classical setting chosen on the screen tasks.
    names, settings = grid("euler", spec["euler"])
    hypers = jnp.log(jnp.asarray(settings, jnp.float32))
    screened = unadjusted("euler", spec)(
        hypers, screen, keys_for(seeds["screen"], 0, shape=(spec["screen_tasks"], 1))
    )
    best, setting, _ = select(names, settings, screened, screen)
    key = jax.random.key(
        int(np.random.default_rng([seeds["train"], 0]).integers(2**32))
    )
    k_init, k_train, k_distill = jax.random.split(key, 3)
    parameters = learning.initial(form, settings[best], k_init)

    start = perf_counter()
    log = []

    def report(entry):
        log.append(entry)
        print(
            f"{form} a={spec['separation']} D={d} {frame} "
            f"[{perf_counter() - start:6.0f}s] {entry}",
            flush=True,
        )

    learned, history = learning.train(
        form,
        parameters,
        draw_training(spec, seeds, frame),
        screen,
        training,
        k_train,
        report,
    )
    seconds = perf_counter() - start

    def rows_of(kind, values):
        run = jax.jit(
            lambda p, tasks, keys: jax.vmap(
                lambda task, row: jax.vmap(
                    lambda key: dynamics.trial(
                        dynamics.euler(kind, p),
                        task,
                        key,
                        chains=spec["chains"],
                        steps=spec["steps"],
                        burn=spec["burn"],
                    )
                )(row)
            )(tasks, keys)
        )
        return lambda tasks, salt: np.asarray(
            run(values, as_arrays(tasks), keys_for(seeds["chain"], salt, shape=shape))
        )

    samplers = {
        "start": rows_of("classical", {"hyper": hypers[best]}),
        "initial": rows_of(form, parameters),
        "learned": rows_of(form, learned),
    }
    explained = None
    if form == "residual":
        linear, explained = learning.distill(
            form,
            learned,
            screen,
            k_distill,
            chains=spec["chains"],
            steps=spec["distill"]["steps"],
            every=spec["distill"]["every"],
        )
        samplers["linear"] = rows_of("linear", linear)
    arms, arrays, transfer = {}, {}, {}
    for s, (name, run) in enumerate(samplers.items()):
        arrays[name] = run(sets["confirmation"], 0)
        arms[name] = reduce(
            arrays[name],
            sets["confirmation"],
            [seeds["bootstrap"], 40, s],
            spec["bootstrap"],
        )
    for c, change in enumerate(spec["transfer"]):
        name = label(change)
        transfer[name] = {
            arm: reduce(
                run(sets[name], 50 + c),
                sets[name],
                [seeds["bootstrap"], 50, c, s],
                spec["bootstrap"],
            )
            for s, (arm, run) in enumerate(samplers.items())
            if arm != "initial"
        }
    flat = np.asarray(ravel_pytree(learned)[0])
    np.savez_compressed(cell_folder() / "learned.npz", parameters=flat, **arrays)
    return {
        "metrics": {
            "separation": spec["separation"],
            "dimension": d,
            "frame": frame,
            "form": form,
            "budget": spec["chains"] * spec["steps"],
            "start": setting,
            "hyper": [float(v) for v in np.exp(learned["hyper"])],
            "arms": arms,
            "transfer_results": transfer,
            "history": history,
            "linear_explained": explained,
            "training_seconds": seconds,
            # Gradients of the target that training spent.
            "training_gradients": int(
                training["iterations"]
                * 2
                * training["pairs"]
                * training["tasks"]
                * spec["chains"]
                * spec["steps"]
            ),
            "parameter_file": "learned.npz",
        },
        "estimands": ["mode_mass_rmse", "second_moment_rmse"],
    }
