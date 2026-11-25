"""Kinetic fields trained as published, on small regression networks.

Kim et al. (2024) train on the Bayesian model average of held-out data. Here a
task is a one-dimensional regression with a gap in its inputs and a tanh
network of one hidden layer under a standard normal prior. Chains start from
short optimizations and spend a fixed number of gradients, few enough that a
tuned classical sampler has not yet converged.

``network``: tuned classical samplers, read on the trained loss and against a
posterior reference of the predictive mean and spread.

``network_learned``: fields trained on the model-average loss over tasks, read
the same way on tasks they have not seen, and without change on the mixtures.
"""

from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
import optax
from bnn.samplers.tempered_smc import SMCConfig, tempered_smc
from jax.flatten_util import ravel_pytree
from runner import CellSpec, Design, ResourceHint, cell_folder

from . import dynamics, frontier, kinetic, learning

QUESTION = (
    "Do kinetic fields trained on the model-average loss predict better per "
    "gradient than tuned classical samplers on unseen regression tasks, and "
    "what happens to the posterior they sample?"
)
ROLES = ("screen", "confirmation", "train", "chain", "reference", "bootstrap")
SEED = 630020
HIDDEN = 16
DIMENSION = 3 * HIDDEN + 1
NOISE = 0.25
# Scale of the inputs inside the network and of its output.
INPUT, OUTPUT = 2.0, 2.0
# Step size, friction and temperature; the stiffest curvature is of order 10^3.
# With a diagonal mass the step is in units of a coordinate's own period.
# The loss of held-out data can prefer another temperature than the posterior's,
# so the classical grids reach well beyond 1 on both sides.
TEMPERATURES = (0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0, 5.6)
BAOAB = {
    "step": (0.0025, 0.005, 0.01, 0.02, 0.03),
    "friction": (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0),
    "temperature": TEMPERATURES,
}
# BAOAB on the bridge between posterior and prior: the likelihood to the power
# of one minus the share.
BRIDGE = {
    "step": (0.0025, 0.005, 0.01, 0.02),
    "friction": (0.3, 1.0),
    "temperature": (1.0, 2.0, 2.8, 4.0),
    "share": (0.2, 0.4, 0.6),
}
# Step size of the diagonal mass's burn-in, at unit mass.
WARM = 0.005
DIAGONAL = {
    "step": (0.05, 0.1, 0.15, 0.2),
    "friction": (0.3, 1.0, 3.0),
    "temperature": TEMPERATURES,
}
# The grid a learner's start is chosen from, and the classical cell's own.
EULER = {
    "step": (0.005, 0.01, 0.02, 0.03),
    "friction": (1.0, 3.0, 10.0),
    "temperature": (0.5, 0.7, 1.0, 1.4, 2.0),
}
EULER_WIDE = {
    "step": (0.005, 0.01, 0.02),
    "friction": (1.0, 3.0, 10.0),
    "temperature": TEMPERATURES,
}
HMC = {"step": (0.01, 0.02, 0.03, 0.045), "length": (8, 32)}
FORMS = ("residual", "published")


def seed(profile, role):
    return int(
        frontier.fingerprint([SEED, profile, role], namespace="kinetic-frontier")[:16],
        16,
    )


def axes(grid, smoke):
    """A sampler's grid; smoke cells keep two values each, temperature 1 among them."""
    if not smoke:
        return {name: list(values) for name, values in grid.items()}
    return {
        name: [0.7, 1.0] if name == "temperature" else list(values[:2])
        for name, values in grid.items()
    }


def shared(profile):
    smoke = profile == "smoke"
    return {
        "profile": profile,
        "observations": 32,
        "validation": 32,
        "inputs": 25,
        "chains": 4 if smoke else 8,
        "steps": 96 if smoke else 512,
        "burn": 32 if smoke else 128,
        "descent": 32 if smoke else 256,
        "screen_tasks": 4 if smoke else 64,
        "targets": 2 if smoke else 8,
        "trials": 2 if smoke else 16,
        "euler": axes(EULER, smoke),
        "bootstrap": 50 if smoke else 1000,
    }


def classical_factors(profile):
    smoke = profile == "smoke"
    return shared(profile) | {
        "euler": axes(EULER_WIDE, smoke),
        "baoab": axes(BAOAB, smoke),
        "diagonal": axes(DIAGONAL, smoke),
        "bridge": axes(BRIDGE, smoke),
        "hmc": {k: list(v[:1] if smoke else v) for k, v in HMC.items()},
        "reference": {
            "particles": 128 if smoke else 4096,
            "moves": 4 if smoke else 16,
            "chains": 8 if smoke else 64,
            "gradients": 256 if smoke else 32768,
            "step": 0.02,
            "length": 16,
        },
    }


def learned_factors(profile, form):
    smoke = profile == "smoke"
    return shared(profile) | {
        "form": form,
        "training": {
            "iterations": 4 if smoke else 200,
            "pairs": 2 if smoke else 16,
            "tasks": 2 if smoke else 8,
            "sigma": 0.03,
            "rate": 0.01,
            "interval": 2 if smoke else 25,
        },
        # The mixtures the trained fields are read on without change.
        "mixture": {"separation": 3.5, "dimension": 8 if smoke else 32},
    }


def design():
    cells = []
    for profile in ("smoke", "pilot"):
        seeds = {role: seed(profile, role) for role in ROLES}
        cells.append(
            CellSpec(
                "network",
                classical_factors(profile),
                seeds,
                role="pilot",
                expected_schema=("metrics", "estimands"),
                resources=ResourceHint(memory_gb=6, walltime_minutes=180),
            )
        )
        for form in FORMS[:1] if profile == "smoke" else FORMS:
            cells.append(
                CellSpec(
                    "network_learned",
                    learned_factors(profile, form),
                    seeds,
                    role="pilot",
                    expected_schema=("metrics", "estimands"),
                    resources=ResourceHint(memory_gb=4, walltime_minutes=300),
                )
            )
    return Design(QUESTION, ("smoke", "pilot"), ROLES, tuple(cells))


# ----------------------------------------------------------------------------
# Tasks and posterior


def tasks(rng, count, spec):
    """Regression tasks as arrays: data, held-out data and prediction inputs."""

    def inputs(size):
        side = rng.integers(2, size=(count, size)) * 2 - 1
        return side * rng.uniform(0.5, 2.0, (count, size))

    amplitude = rng.normal(0, 0.6, (count, 1, 3))
    frequency = rng.uniform(0.5, 3.0, (count, 1, 3))
    phase = rng.uniform(0, 2 * np.pi, (count, 1, 3))

    def function(x):
        return np.sum(amplitude * np.sin(frequency * x[..., None] + phase), axis=-1)

    x, held = inputs(spec["observations"]), inputs(spec["validation"])
    return {
        "x": x,
        "y": function(x) + NOISE * rng.normal(size=x.shape),
        "held_x": held,
        "held_y": function(held) + NOISE * rng.normal(size=held.shape),
        "grid": np.broadcast_to(
            np.linspace(-3, 3, spec["inputs"]), (count, spec["inputs"])
        ),
    }


def as_arrays(task, dtype=jnp.float32):
    return {name: jnp.asarray(value, dtype) for name, value in task.items()}


def predict(theta, x):
    """Network outputs of one position at inputs ``x``."""
    w, b, v = (theta[i * HIDDEN : (i + 1) * HIDDEN] for i in range(3))
    hidden = jnp.tanh(INPUT * x[:, None] * w + b)
    return theta[-1] + OUTPUT / np.sqrt(HIDDEN) * hidden @ v


def log_likelihood(theta, task):
    return -0.5 * jnp.sum((task["y"] - predict(theta, task["x"])) ** 2) / NOISE**2


def potential(theta, task):
    return 0.5 * jnp.sum(theta**2) - log_likelihood(theta, task)


def force(theta, task):
    return jax.vmap(jax.grad(potential), in_axes=(0, None))(theta, task)


def observe(theta, task):
    """Held-out likelihoods, and predictions and their squares on the grid."""
    held = predict(theta, task["held_x"])
    density = jnp.exp(-0.5 * ((task["held_y"] - held) / NOISE) ** 2) / (
        NOISE * np.sqrt(2 * np.pi)
    )
    grid = predict(theta, task["grid"])
    return jnp.concatenate([density, grid, grid**2])


def starts(key, task, chains, steps):
    """Prior draws after a short optimization, the same for every sampler."""
    theta = jax.random.normal(key, (chains, DIMENSION), task["x"].dtype)
    optimizer = optax.adam(0.03)

    def one(carry, _):
        theta, state = carry
        updates, state = optimizer.update(force(theta, task), state)
        return (optax.apply_updates(theta, updates), state), None

    (theta, _), _ = jax.lax.scan(
        one, (theta, optimizer.init(theta)), None, length=steps
    )
    return theta


def reduce_sums(sums, spec):
    """Model-average loss, predictive mean and spread, and whether all is finite."""
    held, count = spec["validation"], spec["inputs"]
    loss = -jnp.mean(jnp.log(jnp.maximum(sums[:held], 1e-30)))
    mean = sums[held : held + count]
    spread = jnp.sqrt(jnp.maximum(sums[held + count :] - mean**2, 0.0))
    row = jnp.concatenate([loss[None], mean, spread])
    finite = jnp.all(jnp.isfinite(row))
    # A diverged trial predicts nothing: the loss of a density of 1e-30.
    row = jnp.where(finite, row, jnp.zeros_like(row).at[0].set(-np.log(1e-30)))
    return jnp.concatenate([row, finite[None].astype(row.dtype)])


def trial(sampler, task, key, spec):
    """One trial: ``[loss, mean on the grid, spread on the grid, finite]``."""
    start, step = sampler
    k1, k2, k3 = jax.random.split(key, 3)
    state = start(k2, starts(k1, task, spec["chains"], spec["descent"]), task)

    def one(carry, inputs):
        state, sums = carry
        key, keep = inputs
        state = step(key, state, task)
        seen = jax.vmap(observe, in_axes=(0, None))(state[0], task).mean(0)
        return (state, sums + keep * seen), None

    keep = (jnp.arange(spec["steps"]) >= spec["burn"]).astype(task["x"].dtype)
    size = spec["validation"] + 2 * spec["inputs"]
    (_, sums), _ = jax.lax.scan(
        one,
        (state, jnp.zeros(size, task["x"].dtype)),
        (jax.random.split(k3, spec["steps"]), keep),
    )
    return reduce_sums(sums / (spec["steps"] - spec["burn"]), spec)


def runner(make, spec):
    """``(parameters, tasks, keys[task, trial]) -> rows[task, trial]``."""
    return jax.jit(
        lambda parameters, tasks_, keys: jax.vmap(
            lambda task, row: jax.vmap(
                lambda key: trial(make(parameters), task, key, spec)
            )(row)
        )(tasks_, keys)
    )


def euler(form):
    return lambda parameters: dynamics.euler(form, parameters, force)


def baoab(parameters):
    return dynamics.baoab(parameters, force)


def diagonal(spec):
    """BAOAB whose diagonal mass is the mean squared gradient of its burn-in.

    Until ``burn`` it runs with unit mass at the step size ``WARM``. For a
    Gaussian the mean squared gradient is the diagonal of the precision, so
    afterwards the step size is in units of the stiffest coordinate's period.
    No mass is below 1, the precision of the prior: a coordinate whose burn-in
    saw small gradients would otherwise take steps without bound.
    """
    warm, burn = WARM, spec["burn"]

    def make(parameters):
        step_size, friction, temperature = jnp.exp(parameters["hyper"])

        def start(key, theta, task):
            r = jnp.sqrt(temperature) * jax.random.normal(key, theta.shape, theta.dtype)
            g = force(theta, task)
            return theta, r, g, jnp.zeros_like(theta), jnp.zeros((), jnp.int32)

        def step(key, state, task):
            theta, r, g, squares, count = state
            adapted = count >= burn
            mass = jnp.where(adapted, jnp.maximum(squares / burn, 1.0), 1.0)
            eps = jnp.where(adapted, step_size, warm)
            # Momenta stay thermal when the mass changes.
            r = jnp.where(count == burn, r * jnp.sqrt(mass), r)
            # The friction is per unit of each coordinate's own time.
            decay = jnp.exp(-friction * eps)
            r = r - 0.5 * eps * g
            theta = theta + 0.5 * eps * r / mass
            noise = jax.random.normal(key, r.shape, r.dtype)
            r = decay * r + jnp.sqrt(temperature * mass * (1 - decay**2)) * noise
            theta = theta + 0.5 * eps * r / mass
            g = force(theta, task)
            squares = squares + jnp.where(adapted, 0.0, g**2)
            return theta, r - 0.5 * eps * g, g, squares, count + 1

        return start, step

    return make


def exact(spec, length):
    """Exact HMC at the same number of gradients, from the same starts."""
    run = kinetic.hmc(
        kinetic.FAMILIES["gaussian"],
        potential,
        jax.grad(potential),
        observe,
        lambda theta, task, previous: previous,
    )
    transitions = kinetic.transitions_for(spec["steps"], length)
    burn = transitions * spec["burn"] // spec["steps"]

    def one(step, task, key):
        k1, k2 = jax.random.split(key)
        sums, _, _, _ = run(
            k2,
            starts(k1, task, spec["chains"], spec["descent"]),
            step,
            task,
            transitions=transitions,
            burn=burn,
            length=length,
        )
        return reduce_sums(sums.mean(0), spec)

    return jax.jit(
        lambda step, tasks_, keys: jax.vmap(
            lambda task, row: jax.vmap(lambda key: one(step, task, key))(row)
        )(tasks_, keys)
    )


def _prior(x, params):
    return -0.5 * jnp.sum(x**2)


def bridge(parameters):
    return dynamics.bridge(parameters, force, lambda theta, task: theta)


def ensemble(parameters):
    """The starts themselves, unmoved: what the optimizations alone predict."""
    return (lambda key, theta, task: (theta,)), (lambda key, state, task: state)


def _likelihood(x, params):
    return log_likelihood(x, params)


def reference(spec, task_set, seed_):
    """Predictive mean and spread by tempered SMC and by long exact chains.

    Returns both per task. The chains are the reference, and the population
    checks that they have not missed a region.
    """
    settings = spec["reference"]
    count = len(task_set["x"])
    out = {"smc": [], "hmc": [], "smc_loss": [], "hmc_loss": [], "hmc_rhat": []}
    with jax.enable_x64():
        seen = jax.jit(jax.vmap(observe, in_axes=(0, None)))
        run = jax.jit(
            kinetic.hmc(
                kinetic.FAMILIES["gaussian"],
                potential,
                jax.grad(potential),
                observe,
                lambda theta, task, previous: previous,
            ),
            static_argnames=("transitions", "burn", "length"),
        )
        for t in range(count):
            task = {k: jnp.asarray(v[t], jnp.float64) for k, v in task_set.items()}
            rng = np.random.default_rng([seed_, t])
            result = tempered_smc(
                _prior,
                _likelihood,
                task,
                rng.normal(size=(settings["particles"], DIMENSION)),
                SMCConfig(n_mcmc=settings["moves"], seed=int(rng.integers(2**31))),
            )
            row = reduce_sums(seen(jnp.asarray(result.positions), task).mean(0), spec)
            out["smc"].append(np.asarray(row))
            key = jax.random.key(int(rng.integers(2**32)))
            k1, k2 = jax.random.split(key)
            transitions = kinetic.transitions_for(
                settings["gradients"], settings["length"]
            )
            sums, _, _, _ = run(
                k2,
                starts(k1, task, settings["chains"], spec["descent"]),
                jnp.asarray(settings["step"]),
                task,
                transitions=transitions,
                burn=transitions // 4,
                length=settings["length"],
            )
            out["hmc"].append(np.asarray(reduce_sums(sums.mean(0), spec)))
            # Spread of the chains' predictive means against the posterior's
            # spread: a potential scale reduction of the predictions.
            held, grid = spec["validation"], spec["inputs"]
            means = np.asarray(sums[:, held : held + grid])
            total = np.asarray(sums[:, held + grid :]).mean(0) - means.mean(0) ** 2
            out["hmc_rhat"].append(
                float(np.sqrt(np.max(total / np.maximum(total - means.var(0), 1e-12))))
            )
    return {name: np.array(value) for name, value in out.items()}


# ----------------------------------------------------------------------------
# Reduction


def summary(rows, seed_, count):
    """Mean loss of ``rows[target, trial]`` with an interval, and diverged trials."""
    rows = np.asarray(rows, np.float64)
    loss = rows[..., 0]
    groups = np.repeat(np.arange(rows.shape[0]), rows.shape[1])
    samples = frontier.nested_resamples(groups, count, np.random.default_rng(seed_))
    return {
        "loss": float(loss.mean()),
        "loss_interval": [
            float(v)
            for v in np.quantile(loss.reshape(-1)[samples].mean(1), [0.025, 0.975])
        ],
        "loss_targets": loss.mean(1).tolist(),
        "diverged": float(1 - rows[..., -1].mean()),
        # Predictive mean and spread per target, averaged over trials, and the
        # spread of the predictive mean between trials.
        "mean": rows[..., 1 : 1 + (rows.shape[-1] - 2) // 2].mean(1).tolist(),
        "spread": rows[..., 1 + (rows.shape[-1] - 2) // 2 : -1].mean(1).tolist(),
        "mean_between_trials": rows[..., 1 : 1 + (rows.shape[-1] - 2) // 2]
        .std(1)
        .tolist(),
    }


def keys_for(seed_, *index, shape):
    return frontier.keys_for(seed_, *index, shape=shape)


def task_sets(spec, seeds):
    return {
        "screen": tasks(
            np.random.default_rng([seeds["screen"], 0]), spec["screen_tasks"], spec
        ),
        "confirmation": tasks(
            np.random.default_rng([seeds["confirmation"], 0]), spec["targets"], spec
        ),
    }


def screen_loss(rows):
    return np.asarray(rows)[..., 0, 0].mean(-1)


# ----------------------------------------------------------------------------
# Cells


def measure_classical(cell):
    spec, seeds = cell.factors, cell.seeds
    sets = task_sets(spec, seeds)
    screen, confirmation = as_arrays(sets["screen"]), as_arrays(sets["confirmation"])
    shape = (spec["targets"], spec["trials"])
    arms, seconds = {}, {}

    start = perf_counter()
    samplers = (
        ("euler", "baoab", euler("classical")),
        ("baoab", "baoab", baoab),
        ("diagonal", "baoab", diagonal(spec)),
        ("bridge", "bridge", bridge),
    )
    for f, (kind, order, make) in enumerate(samplers):
        names, settings = frontier.grid(order, spec[kind])
        hypers = jnp.log(jnp.asarray(settings, jnp.float32))
        run = runner(make, spec)
        screened = np.stack(
            [
                screen_loss(
                    run(
                        {"hyper": h},
                        screen,
                        keys_for(seeds["screen"], f, shape=(spec["screen_tasks"], 1)),
                    )
                )
                for h in hypers
            ]
        )
        # Tuned for the loss at any temperature, and at the posterior's.
        choices = {kind: np.arange(len(settings))}
        choices[f"{kind}_unit"] = np.flatnonzero(settings[:, 2] == 1.0)
        for c, (name, candidates) in enumerate(choices.items()):
            best = int(candidates[np.argmin(screened[candidates])])
            rows = run(
                {"hyper": hypers[best]},
                confirmation,
                keys_for(seeds["chain"], f, c, shape=shape),
            )
            arms[name] = {
                "setting": dict(zip(names, map(float, settings[best]), strict=True)),
                "screen_loss": float(screened[best]),
            } | summary(rows, [seeds["bootstrap"], f, c], spec["bootstrap"])
        arms[kind]["grid"] = [
            dict(zip(names, map(float, s), strict=True)) | {"screen_loss": float(v)}
            for s, v in zip(settings, screened, strict=True)
        ]
    rows = runner(ensemble, spec)(
        {}, confirmation, keys_for(seeds["chain"], 5, shape=shape)
    )
    arms["starts"] = summary(rows, [seeds["bootstrap"], 5], spec["bootstrap"])
    seconds["grids"] = perf_counter() - start

    start = perf_counter()
    with jax.enable_x64():
        screen64, confirmation64 = (
            as_arrays(sets[name], jnp.float64) for name in ("screen", "confirmation")
        )
        candidates = []
        for i, length in enumerate(spec["hmc"]["length"]):
            run = exact(spec, length)
            for j, step in enumerate(spec["hmc"]["step"]):
                rows = run(
                    jnp.asarray(step),
                    screen64,
                    keys_for(
                        seeds["screen"], 10, i, j, shape=(spec["screen_tasks"], 1)
                    ),
                )
                candidates.append((float(screen_loss(rows)), length, step))
        loss, length, step = min(candidates)
        rows = exact(spec, length)(
            jnp.asarray(step), confirmation64, keys_for(seeds["chain"], 10, shape=shape)
        )
        arms["hmc"] = {
            "setting": {"step": step, "length": length},
            "screen_loss": loss,
        } | summary(rows, [seeds["bootstrap"], 10], spec["bootstrap"])
    seconds["hmc"] = perf_counter() - start

    start = perf_counter()
    exact_ = reference(spec, sets["confirmation"], seeds["reference"])
    seconds["reference"] = perf_counter() - start
    count = spec["inputs"]
    np.savez_compressed(cell_folder() / "reference.npz", **exact_)
    return {
        "metrics": {
            "dimension": DIMENSION,
            "budget": spec["chains"] * spec["steps"],
            "grid": sets["confirmation"]["grid"][0].tolist(),
            "arms": arms,
            "reference_results": {
                name: {
                    "loss": exact_[name][:, 0].tolist(),
                    "mean": exact_[name][:, 1 : 1 + count].tolist(),
                    "spread": exact_[name][:, 1 + count : -1].tolist(),
                }
                for name in ("smc", "hmc")
            }
            | {"hmc_rhat": exact_["hmc_rhat"].tolist()},
            "seconds": seconds,
        },
        "estimands": [
            "model_average_loss",
            "predictive_mean_error",
            "predictive_spread_ratio",
        ],
    }


def evolve_loss(form, spec):
    rows = runner(euler(form), spec)

    def loss(parameters, tasks_, key):
        out = rows(parameters, tasks_, jax.random.split(key, (len(tasks_["x"]), 1)))
        value = jnp.mean(out[..., 0])
        return value, value, 1 - jnp.mean(out[..., -1])

    return loss


def measure_learned(cell):
    spec, seeds = cell.factors, cell.seeds
    form = spec["form"]
    sets = task_sets(spec, seeds)
    screen, confirmation = as_arrays(sets["screen"]), as_arrays(sets["confirmation"])
    shape = (spec["targets"], spec["trials"])

    # The learner starts from the SGHMC setting with the best loss on the screen tasks.
    names, settings = frontier.grid("euler", spec["euler"])
    hypers = jnp.log(jnp.asarray(settings, jnp.float32))
    run = runner(euler("classical"), spec)
    screened = np.stack(
        [
            screen_loss(
                run(
                    {"hyper": h},
                    screen,
                    keys_for(seeds["screen"], 0, shape=(spec["screen_tasks"], 1)),
                )
            )
            for h in hypers
        ]
    )
    best = int(np.argmin(screened))
    key = jax.random.key(
        int(np.random.default_rng([seeds["train"], 0]).integers(2**32))
    )
    k_init, k_train = jax.random.split(key)
    parameters = learning.initial(form, settings[best], k_init)

    def draw(iteration):
        rng = np.random.default_rng([seeds["train"], 1, iteration])
        return as_arrays(tasks(rng, spec["training"]["tasks"], spec))

    start = perf_counter()

    def report(entry):
        print(f"{form} [{perf_counter() - start:6.0f}s] {entry}", flush=True)

    learned, history = learning.evolve(
        evolve_loss(form, spec),
        parameters,
        draw,
        screen,
        spec["training"],
        k_train,
        report,
    )
    seconds = perf_counter() - start
    arms = {}
    for s, (name, kind, values) in enumerate(
        (
            ("start", "classical", {"hyper": hypers[best]}),
            ("initial", form, parameters),
            ("learned", form, learned),
        )
    ):
        rows = runner(euler(kind), spec)(
            values, confirmation, keys_for(seeds["chain"], 20, shape=shape)
        )
        arms[name] = summary(rows, [seeds["bootstrap"], 20, s], spec["bootstrap"])

    # Without change on two-mode mixtures in a rotated frame.
    mixture = frontier.shared(spec["profile"], **spec["mixture"]) | {"transfer": []}
    mixture_seeds = {
        role: frontier.seed(
            spec["profile"],
            spec["mixture"]["separation"],
            spec["mixture"]["dimension"],
            role,
        )
        for role in frontier.ROLES
    }
    mixture_sets = frontier.task_sets(mixture, mixture_seeds, "rotated")
    mixture_arms = {}
    for s, (name, kind, values) in enumerate(
        (("start", "classical", {"hyper": hypers[best]}), ("learned", form, learned))
    ):
        rows = jax.jit(
            lambda p, tasks_, keys, kind=kind: jax.vmap(
                lambda task, row: jax.vmap(
                    lambda key: dynamics.trial(
                        dynamics.euler(kind, p),
                        task,
                        key,
                        chains=mixture["chains"],
                        steps=mixture["steps"],
                        burn=mixture["burn"],
                    )
                )(row)
            )(tasks_, keys)
        )(
            values,
            frontier.as_arrays(mixture_sets["confirmation"]),
            keys_for(seeds["chain"], 30, shape=(mixture["targets"], mixture["trials"])),
        )
        mixture_arms[name] = frontier.reduce(
            rows,
            mixture_sets["confirmation"],
            [seeds["bootstrap"], 30, s],
            spec["bootstrap"],
        )
    np.savez_compressed(
        cell_folder() / "learned.npz", parameters=np.asarray(ravel_pytree(learned)[0])
    )
    return {
        "metrics": {
            "form": form,
            "dimension": DIMENSION,
            "budget": spec["chains"] * spec["steps"],
            "start": dict(zip(names, map(float, settings[best]), strict=True)),
            "hyper": [float(v) for v in np.exp(learned["hyper"])],
            "arms": arms,
            "mixture_results": mixture_arms,
            "history": history,
            "training_seconds": seconds,
            "training_gradients": int(
                spec["training"]["iterations"]
                * 2
                * spec["training"]["pairs"]
                * spec["training"]["tasks"]
                * spec["chains"]
                * spec["steps"]
            ),
        },
        "estimands": [
            "model_average_loss",
            "predictive_mean_error",
            "predictive_spread_ratio",
        ],
    }


def measure(cell):
    return {"network": measure_classical, "network_learned": measure_learned}[
        cell.stage
    ](cell)
