"""Does a fitted flow reach the headroom of ideal coordinates on exact targets?"""

from dataclasses import asdict

import jax
import jax.numpy as jnp
import numpy as np
from bnn.target.synthetic import Task
from runner import CellSpec, Design, ResourceHint, cell_folder, finite

from . import targets
from .arms import compare
from .chains import reduce_cell

QUESTION = (
    "Does a sampler in the coordinates of a fitted flow need less total compute "
    "than the cheapest classical chain, once the fit is paid?"
)
# Bananas with bend 0.1 and 0.4, and Neal's funnel.
SHAPES = {"banana": ({"bend": 0.1}, {"bend": 0.4}), "funnel": ({"funnel": 3.0},)}
DIMENSIONS = (8, 32, 128)
# The sparse path keeps a family's structure in this many coordinates.
SPARSE = 8
# Family, dimension, hard block (None is the dense path) and shape.
CONDITIONS = (
    *(
        (family, d, None, shape)
        for family, shapes in SHAPES.items()
        for shape in shapes
        for d in DIMENSIONS
    ),
    *((family, 128, SPARSE, shape) for family in SHAPES for shape in SHAPES[family]),
)
# Gradients per chain after adaptation at which estimates are read.
RUNGS = (512, 2048, 8192, 32768)
ACCEPTANCES = (0.15, 0.25, 0.45, 0.65, 0.85)
SEED = 630119
ROLES = ("starts", "screen", "chain", "adapt", "flow", "bootstrap")
WALLTIME = {8: 120, 32: 300, 128: 420}
# Steps after which a flow is read. A step takes `batch` gradients, so the
# longest fit of a target costs 16 to 128 times the largest rung of one chain.
SNAPSHOTS = ((16, [1024, 8192, 65536]), (64, [1024, 8192, 32768]))


def snapshots(dimension):
    return next(
        (steps for limit, steps in SNAPSHOTS if dimension <= limit), [1024, 8192]
    )


def comparison(smoke, dimension):
    """What every part of the experiment shares."""
    spec = _comparison(smoke, dimension)
    if dimension < 128 or smoke:
        return spec
    # A chain in a flow takes about 0.6 ms per leapfrog step at D = 128, so
    # fewer chains run there, one trajectory length, and no flow on exact draws.
    return spec | {
        "targets": 4,
        "chains": 4,
        "flow_lengths": [16],
        "flow_acceptances": [0.65, 0.85],
        "flow_screen_chains": 16,
        "exact_fit": False,
        "fit": spec["fit"] | {"snapshots": [8192, 32768]},
    }


def _comparison(smoke, dimension):
    return {
        "targets": 2 if smoke else 8,
        "chains": 4 if smoke else 8,
        "screen_chains": 8 if smoke else 64,
        # Largest number of leapfrog steps of a transition: of the chain without
        # a metric, of chains with a constant one, and of chains in a flow.
        "lengths": [1, 4] if smoke else [1, 4, 16, 64],
        "metric_lengths": [1, 4] if smoke else [1, 4, 16],
        "flow_lengths": [4] if smoke else [4, 16],
        "rungs": [128, 512] if smoke else list(RUNGS),
        "screen_budget": 256 if smoke else 4096,
        "warmup": 48 if smoke else 200,
        # Draws per chain that each adaptation window records, and the gradients
        # between two of them.
        "windows": [16, 32] if smoke else [100, 200, 400],
        "window_gradients": 8,
        # States per window and target at which the curvature arm takes a Hessian,
        # and the Hessian-vector products of one.
        "hessians": 8 if smoke else 32,
        "curvature_products": dimension,
        "acceptances": [0.45, 0.65] if smoke else list(ACCEPTANCES),
        "flow_acceptances": [0.65] if smoke else [0.45, 0.65, 0.85],
        "flow_screen_chains": 8 if smoke else 32,
        "jitter": 0.2,
        # Initial step size times D^(1/4), in coordinates with unit stiffness.
        "initial_step": 0.5,
        # Cost of a Hessian-vector product in gradients.
        "hvp_cost": 2.0,
        "mixings": ["rotation", "permutation"],
        "fit": {
            "layers": 2 if smoke else 6,
            "hidden": 16 if smoke else max(64, dimension),
            "snapshots": [32, 64] if smoke else snapshots(dimension),
            "batch": 16 if smoke else 64,
            "rate": 1e-3,
            "quality_draws": 256 if smoke else 4096,
        },
        "bootstrap": 100 if smoke else 1000,
    }


def factors(profile, family, dimension, hard, shape):
    task = Task(family, "screen", 0, SEED, dimension, hard=hard, **shape)
    return {
        "profile": profile,
        "family": family,
        "dimension": dimension,
        "path": "dense" if hard is None else "sparse",
        "shape": shape,
        "task": asdict(task),
    } | comparison(profile == "smoke", dimension)


def short(spec):
    """A second, smaller cell of a condition whose flows take short trajectories.

    The first cells gave the flows 4 and 16 leapfrog steps at most. The exact map
    does best with one, so those cells understate what a well fitted flow gains.
    These repeat the comparison with 1 and 4 steps and shorter fits.
    """
    d = spec["dimension"]
    snapshots = {8: [1024, 8192], 32: [8192, 32768], 128: [8192]}[d]
    return spec | {
        "trajectories": "short",
        "flow_lengths": [1, 4],
        "targets": 8 if d == 8 else 4,
        "fit": spec["fit"] | {"snapshots": snapshots},
    }


# Conditions at D = 128 that get a cell with short trajectories: those whose
# flows had fitted at all after 8,192 steps.
SHORT_LARGE = (
    ("banana", None, 0.1),
    ("banana", SPARSE, 0.1),
    ("banana", SPARSE, 0.4),
    ("funnel", SPARSE, None),
)


def design():
    cells = []
    for profile in ("smoke", "pilot"):
        conditions = (
            [("banana", 8, None, {"bend": 0.1})] if profile == "smoke" else CONDITIONS
        )
        specs = [factors(profile, *condition) for condition in conditions]
        if profile == "pilot":
            specs += [
                short(spec)
                for spec in specs
                if spec["dimension"] < 128
                or (
                    spec["family"],
                    spec["task"]["hard"] if spec["path"] == "sparse" else None,
                    spec["shape"].get("bend"),
                )
                in SHORT_LARGE
            ]
        for spec in specs:
            dimension = spec["dimension"]
            task = Task(**spec["task"])
            cells.append(
                CellSpec(
                    "synthetic",
                    spec,
                    {
                        role: task.stream(f"learned-coordinates:{profile}:{role}")
                        for role in ROLES
                    },
                    role="pilot",
                    expected_schema=("metrics", "estimands"),
                    resources=ResourceHint(
                        cpu=2 if dimension > 64 else 1,
                        memory_gb=4 if dimension > 64 else 2,
                        walltime_minutes=WALLTIME[dimension],
                    ),
                )
            )
    return Design(QUESTION, ("smoke", "pilot"), ROLES, tuple(cells))


def report(spec, runs, fits, seed):
    """The saved reading of a cell's runs."""
    return {
        "curves": reduce_cell(spec, runs, seed),
        "fits": fits,
        "settings": {
            arm: {
                sampler: finite(
                    {
                        "target_acceptance": float(run["target_acceptance"]),
                        "acceptance": float(np.mean(run["acceptance"])),
                        "step_size": float(np.median(run["step"])),
                        "seconds": float(run["seconds"]),
                    }
                )
                for (name, sampler), run in runs.items()
                if name == arm
            }
            for arm in dict.fromkeys(name for name, _ in runs)
        },
        "repeat_file": "repeats.npz",
    }


def save_repeats(runs):
    np.savez_compressed(
        cell_folder() / "repeats.npz",
        **{
            f"{arm}_{sampler}": run["estimates"] for (arm, sampler), run in runs.items()
        },
    )


def measure(cell):
    spec, seeds = cell.factors, cell.seeds
    task = Task(**spec["task"])
    frame = targets.frame(task)
    d, count, chains = task.dimension, spec["targets"], spec["chains"]
    rng = np.random.default_rng([seeds["starts"]])
    with jax.enable_x64():
        energy = targets.potential(task)
        forward, inverse = targets.maps(task)
        estimand = targets.estimands(task)
        unit = spec["initial_step"] * d**-0.25

        def diagonal(scales):
            factor = jnp.diag(jnp.asarray(scales))
            return jnp.stack([factor] * count), factor, unit

        def draw(key, data, batch):
            return jax.vmap(inverse)(jax.random.normal(key, (batch, d), jnp.float64))

        problem = {
            "energy": lambda u, data: energy(u),
            "estimand": lambda u, data: estimand(u),
            "data": {"none": jnp.zeros(count)},
            "screen_data": {"none": jnp.zeros(())},
            "screen": jnp.asarray(targets.sample(task, spec["screen_chains"], rng)),
            "starts": jnp.asarray(targets.sample(task, count * chains, rng)).reshape(
                count, chains, d
            ),
            "constant": {
                "covariance_exact": diagonal(np.sqrt(frame.variances)),
                "curvature_exact": diagonal(1 / np.sqrt(frame.curvatures)),
            },
            # Where the stiffest mean curvature of the target sets the step.
            "stiff": unit / np.sqrt(frame.curvatures.max()),
            "unit": unit,
            "oracle": (forward, inverse),
            "draw": draw,
        }
        runs, fits = compare(spec, seeds, problem)
    save_repeats(runs)
    return {
        "metrics": {
            "family": spec["family"],
            "path": spec["path"],
            "shape": spec["shape"],
            "dimension": d,
            "estimands": list(targets.ESTIMANDS[task.family]),
        }
        | report(spec, runs, fits, seeds["bootstrap"]),
        "estimands": ["normalized_estimand_rmse", "flow_evaluation_break_even"],
    }
