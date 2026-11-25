"""How much can a metric gain a sampler, and what may one metric evaluation cost?"""

from dataclasses import asdict

import jax
import jax.numpy as jnp
import numpy as np
from bnn.target.synthetic import Task
from runner import CellSpec, Design, ResourceHint, cell_folder, finite

from . import kernels, metrics, targets
from .chains import reduce_cell, run_adapted, run_fixed

QUESTION = (
    "How much can a metric gain a sampler beyond an adapted constant one, and "
    "what may one metric evaluation cost?"
)
# Task parameters by family: Gaussians with condition 100 and 10,000, bananas
# with bend 0.1 and 0.4, and Neal's funnel.
SHAPES = {
    "gaussian": ({"condition": 100.0}, {"condition": 1e4}),
    "banana": ({"bend": 0.1}, {"bend": 0.4}),
    "funnel": ({"funnel": 3.0},),
}
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
    *(
        (family, d, SPARSE, shape)
        for family in ("banana", "funnel")
        for shape in SHAPES[family]
        for d in DIMENSIONS[1:]
    ),
)
# Gradients per chain after adaptation at which estimates are read.
RUNGS = (512, 2048, 8192, 32768)
ACCEPTANCES = (0.15, 0.25, 0.45, 0.65, 0.85)
TARGETS = 8
# The explicit metric needs an eigendecomposition per state.
EXPLICIT = 32
SEED = 620014
ROLES = ("starts", "screen", "chain", "adapt", "bootstrap")
WALLTIME = {8: 20, 32: 30, 128: 60}


def factors(profile, family, dimension, hard, shape):
    smoke = profile == "smoke"
    task = Task(family, "screen", 0, SEED, dimension, hard=hard, **shape)
    return {
        "profile": profile,
        "family": family,
        "dimension": dimension,
        "path": "dense" if hard is None else "sparse",
        "shape": shape,
        "task": asdict(task),
        "targets": 2 if smoke else TARGETS,
        "chains": 4 if smoke else 8,
        "screen_chains": 8 if smoke else 64,
        # Largest number of leapfrog steps of a transition: of the chain without
        # a metric, and of chains with one.
        "lengths": [1, 4] if smoke else [1, 4, 16, 64],
        "metric_lengths": [1, 4] if smoke else [1, 4, 16],
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
        "jitter": 0.2,
        # Initial step size times D^(1/4), in coordinates with unit stiffness.
        "initial_step": 0.5,
        "explicit": dimension <= EXPLICIT,
        # Cost of a Hessian-vector product in gradients.
        "hvp_cost": 2.0,
        "bootstrap": 100 if smoke else 1000,
    }


def design():
    cells = []
    for profile in ("smoke", "pilot"):
        conditions = (
            [("banana", 8, None, {"bend": 0.1})] if profile == "smoke" else CONDITIONS
        )
        for family, dimension, hard, shape in conditions:
            spec = factors(profile, family, dimension, hard, shape)
            task = Task(**spec["task"])
            cells.append(
                CellSpec(
                    "headroom",
                    spec,
                    {
                        role: task.stream(f"learned-metric:{profile}:{role}")
                        for role in ROLES
                    },
                    role="pilot",
                    expected_schema=("metrics", "estimands"),
                    resources=ResourceHint(
                        memory_gb=4, walltime_minutes=WALLTIME[dimension]
                    ),
                )
            )
    return Design(QUESTION, ("smoke", "pilot"), ROLES, tuple(cells))


# ----------------------------------------------------------------------------
# One cell


def measure(cell):
    spec, seeds = cell.factors, cell.seeds
    task = Task(**spec["task"])
    frame = targets.frame(task)
    d, count, chains = task.dimension, spec["targets"], spec["chains"]

    def rng(role, *index):
        return np.random.default_rng([seeds[role], *index])

    def key(role, *index):
        return jax.random.key(int(rng(role, *index).integers(2**32)))

    def keys(*index):
        return {"screen": key("screen", *index), "chain": key("chain", *index)}

    runs = {}
    with jax.enable_x64():
        energy = targets.potential(task)
        forward, inverse = targets.maps(task)
        estimand = targets.estimands(task)
        screen = jnp.asarray(targets.sample(task, spec["screen_chains"], rng("starts")))
        starts = jnp.asarray(
            targets.sample(task, count * chains, rng("starts", 1))
        ).reshape(count, chains, d)
        # In coordinates with unit stiffness, and where the stiffest mean
        # curvature of the target sets the step.
        unit = spec["initial_step"] * d**-0.25
        stiff = unit / np.sqrt(frame.curvatures.max())
        none = {"none": jnp.zeros(())}

        def arm(enter, group, initial):
            """An arm from the map of aligned positions to its chain's coordinates."""
            flat = enter(starts.reshape(-1, d))
            return {
                "screen": enter(screen),
                "starts": flat.reshape(count, chains, d),
                "group": group,
                "initial": initial,
            }

        def scaled(scales, initial):
            """A constant diagonal metric as an arm of the whitened chain."""
            scales = jnp.asarray(scales)
            return arm(lambda u: u / scales, {"factor": jnp.diag(scales)}, initial)

        fixed = {
            "identity": scaled(np.ones(d), stiff),
            "covariance_exact": scaled(np.sqrt(frame.variances), unit),
            "curvature_exact": scaled(1 / np.sqrt(frame.curvatures), unit),
        }
        hessian = jax.jit(jax.vmap(jax.hessian(energy)))

        def curvature(draws):
            recorded = draws.shape[2]
            picks = np.linspace(recorded - 1, 0, spec["hessians"] // chains).astype(int)
            states = draws[:, :, picks].reshape(-1, d)
            hessians = np.asarray(hessian(jnp.asarray(states)))
            return metrics.curvature_factor(hessians.reshape(count, -1, d, d)), len(
                picks
            ) * chains

        def covariance(draws):
            return metrics.covariance_factor(draws.reshape(count, -1, d)), 0

        # Adapted arms take the target acceptance screened for their exact limit.
        adapted = {
            "covariance": (covariance, "covariance_exact"),
            "curvature": (curvature, "curvature_exact"),
        }
        for length in spec["lengths"]:
            sampler = f"hmc{length}"
            metric = length in spec["metric_lengths"]
            kernel = kernels.hmc(
                lambda y, group: energy(group["factor"] @ y),
                lambda y, group: estimand(group["factor"] @ y),
                length,
                spec["jitter"],
            )
            names = list(fixed) if metric else ["identity"]
            records = run_fixed(
                kernel, [fixed[name] for name in names], length, spec, keys(length)
            )
            runs |= {(name, sampler): r for name, r in zip(names, records, strict=True)}
            if not metric:
                continue
            for a, (name, (estimate, limit)) in enumerate(adapted.items()):
                runs[name, sampler] = run_adapted(
                    kernel,
                    estimate,
                    {},
                    starts,
                    length,
                    runs[limit, sampler]["target_acceptance"],
                    stiff,
                    spec,
                    key("adapt", length, a),
                )
            kernel = kernels.hmc(
                lambda y, group: 0.5 * y @ y,
                lambda y, group: estimand(inverse(y)),
                length,
                spec["jitter"],
            )
            (runs["transport", sampler],) = run_fixed(
                kernel,
                [arm(jax.vmap(forward), none, unit)],
                length,
                spec,
                keys(length, 1),
            )
        geometries = {
            "metric": (kernels.pullback(forward, inverse, False), none, unit),
            "metric_drift": (kernels.pullback(forward, inverse, True), none, unit),
        }
        if spec["explicit"]:
            # The largest variance of the target bounds the proposal's.
            alpha = {"alpha": jnp.asarray(float(frame.variances.max()))}
            geometries["softabs"] = (
                kernels.softabs(lambda u, group: energy(u)),
                alpha,
                unit,
            )
        for a, (name, (geometry, group, initial)) in enumerate(geometries.items()):
            kernel = kernels.pmala(
                lambda u, group: energy(u),
                lambda u, group: estimand(u),
                geometry,
                spec["jitter"],
            )
            (runs[name, "mala"],) = run_fixed(
                kernel, [arm(lambda u: u, group, initial)], 1, spec, keys(0, a)
            )
    metrics_out = {
        "family": spec["family"],
        "path": spec["path"],
        "shape": spec["shape"],
        "dimension": d,
        "hard": task.hard,
        "estimands": list(targets.ESTIMANDS[task.family]),
        "targets": count,
        "chains": chains,
        "hvp_cost": spec["hvp_cost"],
        "curves": reduce_cell(spec, runs, seeds["bootstrap"]),
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
                | ({"screen_rmse": run["screen_rmse"]} if "screen_rmse" in run else {})
                for (name, sampler), run in runs.items()
                if name == arm
            }
            for arm in dict.fromkeys(name for name, _ in runs)
        },
        "repeat_file": "repeats.npz",
    }
    np.savez_compressed(
        cell_folder() / "repeats.npz",
        **{
            f"{arm}_{sampler}": run["estimates"] for (arm, sampler), run in runs.items()
        },
    )
    return {
        "metrics": metrics_out,
        "estimands": ["normalized_estimand_rmse", "metric_evaluation_break_even"],
    }
