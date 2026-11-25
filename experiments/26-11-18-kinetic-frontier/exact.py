"""What the momentum law can gain an exact sampler.

``crossing``: stationary chains on a two-mode mixture. Every kinetic energy is
tuned on screen chains for mode changes per gradient and read on fresh chains,
next to the bound ``2 * eps * pi_0(x) * rho(0)`` of ``kinetic``.

``local``: stationary chains on unimodal targets with exact moments. Every
kinetic energy is tuned for, and read on, the error of two moments at a fixed
number of gradients.
"""

from dataclasses import asdict

import jax
import jax.numpy as jnp
import numpy as np
from bnn.target.synthetic import SyntheticTarget, Task
from runner import CellSpec, Design, ResourceHint

from . import kinetic, targets
from .frontier import seed

QUESTION = (
    "Can a normalized separable kinetic energy other than the Gaussian make "
    "exact HMC cross modes or estimate moments with fewer gradients?"
)
ROLES = ("starts", "screen", "chain", "bootstrap")
WEIGHT = 0.2
# Step sizes in units of D^(-1/4), times the stiffest scale of a local target.
STEPS = (0.4, 0.6, 0.85, 1.2, 1.7, 2.4)
LOCAL = {
    "gaussian": {"condition": 100.0},
    "banana": {"bend": 0.1},
    "funnel": {"funnel": 3.0, "hard": 8},
}
LOCAL_DIMENSION = 32
TARGET_SEED = 630019


def crossing_factors(profile, separation, dimension):
    smoke = profile == "smoke"
    return {
        "profile": profile,
        "separation": separation,
        "dimension": dimension,
        "weight": WEIGHT,
        "families": list(kinetic.FAMILIES)[:2] if smoke else list(kinetic.FAMILIES),
        "steps": list(STEPS[:2] if smoke else STEPS),
        "lengths": [2] if smoke else [2, 4, 8, 16],
        "screen": {"chains": 16 if smoke else 256, "gradients": 256 if smoke else 2048},
        "confirmation": {
            "chains": 32 if smoke else 1024,
            "gradients": 256 if smoke else 4096,
        },
        "bootstrap": 50 if smoke else 1000,
    }


def local_factors(profile, family):
    smoke = profile == "smoke"
    shape = LOCAL[family]
    task = Task(
        family, "screen", 0, TARGET_SEED, 8 if smoke else LOCAL_DIMENSION, **shape
    )
    return {
        "profile": profile,
        "family": family,
        "task": asdict(task),
        "families": list(kinetic.FAMILIES)[:2] if smoke else list(kinetic.FAMILIES),
        "steps": list(STEPS[:2] if smoke else (0.1, 0.2, *STEPS)),
        "lengths": [4] if smoke else [4, 16, 64],
        "screen": {"chains": 16 if smoke else 128, "gradients": 256 if smoke else 4096},
        "confirmation": {
            "chains": 32 if smoke else 512,
            "gradients": 256 if smoke else 8192,
        },
        "bootstrap": 50 if smoke else 1000,
    }


def design():
    cells = []
    for profile in ("smoke", "pilot"):
        smoke = profile == "smoke"
        for separation, dimension in (
            [(2.5, 8)] if smoke else [(a, d) for a in (2.5, 3.5) for d in (8, 32, 128)]
        ):
            cells.append(
                CellSpec(
                    "crossing",
                    crossing_factors(profile, separation, dimension),
                    {
                        r: seed(profile, separation, dimension, f"exact:{r}")
                        for r in ROLES
                    },
                    role="pilot",
                    expected_schema=("metrics", "estimands"),
                    resources=ResourceHint(memory_gb=4, walltime_minutes=90),
                )
            )
        for family in ["banana"] if smoke else list(LOCAL):
            cells.append(
                CellSpec(
                    "local",
                    local_factors(profile, family),
                    {r: seed(profile, family, 0, f"local:{r}") for r in ROLES},
                    role="pilot",
                    expected_schema=("metrics", "estimands"),
                    resources=ResourceHint(memory_gb=4, walltime_minutes=120),
                )
            )
    return Design(QUESTION, ("smoke", "pilot"), ROLES, tuple(cells))


def key_of(seed_, *index):
    return jax.random.key(int(np.random.default_rng([seed_, *index]).integers(2**32)))


def interval(values):
    return [float(v) for v in np.quantile(values, [0.025, 0.975])]


# ----------------------------------------------------------------------------
# Crossing


def measure_crossing(cell):
    spec, seeds = cell.factors, cell.seeds
    a, d, w = spec["separation"], spec["dimension"], spec["weight"]
    density = targets.barrier_density(a, w)
    rng = np.random.default_rng(seeds["bootstrap"])
    families = {}
    with jax.enable_x64():
        task = {
            name: jnp.asarray(value[0])
            for name, value in targets.tasks(
                np.random.default_rng(0), 1, d, a, "aligned", weights=[w]
            ).items()
        }
        for f, name in enumerate(spec["families"]):
            family = kinetic.FAMILIES[name]
            run = jax.jit(
                kinetic.hmc(
                    family,
                    targets.potential,
                    targets.force,
                    targets.statistics,
                    targets.core,
                ),
                static_argnames=("transitions", "burn", "length"),
            )

            def rate(stage, step, length, salt, run=run, f=f):
                chains, gradients = spec[stage]["chains"], spec[stage]["gradients"]
                starts = targets.draws(key_of(seeds["starts"], f, *salt), task, chains)
                _, acceptance, spent, switches = run(
                    key_of(seeds[stage if stage == "screen" else "chain"], f, *salt),
                    starts,
                    jnp.asarray(step),
                    task,
                    transitions=kinetic.transitions_for(gradients, length),
                    burn=0,
                    length=length,
                )
                return np.asarray(switches), np.asarray(spent), float(acceptance.mean())

            screened = []
            for i, length in enumerate(spec["lengths"]):
                for j, factor in enumerate(spec["steps"]):
                    step = factor * d**-0.25
                    switches, spent, acceptance = rate("screen", step, length, [i, j])
                    screened.append((switches.sum() / spent.sum(), step, length))
            _, step, length = max(screened)
            switches, spent, acceptance = rate("confirmation", step, length, [99])
            draws = rng.integers(len(switches), size=(spec["bootstrap"], len(switches)))
            resampled = switches[draws].sum(1) / spent[draws].sum(1)
            bound = 2 * step * density * family["rest"]
            families[name] = {
                "step": float(step),
                "length": length,
                "acceptance": acceptance,
                "rate": float(switches.sum() / spent.sum()),
                "rate_interval": interval(resampled),
                "rate_resampled": resampled.tolist(),
                "bound": float(bound),
                "transmission": float(switches.sum() / spent.sum() / bound),
                "rest_density": family["rest"],
                "screen_best": float(max(screened)[0]),
            }
    reference = np.array(families["gaussian"]["rate_resampled"])
    for entry in families.values():
        ratio = np.array(entry.pop("rate_resampled")) / reference
        entry["over_gaussian"] = entry["rate"] / families["gaussian"]["rate"]
        entry["over_gaussian_interval"] = interval(ratio)
    return {
        "metrics": {
            "separation": a,
            "dimension": d,
            "weight": w,
            "barrier_density": density,
            # Mode changes per gradient at the leapfrog's stability limit of a
            # unit Gaussian mode, eps = 2, which no chain reaches.
            "stability_bound": float(2 * 2 * density / np.sqrt(2 * np.pi)),
            "family_results": families,
        },
        "estimands": ["mode_changes_per_gradient"],
    }


# ----------------------------------------------------------------------------
# Local targets


def local_target(task):
    """Potential and two normalized estimands in aligned coordinates, and a scale."""
    target = SyntheticTarget(task)
    k = task.hard
    centre = target.mean if task.family == "gaussian" else target.location

    def sample(count, seed_):
        return (target.sample(count, seed_) - centre) @ target.rotation

    if task.family == "gaussian":
        variances = jnp.asarray(target.eigenvalues)
        stiff = float(np.sqrt(target.eigenvalues.min()))

        def energy(u, _):
            return 0.5 * jnp.sum(u**2 / variances)

        # Second moments of the stiffest and of the softest coordinate.
        def observe(u, _):
            return jnp.stack(
                [u[0] ** 2 / variances[0], u[k - 1] ** 2 / variances[k - 1]]
            )

        return energy, observe, np.ones(2), sample, stiff
    aligned = target.aligned_potential()

    def energy(u, _):
        return aligned(u)

    if task.family == "banana":
        exact = np.array([9.0, 1 + 162 * task.bend**2])

        def observe(u, _):
            return jnp.stack([u[0] ** 2, u[1] ** 2]) / exact

        return energy, observe, np.ones(2), sample, 1.0

    # Funnel: mean and second moment of the log-variance coordinate.
    def observe(u, _):
        return jnp.stack([u[0] / task.funnel, (u[0] / task.funnel) ** 2])

    return energy, observe, np.array([0.0, 1.0]), sample, 1.0


def measure_local(cell):
    spec, seeds = cell.factors, cell.seeds
    task = Task(**spec["task"])
    d = task.dimension
    rng = np.random.default_rng(seeds["bootstrap"])
    families = {}
    with jax.enable_x64():
        energy, observe, exact, sample, scale = local_target(task)
        gradient = jax.grad(energy)
        for f, name in enumerate(spec["families"]):
            run = jax.jit(
                kinetic.hmc(
                    kinetic.FAMILIES[name],
                    energy,
                    gradient,
                    observe,
                    lambda theta, task, previous: previous,
                ),
                static_argnames=("transitions", "burn", "length"),
            )

            def errors(stage, step, length, salt, run=run, f=f):
                chains, gradients = spec[stage]["chains"], spec[stage]["gradients"]
                starts = jnp.asarray(
                    sample(
                        chains,
                        int(
                            np.random.default_rng([seeds["starts"], f, *salt]).integers(
                                2**32
                            )
                        ),
                    )
                )
                means, acceptance, _, _ = run(
                    key_of(seeds[stage if stage == "screen" else "chain"], f, *salt),
                    starts,
                    jnp.asarray(step),
                    None,
                    transitions=kinetic.transitions_for(gradients, length),
                    burn=0,
                    length=length,
                )
                return np.asarray(means) - exact, float(acceptance.mean())

            screened = []
            for i, length in enumerate(spec["lengths"]):
                for j, factor in enumerate(spec["steps"]):
                    step = factor * scale * d**-0.25
                    error, _ = errors("screen", step, length, [i, j])
                    mse = np.mean(
                        np.nan_to_num(error, nan=1e6, posinf=1e6, neginf=-1e6) ** 2,
                        axis=0,
                    )
                    screened.append((float(mse.sum()), step, length))
            _, step, length = min(screened)
            error, acceptance = errors("confirmation", step, length, [99])
            error = np.nan_to_num(error, nan=1e6, posinf=1e6, neginf=-1e6)
            draws = rng.integers(len(error), size=(spec["bootstrap"], len(error)))
            families[name] = {
                "step": float(step),
                "length": length,
                "acceptance": acceptance,
                "mse": [float(v) for v in np.mean(error**2, axis=0)],
                "mse_resampled": np.mean(error[draws] ** 2, axis=1).tolist(),
            }
    reference = np.array(families["gaussian"]["mse_resampled"])
    for entry in families.values():
        resampled = np.array(entry.pop("mse_resampled"))
        # Gradients the Gaussian needs for this error, per gradient spent.
        entry["over_gaussian"] = [
            float(v)
            for v in np.array(families["gaussian"]["mse"]) / np.array(entry["mse"])
        ]
        entry["over_gaussian_interval"] = [
            interval(reference[:, e] / resampled[:, e]) for e in range(2)
        ]
    return {
        "metrics": {
            "family": spec["family"],
            "dimension": d,
            "task": spec["task"],
            "gradients": spec["confirmation"]["gradients"],
            "family_results": families,
        },
        "estimands": ["moment_mse_ratio"],
    }


def measure(cell):
    return {"crossing": measure_crossing, "local": measure_local}[cell.stage](cell)
