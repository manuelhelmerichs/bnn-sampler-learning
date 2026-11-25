"""Whole-force learning across network widths: nothing in the force stays explicit.

Theory checks and the geometric audit need no inputs. Force fits read the HMC
populations that ``prepare`` acquires into the store as Zarr trajectories, one
independent acquisition per role; fitting never samples populations.
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
from runner import CellSpec, Design, ResourceHint, cell_folder, relative

QUESTION = (
    "Can one permutation-equivariant force model learn the likelihood force across "
    "network widths?"
)
PROFILES = ("smoke", "pilot", "production")
FIT_SEEDS = (6101, 6102, 6103)
# Parameters-only and forward-feature attention, each with and without force
# normalization, and the feature-assisted mean-message mixer.
CONFIGURATIONS = {
    "parameters": dict(features=False, mixer="attention", normalize_force=False),
    "parameters_normalized": dict(features=False, mixer="attention"),
    "features": dict(features=True, mixer="attention", normalize_force=False),
    "features_normalized": dict(features=True, mixer="attention"),
    "mean_mixer_normalized": dict(features=True, mixer="mean"),
}
STUDENT = dict(channels=32, heads=4, blocks=3, steps=2400, learning_rate=1e-3)


def design():
    cells = []
    for profile in PROFILES:
        cells.append(
            CellSpec("theory", {"profile": profile}, {}, expected_schema=("metrics",))
        )
        cells.append(
            CellSpec(
                "geometric_audit",
                {"profile": profile, "draws": 1000 if profile == "smoke" else 100000},
                {"audit": 141026},
                expected_schema=("metrics",),
            )
        )
    cells.append(
        CellSpec(
            "force_fit",
            {"profile": "smoke", "configuration": "features"},
            {"fit": 2},
            expected_schema=("metrics",),
        )
    )
    for profile, seeds in (("pilot", FIT_SEEDS[:1]), ("production", FIT_SEEDS)):
        cells.extend(
            CellSpec(
                "force_fit",
                {"profile": profile, "configuration": name},
                {"fit": seed},
                expected_schema=("metrics",),
                resources=ResourceHint(cpu=2, memory_gb=4, walltime_minutes=120),
            )
            for name in CONFIGURATIONS
            for seed in seeds
        )
    return Design(QUESTION, PROFILES, ("audit", "fit"), tuple(cells))


def prepare(profile, store):
    from .width_populations import (
        prepare_populations,
        recipe,
    )

    prepare_populations(recipe(profile), store)


def bind(design, profile, store=None):
    if profile == "smoke":
        return design
    from .width_populations import (
        load_metadata,
        recipe,
    )

    root = Path(store).resolve()
    metadata = load_metadata(recipe(profile), root)
    binding = {
        "populations": {
            "root": relative(root),
            "recipe": metadata["recipe"],
            "source_identity": metadata["source_identity"],
            "checksums": metadata["checksums"],
        }
    }
    return replace(
        design,
        cells=tuple(
            replace(cell, inputs=binding, cell_id="")
            if cell.factors["profile"] == profile and cell.stage == "force_fit"
            else cell
            for cell in design.cells
        ),
    )


def _smoke_populations():
    import jax

    widths = {"train": (3, 4), "validation": (3, 4), "test": (2, 5)}
    keys = iter(jax.random.split(jax.random.key(73), 8))
    x = np.asarray(jax.random.normal(next(keys), (8, 2)))
    y = np.asarray(jax.random.normal(next(keys), (8,)))
    states = {
        role: {
            width: np.asarray(jax.random.normal(next(keys), (4, width, 3)))
            for width in role_widths
        }
        for role, role_widths in widths.items()
    }
    return states, {role: f"independent-{role}" for role in states}, x, y


def measure(cell):
    if cell.stage == "theory":
        from .width_theory import run

        return {"metrics": run()}
    if cell.stage == "geometric_audit":
        from .width_diagnostics import (
            geometric_audit,
        )

        return {
            "metrics": geometric_audit(
                seed=cell.seeds["audit"], draws=cell.factors["draws"]
            )
        }
    from .width_models import ShallowBNN
    from .width_study import (
        FitConfig,
        fit_force,
    )

    if cell.factors["profile"] == "smoke":
        states, identities, x, y = _smoke_populations()
        bnn = ShallowBNN(input_dim=2)
        config = FitConfig(
            steps=3, channels=8, heads=2, blocks=1, seed=cell.seeds["fit"]
        )
    else:
        from .width_populations import (
            load_populations,
        )

        states, identities, x, y = load_populations(cell.inputs["populations"])
        bnn = ShallowBNN(input_dim=x.shape[1])
        options = CONFIGURATIONS[cell.factors["configuration"]]
        config = FitConfig(**STUDENT, **options, seed=cell.seeds["fit"])
    _, _, record = fit_force(
        bnn, states, identities, x, y, config=config, output=cell_folder() / "fit"
    )
    return {"metrics": {"validation": record["validation"], "test": record["test"]}}
