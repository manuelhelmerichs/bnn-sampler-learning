"""Measure finite sampler laws using explicitly prepared tasks and fitted forces."""

from __future__ import annotations

import json
import time
from dataclasses import replace
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from bnn.store import trajectories
from runner import (
    CellSpec,
    Design,
    ResourceHint,
    cell_folder,
    derive_seed,
    fingerprint,
    relative,
    repository_root,
)
from runner import reduce_protocol as reduce_base

from . import learned_sampler_inputs as inputs
from .contracts import SamplerContract
from .core import force, forward
from .summary import (
    aggregate_rows,
    validate_panel,
)
from .two_sample import compare_samples

QUESTION = "Which parts of a BNN force computation need to remain explicit for a shared learner to reproduce finite sampler trajectories on new architectures?"
NAME = "learned_force_fidelity"
PROFILES = ("smoke", "pilot", "production")
SEED = 30092026
ARTIFACT = "endpoints.zarr"


def suite_spec():
    cells = [
        CellSpec(
            "learned_sampler",
            {"profile": "smoke", "fixture": True},
            {"parameter_c2st": SEED, "prediction_c2st": SEED + 1},
            expected_schema=("metrics", "estimands"),
            role="pilot",
        )
    ]
    for profile in ("pilot", "production"):
        spec = inputs.recipe(profile)
        groups = {}
        for row_id, factors in inputs.expected_rows(spec).items():
            key = (
                factors["architecture"],
                factors["method"],
                factors["fit_seed"],
                factors["start"],
            )
            groups.setdefault(key, []).append((row_id, factors))
        for rows in groups.values():
            factors = {
                key: value for key, value in rows[0][1].items() if key != "horizon"
            }
            factors.update(
                profile=profile, horizons=[row[1]["horizon"] for row in rows]
            )
            cells.append(
                CellSpec(
                    "learned_sampler",
                    factors,
                    {
                        name: derive_seed(SEED, name, factors)
                        for name in ("parameter_c2st", "prediction_c2st")
                    },
                    expected_schema=("metrics", "estimands"),
                    role="pilot" if profile == "pilot" else "confirmation",
                    resources=ResourceHint(cpu=1, memory_gb=4, walltime_minutes=20),
                )
            )
    return Design(
        QUESTION,
        PROFILES,
        ("parameter_c2st", "prediction_c2st"),
        tuple(cells),
    )


def _metadata(root, profile):
    path = root / "metadata.json"
    metadata = json.loads(path.read_text())
    if metadata.get("profile") != profile or metadata.get("recipe") != inputs.recipe(
        profile
    ):
        raise ValueError("Prepared learned-sampler recipe or profile differs")
    if not isinstance(metadata.get("source_identity"), str):
        raise ValueError("Prepared metadata needs a source identity")
    if metadata.get("expected_rows") != inputs.expected_rows(metadata["recipe"]):
        raise ValueError("Prepared endpoint design differs from the declared recipe")
    files = metadata.get("source_files", {})
    if (
        fingerprint(files, namespace="learned-sampler-source")
        != metadata["source_identity"]
    ):
        raise ValueError("Prepared source identity differs")
    for name, checksum in files.items():
        if trajectories.checksum(inputs.REPOSITORY / name) != checksum:
            raise ValueError(f"Prepared source changed: {name}")
    return metadata, trajectories.checksum(path)


def _input_paths(root, factors, metadata):
    def validate_receipt(path):
        receipt_path = inputs._receipt_path(path)
        if metadata.get("artifact_receipts", {}).get(
            str(receipt_path.relative_to(root))
        ) != trajectories.checksum(receipt_path):
            raise ValueError("Prepared artifact receipt differs from metadata")
        identity = json.loads(receipt_path.read_text())["identity"]
        if identity["source_identity"] != metadata["source_identity"] or identity[
            "recipe_identity"
        ] != fingerprint(metadata["recipe"], namespace="learned-sampler-recipe"):
            raise ValueError("Prepared artifact source or recipe differs")

    task = root / "tasks" / f"{factors['architecture']}.zarr"
    if not inputs.validate_artifact(task):
        raise ValueError(f"Missing prepared task: {task}")
    validate_receipt(task)
    task_hash = trajectories.checksum(task)
    if metadata.get("task_artifacts", {}).get(str(task.relative_to(root))) != task_hash:
        raise ValueError("Prepared task checksum differs from metadata")
    checkpoint_hash = None
    if factors["method"] in inputs.METHODS:
        name = f"{factors['method']}-{factors['fit_seed']}"
        checkpoint = root / "students" / f"{name}.npz"
        if not inputs.validate_artifact(checkpoint):
            raise ValueError(f"Missing fitted student: {checkpoint}")
        validate_receipt(checkpoint)
        checkpoint_hash = trajectories.checksum(checkpoint)
        if (
            metadata.get("training_stats", {}).get(name, {}).get("sha256")
            != checkpoint_hash
        ):
            raise ValueError("Fitted student checksum differs from metadata")
    return task_hash, checkpoint_hash


def bind_inputs(spec, profile, store=None):
    if profile not in PROFILES:
        raise ValueError(f"Unknown learned-sampler profile: {profile}")
    if profile == "smoke":
        return spec
    root = Path(store).resolve()
    metadata, checksum = _metadata(root, profile)
    cells = []
    for cell in spec.cells:
        if cell.factors["profile"] != profile:
            cells.append(cell)
            continue
        task_hash, checkpoint_hash = _input_paths(root, cell.factors, metadata)
        cells.append(
            replace(
                cell,
                cell_id="",
                inputs={
                    "prepared": {
                        "root": relative(root),
                        "metadata_sha256": checksum,
                        "source_identity": metadata["source_identity"],
                        "recipe_identity": fingerprint(metadata["recipe"]),
                        "task_sha256": task_hash,
                        "checkpoint_sha256": checkpoint_hash,
                    }
                },
            )
        )
    return replace(spec, cells=tuple(cells))


def _action_summary(values):
    if values is None:
        return None
    finite = bool(np.isfinite(values).all())
    mean = float(np.mean(values)) if finite else None
    return {
        "paths": len(values),
        "nonfinite_paths": int(np.count_nonzero(~np.isfinite(values))),
        "mean": mean,
        "mean_mcse": float(np.std(values, ddof=1) / np.sqrt(len(values)))
        if finite and len(values) > 1
        else None,
        "empirical_pinsker_proxy": min(1.0, float(np.sqrt(mean / 2)))
        if mean is not None
        else None,
        "scope": "Independent reference-path Monte Carlo estimate; the proxy is not a certified upper confidence bound.",
    }


def _measure_panel(root, spec, factors):
    """Simulate the common reference and candidate paths once for all horizons."""
    jax.config.update("jax_enable_x64", True)
    arch = tuple(map(int, factors["architecture"].split("-")))
    arch_id = [tuple(a) for a in spec["architectures"]].index(arch)
    task = inputs.load_task(root, arch)
    x, y = jnp.asarray(task["x"]), jnp.asarray(task["y"])
    rng = np.random.default_rng(940000 + arch_id)
    prediction_x = rng.normal(size=(20, arch[0]))
    starts = task["anchors"][rng.integers(8, size=2)] + rng.normal(
        scale=0.08, size=(2, len(task["anchors"][0]))
    )
    start = factors["start"]
    q0 = starts[start]
    seed = 950000 + 1000 * arch_id + 100 * start
    noise_shape = (spec["draws"], 1024, len(q0))
    teacher = partial(force, x=x, y=y, architecture=arch)
    reference, ref_bad = inputs.panel_paths(
        q0,
        jax.random.normal(jax.random.key(seed), noise_shape),
        teacher,
        dtype=jnp.float64,
    )
    params, candidate = inputs.candidate_force(
        root, arch, x, y, factors["method"], factors["fit_seed"]
    )
    begin = time.perf_counter()
    paths, bad = inputs.panel_paths(
        q0,
        jax.random.normal(jax.random.key(seed + 1), noise_shape),
        candidate,
        dtype=jnp.float64 if factors["method"] == "teacher" else jnp.float32,
    )
    seconds = time.perf_counter() - begin
    predict = jax.jit(
        jax.vmap(partial(forward, x=jnp.asarray(prediction_x), architecture=arch))
    )
    horizons = factors["horizons"]
    arrays = {
        "reference_parameters": reference[:, horizons].transpose(1, 0, 2),
        "candidate_parameters": paths[:, horizons].transpose(1, 0, 2),
        "reference_failures": ref_bad[:, horizons].T,
        "failures": bad[:, horizons].T,
    }
    arrays["reference_predictions"] = np.stack(
        [
            np.asarray(predict(reference[:, h])).reshape(spec["draws"], -1)
            for h in horizons
        ]
    )
    arrays["candidate_predictions"] = np.stack(
        [np.asarray(predict(paths[:, h])).reshape(spec["draws"], -1) for h in horizons]
    )
    if factors["method"] not in ("teacher", "zero"):
        action = inputs.force_action(
            reference[: spec["force_paths_per_start"], :-1],
            arch,
            x,
            y,
            params,
            odd=factors["method"] == "odd_generic",
        )
        arrays["force_path_action"] = np.stack([action[:, h - 1] for h in horizons])
    return arrays, {
        "evaluation_seed": seed,
        "candidate_seed": seed + 1,
        "timing_seconds_including_compile": seconds,
    }


def execute(cell):
    if cell.stage != "learned_sampler":
        raise ValueError("Computation fidelity requires a learned_sampler cell")
    if cell.factors["profile"] == "smoke":
        return {
            "metrics": {"accepted": True, "fixture": True, "factors": cell.factors},
            "estimands": ["learned_sampler_finite_path_fidelity"],
        }
    prepared = cell.inputs.get("prepared")
    if prepared is None:
        raise ValueError(
            "Learned-sampler measurement requires bound prepared artifacts"
        )
    if cell.seeds != {
        name: derive_seed(SEED, name, cell.factors)
        for name in ("parameter_c2st", "prediction_c2st")
    }:
        raise ValueError("Manifest diagnostic seeds differ from the declared protocol")
    root = cell.path("prepared")
    metadata, checksum = _metadata(root, cell.factors["profile"])
    task_hash, checkpoint_hash = _input_paths(root, cell.factors, metadata)
    if (
        checksum != prepared["metadata_sha256"]
        or metadata["source_identity"] != prepared["source_identity"]
        or fingerprint(metadata["recipe"]) != prepared["recipe_identity"]
        or task_hash != prepared["task_sha256"]
        or checkpoint_hash != prepared["checkpoint_sha256"]
    ):
        raise ValueError("Prepared inputs changed after planning")
    arrays, costs = _measure_panel(root, metadata["recipe"], cell.factors)
    path = cell_folder() / ARTIFACT
    trajectories.save(path, **arrays)
    digest = trajectories.checksum(path)
    rows = []
    for index, horizon in enumerate(cell.factors["horizons"]):
        factors = {
            k: v for k, v in cell.factors.items() if k not in {"profile", "horizons"}
        }
        row_id = f"{factors['architecture']}-{factors['start']}-{factors['method']}-{factors['fit_seed']}-{horizon}"
        factors = {**factors, "horizon": horizon}
        if metadata["expected_rows"].get(row_id) != factors:
            raise ValueError("Cell factors differ from the declared endpoint recipe")
        factors.update(profile=cell.factors["profile"], row_id=row_id)
        candidate_bad = arrays["failures"][index].copy()
        reference_bad = arrays["reference_failures"][index].copy()
        for kind in ("parameters", "predictions"):
            candidate_bad |= ~np.isfinite(arrays[f"candidate_{kind}"][index]).all(
                axis=1
            )
            reference_bad |= ~np.isfinite(arrays[f"reference_{kind}"][index]).all(
                axis=1
            )
        method = factors["method"]
        row = {
            "accepted": True,
            "fixture": False,
            "factors": factors,
            "source": {
                "metadata_sha256": checksum,
                "arrays_sha256": digest,
                "source_identity": metadata["source_identity"],
                "recipe_identity": fingerprint(metadata["recipe"]),
                "task_sha256": task_hash,
                "checkpoint_sha256": checkpoint_hash,
            },
            "training_stats": metadata.get("training_stats", {}).get(
                f"{method}-{factors['fit_seed']}"
            ),
            "force_path_action": _action_summary(
                arrays.get("force_path_action", [None] * len(cell.factors["horizons"]))[
                    index
                ]
            ),
            "costs": costs,
            "sampler_contract": SamplerContract(
                learned=("force",)
                if method not in {"teacher", "exact", "zero"}
                else (),
                target_access="gradient" if method == "exact" else "none",
                proposal_accounting="implicit",
                correction="none",
                validity="finite_teacher",
            ).to_dict(),
        }
        for kind, stream in (
            ("parameters", "parameter_c2st"),
            ("predictions", "prediction_c2st"),
        ):
            row[kind] = compare_samples(
                arrays[f"reference_{kind}"][index],
                arrays[f"candidate_{kind}"][index],
                seed=derive_seed(SEED, stream, factors),
                failures=candidate_bad,
                reference_failures=reference_bad,
            )
        rows.append(row)
    return {
        "metrics": {
            "accepted": True,
            "fixture": False,
            "rows": rows,
            "artifact": {"path": ARTIFACT, "sha256": digest},
            "costs": costs,
        },
        "estimands": [
            "learned_sampler_finite_path_fidelity",
            "learned_sampler_force_path_action",
        ],
    }


def reduce_protocol(manifest, results):
    summary = reduce_base(manifest, results)
    metadata = None
    if manifest.profile == "smoke":
        full = len(manifest.cells) == 1
    else:
        bound = [c.inputs["prepared"] for c in manifest.cells]
        checksums = {item["metadata_sha256"] for item in bound}
        metadata, checksum = _metadata(
            repository_root() / bound[0]["root"], manifest.profile
        )
        observed = [
            row["factors"]["row_id"]
            for result in results
            for row in result["metrics"]["rows"]
        ]
        full = (
            len(checksums) == 1
            and checksum in checksums
            and len(observed) == len(set(observed))
            and set(observed) == set(metadata["expected_rows"])
        )
    accepted = all(row["metrics"]["accepted"] for row in results)
    summary.update(
        accepted=accepted,
        full_protocol=full,
        publishable=manifest.profile == "production" and full and accepted,
        qualification_policy="All attempted draws remain in failure denominators. Distribution scores are suppressed for any population containing failures. Near-chance C2ST does not establish equality or posterior convergence.",
    )
    if metadata is not None:
        summary["preparation"] = {
            name: metadata.get(name)
            for name in (
                "recipe",
                "source_identity",
                "source_files",
                "findings",
                "training_stats",
                "source_reference",
                "expected_rows",
                "force_action_law",
                "stage_receipts",
                "artifact_receipts",
                "environment",
            )
        }
    rows = [row for result in results for row in result["metrics"].get("rows", [])]
    if manifest.profile == "production" and full:
        validate_panel(
            summary,
            [{"cell_id": c.cell_id, "factors": c.factors} for c in manifest.cells],
            summary["cell_metrics"],
            [c.cell_id for c in manifest.cells],
        )
    summary["summary_rows"] = aggregate_rows(rows)
    summary["uncertainty_unit"] = (
        "Starts averaged within each fit; mean and range over fitted students under shared evaluation starts and noise. These are not confidence intervals."
    )
    return summary
