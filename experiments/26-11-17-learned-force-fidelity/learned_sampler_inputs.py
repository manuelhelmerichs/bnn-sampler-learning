"""Prepare synthetic teacher states and fitted forces for sampler measurements."""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from bnn.store import trajectories
from runner import atomic_json, fingerprint, repository_root

from .core import (
    attention_force,
    force,
    initialize,
    kinetic_trajectory,
    parameter_count,
    potential,
)
from .learned_sampler_training import (
    fit_primary,
    fit_queries,
    generic_queries,
    load_student,
    make_queries,
    query_mse,
    save_student,
)
from .student import (
    bilinear_operation,
    local_operation,
    zero_operation,
)

ARCHITECTURES = ((5, 5, 1), (6, 6, 6, 1), (7, 7, 7, 7, 1), (5, 4, 3, 1), (6, 5, 5, 1))
SOURCE_ARCHITECTURES = ARCHITECTURES[:3]
METHODS = ("primary", "boundary", "composed", "generic", "odd_generic")
CONTROLS = ("teacher", "zero", "exact")
REPOSITORY = repository_root()
_STAGE = None


def label(architecture):
    return "-".join(map(str, architecture))


def recipe(profile):
    production = profile == "production"
    return {
        "profile": profile,
        "data_seed": 9302026,
        "data_signal": "sin(x0)+0.5*x1*x2+0.3*x3; iid N(0,1) x; N(0,.15²) noise",
        "scope": "independent conceptual replication; original data/anchors unavailable",
        "observations": 64,
        "sigma": 0.5,
        "tau": 1.0,
        "gate_width": 1e-4,
        "step_size": 0.01,
        "friction": 0.15,
        "anchor_steps": 1000,
        "anchor_count": 8,
        "anchor_adam_rate": 0.01,
        "anchor_initial_sd": 0.2,
        "start_perturbation_sd": 0.08,
        "training_paths": 64 if production else 16,
        "training_states_per_path": 64,
        "validation_states": 512,
        "early_times_per_path": 32,
        "late_times_per_path": 32,
        "training_horizon": 1024,
        "updates": 12000 if production else 2000,
        "primary_batch_size": 32,
        "query_batch_size": 128,
        "fit_seeds": [327001, 327002, 327003] if production else [327001],
        "evaluation_starts": 2,
        "draws": 256 if production else 64,
        "horizons": [128, 512, 1024],
        "prediction_inputs": 20,
        "force_paths_per_start": 32 if production else 8,
        "label_implementation": "direct float64 arithmetic; literal attention checked independently",
        "student_dtype": "float32",
        "reference_dtype": "float64",
        "failure_rule": "nonfinite or max|q|>=1e12 or max|v|>=1e14 at any step",
        "methods": list(METHODS),
        "architectures": [list(a) for a in ARCHITECTURES],
        "source_architectures": [list(a) for a in SOURCE_ARCHITECTURES],
        "checkpoint_selection": "primary: early source validation; other fits: final",
        "augmentation": "observation subsampling; neuron permutations omitted because local learner is exactly permutation equivariant",
    }


def source_files():
    paths = [
        Path(__file__),
        Path(__file__).with_name("learned_sampler_training.py"),
        Path(__file__).with_name("core.py"),
        Path(__file__).with_name("student.py"),
    ]
    return {
        str(path.resolve().relative_to(REPOSITORY)): trajectories.checksum(path)
        for path in paths
    }


def _receipt_path(path):
    return path.with_suffix(path.suffix + ".receipt.json")


def begin_stage(root, stage, spec, methods):
    """Freeze the actual source files before execution; old unreceipted work fails."""
    global _STAGE
    files = source_files()
    identity = fingerprint(files, namespace="learned-sampler-source")
    snapshot = root / "sources" / identity
    for relative, checksum in files.items():
        destination = snapshot / relative
        content = (REPOSITORY / relative).read_bytes()
        if hashlib.sha256(content).hexdigest() != checksum:
            raise ValueError("Source changed while taking the preparation snapshot")
        if destination.exists():
            if trajectories.checksum(destination) != checksum:
                raise ValueError("Immutable preparation source snapshot changed")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
    _STAGE = {
        "stage": stage,
        "recipe_identity": fingerprint(spec, namespace="learned-sampler-recipe"),
        "source_identity": identity,
        "source_files": files,
        "methods": list(methods),
    }
    for prior_path in (root / "receipts").glob("*.json"):
        prior = json.loads(prior_path.read_text())
        if any(
            prior.get(key) != _STAGE[key]
            for key in ("recipe_identity", "source_identity")
        ):
            raise ValueError(
                "Preparation source or recipe changed; choose a fresh root"
            )
    stage_key = fingerprint(_STAGE, namespace="learned-sampler-stage")
    path = root / "receipts" / f"{stage}-{stage_key}.json"
    receipt = {
        **_STAGE,
        "started_at": datetime.now(UTC).isoformat(),
        "state": "running",
    }
    atomic_json(path, receipt)
    return path, receipt


def finish_stage(path, receipt, root):
    if source_files() != receipt["source_files"]:
        raise ValueError(
            "Preparation source changed during execution; discard this stage"
        )
    outputs = {
        str(file.relative_to(root)): trajectories.checksum(file)
        for file in sorted(root.rglob("*.receipt.json"))
    }
    atomic_json(
        path,
        {
            **receipt,
            "state": "complete",
            "completed_at": datetime.now(UTC).isoformat(),
            "artifact_receipts": outputs,
        },
    )


def _artifact_identity(kind, factors, inputs=None):
    if _STAGE is None:
        raise ValueError("Preparation artifacts require an active source receipt")
    return {
        "kind": kind,
        "recipe_identity": _STAGE["recipe_identity"],
        "source_identity": _STAGE["source_identity"],
        "source_files": _STAGE["source_files"],
        "factors": factors,
        "inputs": inputs or {},
    }


def validate_artifact(path, expected=None):
    receipt_path = _receipt_path(path)
    if not path.exists() and not receipt_path.exists():
        return False
    if not path.exists() or not receipt_path.exists():
        raise ValueError(f"Missing artifact or at-execution receipt: {path}")
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("sha256") != trajectories.checksum(path):
        raise ValueError(f"Prepared artifact checksum differs: {path}")
    identity = receipt.get("identity")
    if not isinstance(identity, dict) or (
        expected is not None and identity != expected
    ):
        raise ValueError(f"Prepared artifact source/input identity differs: {path}")
    for relative, checksum in identity["source_files"].items():
        if trajectories.checksum(REPOSITORY / relative) != checksum:
            raise ValueError(f"Prepared artifact source changed: {relative}")
    if _STAGE is not None and identity["recipe_identity"] != _STAGE["recipe_identity"]:
        raise ValueError(f"Prepared artifact recipe changed: {path}")
    return True


def record_artifact(path, identity):
    atomic_json(
        _receipt_path(path),
        {"identity": identity, "sha256": trajectories.checksum(path)},
    )


def _array_identity(array):
    value = np.ascontiguousarray(array)
    return fingerprint(
        {
            "shape": value.shape,
            "dtype": str(value.dtype),
            "bytes": hashlib.sha256(value.tobytes()).hexdigest(),
        }
    )


def method_architectures(spec, method):
    architectures = [tuple(a) for a in spec["architectures"]]
    sources = [tuple(a) for a in spec["source_architectures"]]
    if method in CONTROLS:
        return architectures
    if method == "primary":
        return architectures[:-1]
    if method in ("boundary", "composed"):
        return [a for a in architectures if a not in sources[:2]]
    if method in ("generic", "odd_generic"):
        return [a for a in architectures[:-1] if a not in sources[:2]]
    raise ValueError(f"Unknown endpoint method: {method}")


def panel_factors(spec, architecture, method, fit_seed, start, horizon):
    source_family = (
        [label(tuple(a)) for a in spec["source_architectures"]]
        if method == "primary"
        else [label(tuple(spec["source_architectures"][1]))]
        if method in ("boundary", "composed")
        else []
    )
    return {
        "method": method,
        "architecture": label(architecture),
        "fit_seed": fit_seed,
        "start": start,
        "horizon": horizon,
        "split": "control"
        if method in CONTROLS
        else "unseen_all_bnn"
        if not source_family
        else "source"
        if label(architecture) in source_family
        else "unseen",
        "source_family": source_family,
    }


def expected_rows(spec):
    rows = {}
    for method in (*CONTROLS, *spec["methods"]):
        for architecture in method_architectures(spec, method):
            for seed in [0] if method in CONTROLS else spec["fit_seeds"]:
                for start in range(spec["evaluation_starts"]):
                    for horizon in spec["horizons"]:
                        name = (
                            f"{label(architecture)}-{start}-{method}-{seed}-{horizon}"
                        )
                        rows[name] = panel_factors(
                            spec, architecture, method, seed, start, horizon
                        )
    return rows


def validate_complete_rows(rows, spec):
    expected = expected_rows(spec)
    if len(rows) != len(expected) or {row["id"] for row in rows} != set(expected):
        raise ValueError(
            f"Incomplete endpoint protocol: expected {len(expected)} distinct rows"
        )
    if any(row["factors"] != expected[row["id"]] for row in rows):
        raise ValueError("Endpoint factors differ from the recipe-defined protocol")


def finalize(root, spec, profile):
    rows = [
        {"id": name, "factors": factors}
        for name, factors in expected_rows(spec).items()
    ]
    training = json.loads((root / "training.json").read_text())
    expected_students = {
        f"{method}-{seed}" for method in spec["methods"] for seed in spec["fit_seeds"]
    }
    if set(training) != expected_students:
        raise ValueError("Finalization requires every declared independent student fit")
    for name, record in training.items():
        path = root / "students" / f"{name}.npz"
        if not validate_artifact(path) or record["sha256"] != trajectories.checksum(
            path
        ):
            raise ValueError("Finalization found an invalid student checkpoint")
    checks_path = root / "numerical_checks.json"
    if not validate_artifact(checks_path):
        raise ValueError("Finalization requires receipted numerical checks")
    findings = {"numerical_checks": json.loads(checks_path.read_text())}
    stage_receipts = {}
    for path in sorted((root / "receipts").glob("*.json")):
        receipt = json.loads(path.read_text())
        if receipt["stage"] == "finalize":
            continue
        if receipt["state"] != "complete":
            raise ValueError("Cannot finalize an incomplete preparation stage")
        stage_receipts[str(path.relative_to(root))] = trajectories.checksum(path)
    files = source_files()
    atomic_json(
        root / "metadata.json",
        {
            "profile": profile,
            "recipe": spec,
            "source_identity": fingerprint(files, namespace="learned-sampler-source"),
            "source_files": files,
            "source_reference": {
                "title": "Representing and Learning Bayesian Neural Network Samplers",
                "status": "User-supplied anonymous manuscript under review at ICLR 2027",
                "sha256": "8eaba074aadb67c784de7e364f1ce51e3b671e0bdaa2b92ef82449603fdf36d1",
                "scope": "SHA256 identifies the supplied PDF bytes; the unpublished manuscript is not redistributed.",
            },
            "rows": rows,
            "expected_rows": expected_rows(spec),
            "training_stats": training,
            "findings": findings,
            "environment": {
                "jax": jax.__version__,
                "python": platform.python_version(),
                "devices": str(jax.devices()),
            },
            "stage_receipts": stage_receipts,
            "artifact_receipts": {
                str(path.relative_to(root)): trajectories.checksum(path)
                for path in sorted(root.rglob("*.receipt.json"))
            },
            "task_artifacts": {
                str(path.relative_to(root)): trajectories.checksum(path)
                for path in sorted((root / "tasks").glob("*.zarr"))
            },
            "force_action_law": "Teacher ramp-force law versus the learned drift evaluated in float64 with promoted fitted coefficients; independent paths conditional on fixed starts. This diagnostic does not bound the actual float32 arithmetic law.",
        },
    )


def dataset(input_dimension):
    rng = np.random.default_rng(9302026 + input_dimension)
    x = rng.normal(size=(64, input_dimension))
    y = (
        np.sin(x[:, 0])
        + 0.5 * x[:, 1] * x[:, 2]
        + 0.3 * x[:, 3]
        + rng.normal(scale=0.15, size=64)
    )
    return x, y


def sample_paths(q0, noise, gradient, *, h=0.01):
    """Independent paths; fixed start per law and no hidden stabilization."""
    return jax.vmap(
        lambda z: kinetic_trajectory(q0, jnp.zeros_like(q0), z, gradient, h=h)
    )(noise)


def prepare_task(root, architecture, spec):
    path = root / "tasks" / f"{label(architecture)}.zarr"
    identity = _artifact_identity("teacher_task", {"architecture": list(architecture)})
    if validate_artifact(path, identity):
        return
    x, y = map(jnp.asarray, dataset(architecture[0]))
    seed = 100000 + sum((i + 1) * n for i, n in enumerate(architecture))
    optimizer = optax.adam(0.01)

    def one_anchor(key):
        q = initialize(key, architecture, scale=0.2)
        state = optimizer.init(q)

        def step(carry, _):
            q, state = carry
            g = jax.grad(potential)(q, x, y, architecture)
            update, state = optimizer.update(g, state, q)
            return (optax.apply_updates(q, update), state), None

        return jax.lax.scan(step, (q, state), None, length=1000)[0][0]

    anchors = jax.jit(jax.vmap(one_anchor))(jax.random.split(jax.random.key(seed), 8))

    def gradient(q):
        return force(q, x, y, architecture)

    def states(split_seed, paths, per_half):
        rng = np.random.default_rng(split_seed)
        starts = np.asarray(anchors)[rng.integers(8, size=paths)] + rng.normal(
            scale=0.08, size=(paths, parameter_count(architecture))
        )
        noise = jax.random.normal(
            jax.random.key(split_seed + 1), (paths, 1024, parameter_count(architecture))
        )
        qs, _ = jax.jit(
            jax.vmap(lambda q, z: kinetic_trajectory(q, jnp.zeros_like(q), z, gradient))
        )(jnp.asarray(starts), noise)
        q = np.asarray(qs)
        # Sort globally by time so source-only early-state validation matches Appendix O.
        indices = [
            (t, i)
            for i in range(paths)
            for t in np.concatenate(
                (
                    rng.choice(128, per_half, replace=False),
                    rng.choice(np.arange(128, 1024), per_half, replace=False),
                )
            )
        ]
        indices.sort()
        return np.asarray([q[i, t] for t, i in indices]), np.asarray(indices)

    train, train_index = states(seed + 1000, spec["training_paths"], 32)
    val, val_index = states(seed + 2000, 8, 32)
    test, test_index = states(seed + 3000, 8, 32)
    arrays = {
        "x": np.asarray(x),
        "y": np.asarray(y),
        "architecture": np.asarray(architecture),
        "anchors": np.asarray(anchors),
        "train_q": train,
        "validation_q": val,
        "test_q": test,
        "train_path_time": train_index[:, ::-1],
        "validation_path_time": val_index[:, ::-1],
        "test_path_time": test_index[:, ::-1],
    }
    if not all(np.isfinite(a).all() for a in arrays.values()):
        raise ValueError(f"Nonfinite teacher states for {architecture}")
    trajectories.save(path, **arrays)
    record_artifact(path, identity)
    print(f"prepared {label(architecture)}: {len(train)} training states", flush=True)


def load_task(root, arch):
    path = root / "tasks" / f"{label(arch)}.zarr"
    if not validate_artifact(path):
        raise ValueError(f"Missing prepared teacher task: {path}")
    archive = trajectories.load(path)
    return {key: archive[key] for key in archive}


def fit(root, spec, methods):
    tasks = [load_task(root, arch) for arch in SOURCE_ARCHITECTURES]
    source_queries = make_queries(
        tasks[1], seed=390001, max_arity=7, gate_width=spec["gate_width"]
    )
    test_queries = make_queries(
        tasks[1],
        split="test",
        seed=390002,
        count=2048,
        max_arity=7,
        gate_width=spec["gate_width"],
    )
    generic = generic_queries(max_arity=7)
    records_path = root / "training.json"
    records = json.loads(records_path.read_text()) if records_path.exists() else {}
    for method in methods:
        for seed in spec["fit_seeds"]:
            name = f"{method}-{seed}"
            path = root / "students" / f"{name}.npz"
            identity = _artifact_identity(
                "student",
                {"method": method, "fit_seed": seed},
                {
                    str(
                        (root / "tasks" / f"{label(arch)}.zarr").relative_to(root)
                    ): trajectories.checksum(root / "tasks" / f"{label(arch)}.zarr")
                    for arch in SOURCE_ARCHITECTURES
                },
            )
            if validate_artifact(path, identity):
                if name not in records:
                    raise ValueError("Student checkpoint lacks its training record")
                if records[name]["sha256"] != trajectories.checksum(path):
                    raise ValueError("Checkpoint digest changed")
                if records[name].get("preparation_identity") != fingerprint(identity):
                    raise ValueError(
                        "Student training record source/input identity differs"
                    )
                continue
            if name in records:
                raise ValueError(
                    "Training record points to a missing student checkpoint"
                )
            if method == "primary":
                params, record = fit_primary(
                    tasks,
                    seed=seed,
                    updates=spec["updates"],
                    batch_size=spec["primary_batch_size"],
                    sigma=spec["sigma"],
                    tau=spec["tau"],
                    gate_width=spec["gate_width"],
                )
            else:
                queries = (
                    source_queries[-1:]
                    if method == "boundary"
                    else source_queries
                    if method == "composed"
                    else generic
                )
                params, record = fit_queries(
                    queries,
                    seed=seed,
                    updates=spec["updates"],
                    batch_size=spec["query_batch_size"],
                    odd=method == "odd_generic",
                    objective="bounded",
                )
            save_student(path, params)
            record_artifact(path, identity)
            record.update(
                {
                    "sha256": trajectories.checksum(path),
                    "preparation_identity": fingerprint(identity),
                    "heldout_composed_mse": query_mse(
                        params, test_queries[:-1], odd=method == "odd_generic"
                    ),
                    "heldout_boundary_mse": query_mse(
                        params, test_queries[-1:], odd=method == "odd_generic"
                    ),
                    "generic_test_mse": query_mse(
                        params,
                        generic_queries(seed=390003, count=2048, max_arity=7),
                        odd=method == "odd_generic",
                    ),
                }
            )
            records[name] = record
            atomic_json(records_path, records)
    return records


def panel_paths(q0, noises, gradient, *, dtype):
    q0, noises = jnp.asarray(q0, dtype), jnp.asarray(noises, dtype)
    qs, vs = jax.jit(lambda start, z: sample_paths(start, z, gradient))(q0, noises)
    # Preserve every failure, even if a later finite state could hide it.
    bad = (
        ~np.isfinite(qs).all(axis=-1)
        | ~np.isfinite(vs).all(axis=-1)
        | (np.abs(qs).max(axis=-1) >= 1e12)
        | (np.abs(vs).max(axis=-1) >= 1e14)
    )
    return np.asarray(qs), np.maximum.accumulate(bad, axis=1)


def numerical_checks(root):
    result = []
    for arch in ARCHITECTURES:
        task = load_task(root, arch)
        x, y = jnp.asarray(task["x"]), jnp.asarray(task["y"])
        q = jnp.asarray(task["anchors"][0])

        direct = partial(force, x=x, y=y, architecture=arch)
        attention = partial(attention_force, x=x, y=y, architecture=arch)

        noises = jax.random.normal(
            jax.random.key(222 + parameter_count(arch)), (2, 128, len(q))
        )
        dp, _ = panel_paths(q, noises, direct, dtype=jnp.float64)
        ap, _ = panel_paths(q, noises, attention, dtype=jnp.float64)
        result.append(
            {
                "architecture": label(arch),
                "same_state_force_max_error": float(
                    jnp.max(jnp.abs(direct(q) - attention(q)))
                ),
                "128_step_path_max_error": float(np.max(np.abs(dp - ap))),
            }
        )
    return result


def candidate_force(root, architecture, x, y, method, seed):
    params = None
    dtype = jnp.float64 if method == "teacher" else jnp.float32
    operation = None
    if method == "zero":
        operation = zero_operation
    elif method == "exact":
        operation = partial(bilinear_operation, 1.0)
    elif method != "teacher":
        checkpoint = root / "students" / f"{method}-{seed}.npz"
        if not validate_artifact(checkpoint):
            raise ValueError(f"Missing fitted student: {checkpoint}")
        params = load_student(checkpoint)
        operation = partial(local_operation, params, odd=method == "odd_generic")
    return params, partial(
        force,
        x=x.astype(dtype),
        y=y.astype(dtype),
        architecture=architecture,
        local_operation=operation,
    )


def force_action(states, architecture, x, y, params, *, odd):
    """Path-KL integrand under the ramp teacher law, using promoted weights."""
    p64 = (
        jax.tree.map(lambda a: a.astype(jnp.float64), params)
        if params is not None
        else None
    )
    operation = partial(local_operation, p64, odd=odd) if p64 is not None else None
    approximate = partial(
        force, x=x, y=y, architecture=architecture, local_operation=operation
    )
    reference = partial(force, x=x, y=y, architecture=architecture)

    def accumulated(path):
        def step(total, q):
            difference = reference(q) - approximate(q)
            total = total + (0.01**2 / (4 * 0.15)) * jnp.sum(difference**2)
            return total, total

        return jax.lax.scan(step, jnp.asarray(0.0, jnp.float64), path)[1]

    return np.asarray(jax.jit(jax.vmap(accumulated))(jnp.asarray(states)))


def prepare(profile, root):
    """Prepare teacher tasks and learner fits, then finalize the endpoint rows."""
    if profile not in ("pilot", "production"):
        raise ValueError(f"Learned-sampler inputs have no {profile} preparation")
    methods = list(METHODS)
    jax.config.update("jax_enable_x64", True)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for directory in ("tasks", "students"):
        (root / directory).mkdir(exist_ok=True)
    spec = recipe(profile)
    spec_path = root / "recipe.json"
    if spec_path.exists() and json.loads(spec_path.read_text()) != spec:
        raise ValueError("Existing preparation uses another recipe; choose a new root")
    atomic_json(spec_path, spec)
    stage_path, receipt = begin_stage(root, "tasks", spec, methods)
    for arch in ARCHITECTURES:
        prepare_task(root, arch, spec)
    checks = root / "numerical_checks.json"
    checks_identity = _artifact_identity(
        "numerical_checks",
        {},
        {
            str(p.relative_to(root)): trajectories.checksum(p)
            for p in sorted((root / "tasks").glob("*.zarr"))
        },
    )
    if not validate_artifact(checks, checks_identity):
        atomic_json(checks, numerical_checks(root))
        record_artifact(checks, checks_identity)
    finish_stage(stage_path, receipt, root)
    stage_path, receipt = begin_stage(root, "fit", spec, methods)
    fit(root, spec, methods)
    finish_stage(stage_path, receipt, root)
    stage_path, receipt = begin_stage(root, "finalize", spec, methods)
    finalize(root, spec, profile)
    finish_stage(stage_path, receipt, root)
