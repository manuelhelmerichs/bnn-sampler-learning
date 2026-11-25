"""Prepared HMC populations for the width-transfer force fits.

One synthetic bias-free ReLU regression task. Every role and width has its own
independently seeded, full-data MH-corrected HMC acquisition, stored as sharded
Zarr trajectories: ``<role>.zarr`` holds the chains ``M<width>`` and, for the
training role, the retained warmup states ``warmup_M<width>``. Fitting
populations are derived from these trajectories deterministically.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from bnn.store.trajectories import checksum as trajectory_checksum
from bnn.store.trajectories import load as load_trajectories
from bnn.store.trajectories import save as save_trajectories
from runner import (
    atomic_json,
    checkout,
    fingerprint,
    read_json,
    repository_root,
    runtime_provenance,
)

from .width_diagnostics import chain_health
from .width_models import ShallowBNN

ROLES = ("train", "validation", "test")


def recipe(profile):
    sizes = {
        "pilot": dict(
            train_widths=[5, 10],
            test_widths=[3, 5, 15],
            chains=4,
            warmup=30,
            test_draws=64,
            train_states=128,
            validation_states=32,
        ),
        "production": dict(
            train_widths=[5, 10, 20],
            test_widths=[2, 3, 5, 10, 15, 20, 40, 100],
            chains=8,
            warmup=800,
            test_draws=1200,
            train_states=2048,
            validation_states=256,
        ),
    }
    if profile not in sizes:
        raise ValueError(f"No prepared populations for profile {profile!r}")
    return {
        "profile": profile,
        "dataset_seed": 141026,
        "input_dim": 5,
        "true_width": 5,
        "observations": 64,
        "noise_sd": 1.0,
        "max_leapfrog": 16,
        "target_acceptance": 0.8,
        "perturbation_sd": 0.15,
        # Each role's stream is offset by its width.
        "seeds": {
            "train": 710000,
            "validation": 720000,
            "test": 730000,
            "mixture": 740000,
            "test_selection": 830000,
        },
        **sizes[profile],
    }


def source_identity():
    return fingerprint(
        {
            name: hashlib.sha256(
                Path(__file__).with_name(name).read_bytes()
            ).hexdigest()
            for name in ("width_populations.py", "width_models.py")
        }
    )


def task(spec):
    """The fixed regression task: noncollinear balanced truth, inputs, noisy targets."""
    p, width = spec["input_dim"], spec["true_width"]
    incoming_key, input_key, probe_key, noise_key = jax.random.split(
        jax.random.key(spec["dataset_seed"]), 4
    )
    incoming = jax.random.normal(incoming_key, (width, p), jnp.float64)
    incoming = (
        math.sqrt(p) * incoming / jnp.linalg.norm(incoming, axis=-1, keepdims=True)
    )
    outgoing = math.sqrt(p) * jnp.where(jnp.arange(width) % 2 == 0, 1.0, -1.0)
    truth = jnp.concatenate((incoming, outgoing[:, None]), -1)
    bnn = ShallowBNN(p, spec["noise_sd"])
    x = jax.random.normal(input_key, (spec["observations"], p), jnp.float64)
    probe = jax.random.normal(probe_key, (8, p), jnp.float64)
    noise = jax.random.normal(noise_key, (spec["observations"],), jnp.float64)
    y = bnn.predict(truth[None], x)[0] + bnn.noise_sd * noise
    return bnn, {"truth": truth, "x": x, "y": y, "probe": probe}


def hmc(
    potential,
    gradient,
    start,
    *,
    key,
    warmup,
    draws,
    step_size=0.04,
    max_leapfrog=16,
    target_acceptance=0.8,
):
    """Batched full-data HMC with a random path length of 4 to ``max_leapfrog``.

    The step size adapts toward ``target_acceptance`` during warmup and is then
    frozen. Every fourth warmup state is retained; warmup draws are discarded.
    """

    @jax.jit
    def transition(q, step, key):
        momentum_key, length_key, accept_key = jax.random.split(key, 3)
        momentum = jax.random.normal(momentum_key, q.shape, q.dtype)
        count = jax.random.randint(length_key, (), 4, max_leapfrog + 1)
        axes = tuple(range(1, q.ndim))
        energy = potential(q) + 0.5 * jnp.square(momentum).sum(axes)

        def leapfrog(index, state):
            position, velocity = state
            position = position + step * velocity
            factor = jnp.where(index == count - 1, 0.5, 1.0)
            return position, velocity - factor * step * gradient(position)

        proposal, velocity = jax.lax.fori_loop(
            0, count, leapfrog, (q, momentum - 0.5 * step * gradient(q))
        )
        error = potential(proposal) + 0.5 * jnp.square(velocity).sum(axes) - energy
        probability = jnp.nan_to_num(jnp.exp(jnp.minimum(-error, 0.0)), nan=0.0)
        accepted = jax.random.uniform(accept_key, probability.shape) < probability
        divergent = (jnp.abs(error) > 1000) | ~jnp.isfinite(error)
        q = jnp.where(accepted.reshape(-1, *(1,) * (q.ndim - 1)), proposal, q)
        return q, probability, accepted, divergent

    q, step = jnp.asarray(start), float(step_size)
    samples, warm, accepts, divergences = [], [], [], 0
    for iteration in range(warmup + draws):
        q, probability, accepted, divergent = transition(
            q, step, jax.random.fold_in(key, iteration)
        )
        if iteration < warmup:
            rate = 0.05 / (1 + iteration / 50) ** 0.6
            step = min(
                0.3,
                max(
                    1e-5,
                    step
                    * math.exp(rate * (float(probability.mean()) - target_acceptance)),
                ),
            )
            if iteration % 4 == 0:
                warm.append(np.asarray(q))
        else:
            samples.append(np.asarray(q))
            accepts.append(float(accepted.mean()))
            divergences += int(divergent.sum())
    return {
        "samples": np.stack(samples, axis=1),
        "warmup": np.concatenate(warm),
        "step_size": step,
        "acceptance": float(np.mean(accepts)),
        "divergences": divergences,
    }


def _health(bnn, samples, data):
    """Rank/folded diagnostics of predictions, potential, balance and norm."""
    chains, draws = samples.shape[:2]
    flat = jnp.asarray(samples.reshape(chains * draws, *samples.shape[2:]))
    balance = (jnp.square(flat[..., :-1]).sum(-1) - jnp.square(flat[..., -1])).mean(-1)
    values = jnp.concatenate(
        (
            bnn.predict(flat, data["probe"]),
            bnn.potential(flat, data["x"], data["y"])[:, None],
            balance[:, None],
            jnp.square(flat).mean((1, 2))[:, None],
        ),
        -1,
    )
    names = [f"prediction_{i}" for i in range(len(data["probe"]))]
    return chain_health(
        np.asarray(values).reshape(chains, draws, -1),
        [*names, "potential", "balance", "norm"],
    )


def _acquire(spec, bnn, data, role, width):
    draws = {
        "train": max(64, spec["train_states"] // spec["chains"]),
        "validation": max(64, spec["validation_states"] // spec["chains"]),
        "test": spec["test_draws"],
    }[role]
    key = jax.random.key(spec["seeds"][role] + width)
    start_key, chain_key = jax.random.split(key)
    shape = (spec["chains"], width, bnn.input_dim + 1)
    result = hmc(
        lambda q: bnn.potential(q, data["x"], data["y"]),
        lambda q: bnn.gradient(q, data["x"], data["y"]),
        jax.random.normal(start_key, shape, jnp.float64),
        key=chain_key,
        warmup=spec["warmup"],
        draws=draws,
        step_size=0.04
        if bnn.scaling == "sqrt_width"
        else 0.04 / math.sqrt(max(1, width / 10)),
        max_leapfrog=spec["max_leapfrog"],
        target_acceptance=spec["target_acceptance"],
    )
    record = {
        "width": width,
        "draws": draws,
        "step_size": result["step_size"],
        "acceptance": result["acceptance"],
        "divergences": result["divergences"],
        "health": _health(bnn, result["samples"], data),
    }
    return result, record


def prepare_populations(spec, root):
    """Acquire every role and width once; an existing complete store is reused."""
    root = Path(root)
    if (root / "metadata.json").exists():
        return load_metadata(spec, root)
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"Incomplete preparation exists; choose another store: {root}")
    widths = {
        "train": spec["train_widths"],
        "validation": spec["train_widths"],
        "test": spec["test_widths"],
    }
    records, checksums = {}, {}
    with jax.enable_x64(True):
        bnn, data = task(spec)
        for role in ROLES:
            arrays, records[role] = {}, []
            for width in widths[role]:
                result, record = _acquire(spec, bnn, data, role, width)
                arrays[f"M{width}"] = result["samples"]
                if role == "train":
                    arrays[f"warmup_M{width}"] = result["warmup"]
                records[role].append(record)
                print(
                    f"{spec['profile']} {role} M={width}: "
                    f"acceptance {record['acceptance']:.3f}, "
                    f"R-hat {record['health']['max_rhat']:.4f}",
                    flush=True,
                )
            path = save_trajectories(root / f"{role}.zarr", **arrays)
            checksums[role] = trajectory_checksum(path)
        task_data = {name: np.asarray(value).tolist() for name, value in data.items()}
        runtime = runtime_provenance()
    metadata = {
        "recipe": spec,
        "source_identity": source_identity(),
        "task": task_data,
        "records": records,
        "checksums": checksums,
        "code": checkout(__package__),
        "runtime": runtime,
    }
    atomic_json(root / "metadata.json", metadata)
    return metadata


def load_metadata(spec, root, *, source=None):
    """The record of a complete store that matches ``spec`` and the source."""
    root = Path(root)
    if not (root / "metadata.json").is_file():
        raise FileNotFoundError(f"Missing prepared width populations: {root}")
    metadata = read_json(root / "metadata.json")
    if metadata["recipe"] != spec or metadata["source_identity"] != (
        source or source_identity()
    ):
        raise ValueError("Prepared width populations differ from the recipe/source")
    for role in ROLES:
        if trajectory_checksum(root / f"{role}.zarr") != metadata["checksums"][role]:
            raise ValueError(f"Prepared {role} populations changed")
    return metadata


def load_populations(binding):
    """Fitting populations, role identities and observations of a bound store.

    Training states combine posterior draws with retained warmup states, prior
    draws and perturbed posterior draws; validation and test states are posterior
    draws. Every subset is drawn from its own recorded stream.
    """
    spec, root = binding["recipe"], repository_root() / binding["root"]
    metadata = load_metadata(spec, root, source=binding["source_identity"])
    if metadata["checksums"] != binding["checksums"]:
        raise ValueError("Prepared width populations differ from the planned inputs")
    trajectories = {role: load_trajectories(root / f"{role}.zarr") for role in ROLES}
    seeds, p = spec["seeds"], spec["input_dim"]
    populations = {role: {} for role in ROLES}

    def pool(role, width):
        samples = trajectories[role][f"M{width}"]
        return samples.reshape(-1, *samples.shape[2:]).astype(np.float32)

    for width in spec["train_widths"]:
        rng = np.random.default_rng(seeds["mixture"] + width)
        samples = pool("train", width)
        posterior = samples[rng.permutation(len(samples))[: spec["train_states"]]]
        count = max(16, len(posterior) // 4)
        warmup = trajectories["train"][f"warmup_M{width}"].astype(np.float32)
        populations["train"][width] = np.concatenate(
            (
                posterior,
                warmup[rng.integers(len(warmup), size=count)],
                rng.standard_normal((count, width, p + 1)).astype(np.float32),
                posterior[:count]
                + spec["perturbation_sd"]
                * rng.standard_normal(posterior[:count].shape).astype(np.float32),
            )
        )
        samples = pool("validation", width)
        populations["validation"][width] = samples[
            rng.permutation(len(samples))[: spec["validation_states"]]
        ]
    for width in spec["test_widths"]:
        samples = pool("test", width)
        order = np.random.default_rng(seeds["test_selection"] + width).permutation(
            len(samples)
        )
        populations["test"][width] = samples[order[: spec["validation_states"]]]
    x = np.asarray(metadata["task"]["x"], dtype=np.float32)
    y = np.asarray(metadata["task"]["y"], dtype=np.float32)
    return populations, dict(metadata["checksums"]), x, y


__all__ = [
    "ROLES",
    "hmc",
    "load_metadata",
    "load_populations",
    "prepare_populations",
    "recipe",
    "source_identity",
    "task",
]
