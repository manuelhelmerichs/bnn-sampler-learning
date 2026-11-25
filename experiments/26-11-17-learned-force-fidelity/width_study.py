"""Small explicit-input width-transfer experiments.

Prepared source states and their acquisition identities are inputs: every role
maps widths to states of shape ``(states, width, input_dim + 1)``. Fitting never
samples reference populations. Checkpoints contain numeric arrays and JSON.
"""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import traverse_util
from runner import atomic_json, atomic_npz, fingerprint

from .width_models import (
    NeuronForce,
    ShallowBNN,
)


@dataclass(frozen=True)
class FitConfig:
    steps: int = 200
    learning_rate: float = 1e-3
    channels: int = 32
    heads: int = 4
    blocks: int = 3
    seed: int = 0
    features: bool = True
    mixer: str = "attention"
    normalize_force: bool = True
    reference_width: int = 10

    def __post_init__(self):
        if self.steps < 1 or self.learning_rate <= 0 or self.seed < 0:
            raise ValueError("Positive fitting budget and nonnegative seed required")


def _check(populations, identities, bnn):
    if set(populations) != {"train", "validation", "test"} or set(identities) != set(
        populations
    ):
        raise ValueError(
            "Explicit train, validation, and test populations and identities required"
        )
    if len(set(identities.values())) != 3 or any(
        not value for value in identities.values()
    ):
        raise ValueError(
            "Population acquisition identities must be nonempty and disjoint"
        )
    if not populations["train"] or set(populations["validation"]) != set(
        populations["train"]
    ):
        raise ValueError("Validation states must cover exactly the source widths")
    for role, by_width in populations.items():
        if not by_width:
            raise ValueError(f"No {role} widths")
        for width, states in by_width.items():
            if (
                states.ndim != 3
                or states.shape[0] < 1
                or states.shape[1] != width
                or states.shape[-1] != bnn.input_dim + 1
            ):
                raise ValueError(f"Invalid {role} state shape at width {width}")
            if not bool(jnp.isfinite(states).all()):
                raise ValueError(f"Nonfinite {role} states at width {width}")


def _model(bnn, config):
    return NeuronForce(
        bnn,
        channels=config.channels,
        heads=config.heads,
        blocks=config.blocks,
        features=config.features,
        mixer=config.mixer,
        normalize_force=config.normalize_force,
        reference_width=config.reference_width,
    )


def evaluate_force(model, variables, bnn, states, x, y):
    """Measure held-out force error independently of any fitting update."""
    expected = bnn.gradient(states, x, y)
    error = model.apply(variables, states, x, y) - expected
    ratio = jnp.square(error).mean() / jnp.maximum(jnp.square(expected).mean(), 1e-30)
    return {
        "relative_force_rmse": float(jnp.sqrt(ratio)),
        "states": len(states),
        "width": states.shape[1],
    }


def _evaluate_widths(model, variables, bnn, by_width, x, y):
    return [
        evaluate_force(model, variables, bnn, states, x, y)
        for _, states in sorted(by_width.items())
    ]


def _checksum(value):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(value)).tobytes()).hexdigest()


def fit_force(bnn, populations, identities, x, y, *, config=None, output=None):
    """Fit jointly on the source widths, select on their mean validation error,
    then evaluate every test width once.

    Returns the model, its selected variables, and the fit record.
    """
    config = FitConfig() if config is None else config
    _check(populations, identities, bnn)
    if output is not None and Path(output).exists():
        raise FileExistsError("Use a fresh output directory for a fitted study")
    model = _model(bnn, config)
    training = dict(sorted(populations["train"].items()))
    variables = model.init(
        jax.random.key(config.seed), next(iter(training.values())), x, y
    )
    truths = {width: bnn.gradient(q, x, y) for width, q in training.items()}
    multipliers = {width: model.output_multiplier(width) for width in training}
    # One per-coordinate likelihood-force scale shared by all source widths.
    scale = jnp.maximum(
        jnp.sqrt(
            jnp.mean(
                jnp.stack(
                    [
                        jnp.square((truths[w] - q) / multipliers[w]).mean(axis=(0, 1))
                        for w, q in training.items()
                    ]
                ),
                axis=0,
            )
        ),
        1e-6,
    )
    constants = {
        "force_scale": scale.astype(variables["constants"]["force_scale"].dtype)
    }
    optimizer = optax.adam(config.learning_rate)

    @jax.jit
    def step(params, state):
        def loss(params):
            current = {"params": params, "constants": constants}
            return jnp.mean(
                jnp.stack(
                    [
                        jnp.square(
                            (model.apply(current, q, x, y) - truths[w])
                            / (scale * multipliers[w])
                        ).mean()
                        for w, q in training.items()
                    ]
                )
            )

        value, gradient = jax.value_and_grad(loss)(params)
        updates, state = optimizer.update(gradient, state, params)
        return optax.apply_updates(params, updates), state, value

    params, state = variables["params"], optimizer.init(variables["params"])
    best, selected, selected_step = float("inf"), None, 0
    for number in range(1, config.steps + 1):
        params, state, loss = step(params, state)
        if not bool(jnp.isfinite(loss)):
            raise ValueError("Force fitting became nonfinite")
        current = {"params": params, "constants": constants}
        score = float(
            np.mean(
                [
                    row["relative_force_rmse"]
                    for row in _evaluate_widths(
                        model, current, bnn, populations["validation"], x, y
                    )
                ]
            )
        )
        if not np.isfinite(score):
            raise ValueError("Validation force evaluation became nonfinite")
        if score < best:
            best, selected, selected_step = score, current, number
    record = {
        "model": asdict(bnn),
        "config": asdict(config),
        "inputs": identities,
        "selected_step": selected_step,
        "input_checksums": {
            **{
                f"{role}_M{width}": _checksum(states)
                for role, by_width in populations.items()
                for width, states in sorted(by_width.items())
            },
            "observed_x": _checksum(x),
            "observed_y": _checksum(y),
        },
        "source_checksums": {
            name: hashlib.sha256(
                Path(__file__).with_name(name).read_bytes()
            ).hexdigest()
            for name in ("width_models.py", "width_study.py")
        },
        "validation": _evaluate_widths(
            model, selected, bnn, populations["validation"], x, y
        ),
        "test": _evaluate_widths(model, selected, bnn, populations["test"], x, y),
    }
    if output is not None:
        output = Path(output)
        output.mkdir(parents=True)
        flat = traverse_util.flatten_dict(selected, sep="/")
        atomic_npz(
            output / "weights.npz", **{k: np.asarray(v) for k, v in flat.items()}
        )
        record["weights_sha256"] = hashlib.sha256(
            (output / "weights.npz").read_bytes()
        ).hexdigest()
    record["identity"] = fingerprint(record, namespace="width-force-study")
    if output is not None:
        atomic_json(output / "fit.json", record)
    return model, selected, record


def load_fit(path):
    """Read a fitted numeric checkpoint; this never trains a model."""
    path = Path(path)
    record = json.loads((path / "fit.json").read_text())
    identity = record["identity"]
    if (
        fingerprint(
            {key: value for key, value in record.items() if key != "identity"},
            namespace="width-force-study",
        )
        != identity
    ):
        raise ValueError("Fitted width study identity changed")
    if (
        hashlib.sha256((path / "weights.npz").read_bytes()).hexdigest()
        != record["weights_sha256"]
    ):
        raise ValueError("Fitted width study weights changed")
    config, bnn = FitConfig(**record["config"]), ShallowBNN(**record["model"])
    with np.load(path / "weights.npz", allow_pickle=False) as archive:
        flat = {name: jnp.asarray(archive[name]) for name in archive.files}
    variables = traverse_util.unflatten_dict(flat, sep="/")
    return _model(bnn, config), variables, record
