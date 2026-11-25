"""The draft's learned forces, whose operation counts ``learners`` compares.

Copied from ``26-11-17-learned-force-fidelity`` (``core``, ``student`` and
``width_models``), restricted to what the counts evaluate: the explicit force of
a scalar-output ReLU BNN with an optional learned local operation, the student
that provides it, and the width-equivariant whole-force models. The student's
``pairwise`` is renamed ``pair_map`` to keep ``itertools.pairwise``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise

import jax
import jax.numpy as jnp
from flax import linen as nn

Architecture = tuple[int, ...]
LocalOperation = Callable[[jax.Array, jax.Array], jax.Array]


def _validate_architecture(architecture: Architecture) -> None:
    if len(architecture) < 2 or any(width < 1 for width in architecture):
        raise ValueError("An architecture requires at least two positive widths.")
    if architecture[-1] != 1:
        raise ValueError("The learned local recurrence requires a scalar output.")


def parameter_count(architecture: Architecture) -> int:
    """Number of sampled weights and biases, including the affine output."""
    _validate_architecture(architecture)
    return sum((din + 1) * dout for din, dout in pairwise(architecture))


def unpack(
    q: jax.Array, architecture: Architecture
) -> tuple[tuple[jax.Array, jax.Array], ...]:
    """Return ``(W, b)`` pairs without changing the flat parameter ordering."""
    if q.ndim != 1 or q.shape[0] != parameter_count(architecture):
        raise ValueError("q must be a flat vector matching the architecture.")
    offset = 0
    layers = []
    for din, dout in pairwise(architecture):
        size = din * dout
        weights = q[offset : offset + size].reshape(din, dout)
        bias = q[offset + size : offset + size + dout]
        layers.append((weights, bias))
        offset += size + dout
    return tuple(layers)


def derivative_gate(a: jax.Array, gate_width: float = 1e-4) -> jax.Array:
    """One-sided ramp, with the hard ReLU derivative when ``gate_width=0``.

    ``clip(a / delta, 0, 1)`` is the numerically stable evaluation of
    ``relu(a / delta) - relu(a / delta - 1)``. Both assign zero at zero.
    """
    if gate_width < 0:
        raise ValueError("gate_width must be nonnegative.")
    if gate_width == 0:
        return (a > 0).astype(a.dtype)
    return jnp.clip(a / gate_width, 0.0, 1.0)


def _matmul(left: jax.Array, right: jax.Array, dot: LocalOperation | None) -> jax.Array:
    if dot is None:
        return left @ right
    return dot(left[..., :, None, :], right.T[None, ...])


def _forward_cache(
    q: jax.Array,
    x: jax.Array,
    architecture: Architecture,
    nonlinear_layers: tuple[bool, ...] | None = None,
    dot: LocalOperation | None = None,
) -> tuple[tuple[jax.Array, ...], tuple[jax.Array, ...]]:
    if x.ndim != 2 or x.shape[1] != architecture[0]:
        raise ValueError("x must have shape (observations, input_width).")
    nonlinear = (
        nonlinear_layers
        if nonlinear_layers is not None
        else (True,) * (len(architecture) - 2)
    )
    if len(nonlinear) != len(architecture) - 2:
        raise ValueError("One nonlinearity flag is required per hidden layer.")
    activations = [x]
    preactivations = []
    for layer, (weights, bias) in enumerate(unpack(q, architecture)):
        a = _matmul(activations[-1], weights, dot) + bias
        preactivations.append(a)
        hidden = layer < len(nonlinear)
        activations.append(jax.nn.relu(a) if hidden and nonlinear[layer] else a)
    return tuple(activations), tuple(preactivations)


def _responses(y: jax.Array, n: int) -> jax.Array:
    if y.shape == (n,):
        return y[:, None]
    if y.shape != (n, 1):
        raise ValueError("y must have shape (observations,) or (observations, 1).")
    return y


def _sensitivities_from_cache(
    layers: tuple[tuple[jax.Array, jax.Array], ...],
    preactivations: tuple[jax.Array, ...],
    gate_width: float,
    local_operation: LocalOperation | None,
    nonlinear_layers: tuple[bool, ...] | None = None,
    teacher_sensitivities: tuple[jax.Array, ...] | None = None,
) -> tuple[tuple[jax.Array, ...], tuple[jax.Array, ...]]:
    nonlinear = (
        nonlinear_layers
        if nonlinear_layers is not None
        else (True,) * (len(layers) - 1)
    )
    gates = tuple(
        derivative_gate(a, gate_width) if flag else jnp.ones_like(a)
        for a, flag in zip(preactivations[:-1], nonlinear, strict=True)
    )
    # The scalar output sensitivity is one for every observation. Evaluating the
    # final local operation once avoids repeating an identical learned MLP call.
    downstream = jnp.ones((1, 1), dtype=preactivations[-1].dtype)
    if (
        teacher_sensitivities is not None
        and len(teacher_sensitivities) != len(layers) - 1
    ):
        raise ValueError(
            "Teacher sensitivities must contain one array per hidden layer."
        )
    hidden_sensitivities = []
    for layer in range(len(layers) - 2, -1, -1):
        weights = layers[layer + 1][0]
        if local_operation is None:
            sensitivity = downstream @ weights.T
        else:
            sensitivity = local_operation(weights[None, :, :], downstream[:, None, :])
        sensitivity = jnp.broadcast_to(sensitivity, gates[layer].shape)
        hidden_sensitivities.append(sensitivity)
        downstream = (
            sensitivity
            if teacher_sensitivities is None
            else teacher_sensitivities[layer]
        ) * gates[layer]
    return tuple(reversed(hidden_sensitivities)), gates


def _gradient_from_cache(
    layers: tuple[tuple[jax.Array, jax.Array], ...],
    activations: tuple[jax.Array, ...],
    y: jax.Array,
    hidden: tuple[jax.Array, ...],
    gates: tuple[jax.Array, ...],
    sigma: float,
    tau: float,
    dot: LocalOperation | None = None,
) -> jax.Array:
    if len(hidden) != len(gates):
        raise ValueError("One sensitivity array is required per hidden layer.")
    residual = (activations[-1] - _responses(y, activations[0].shape[0])) / sigma**2
    preactivation_sensitivities = (
        *(s * gate for s, gate in zip(hidden, gates, strict=True)),
        jnp.ones_like(residual),
    )
    gradient = []
    for (weights, bias), a, sensitivity in zip(
        layers, activations[:-1], preactivation_sensitivities, strict=True
    ):
        delta = residual * sensitivity
        weight_gradient = _matmul(a.T, delta, dot) + weights / tau**2
        bias_gradient = jnp.sum(delta, axis=0) + bias / tau**2
        gradient.extend((weight_gradient.ravel(), bias_gradient))
    return jnp.concatenate(gradient)


def _force(
    q: jax.Array,
    x: jax.Array,
    y: jax.Array,
    architecture: Architecture,
    gate_width: float,
    sigma: float,
    tau: float,
    local_operation: LocalOperation | None,
    nonlinear_layers: tuple[bool, ...] | None = None,
    dot: LocalOperation | None = None,
    teacher_sensitivities: tuple[jax.Array, ...] | None = None,
) -> jax.Array:
    layers = unpack(q, architecture)
    activations, preactivations = _forward_cache(
        q, x, architecture, nonlinear_layers, dot
    )
    hidden, gates = _sensitivities_from_cache(
        layers,
        preactivations,
        gate_width,
        local_operation,
        nonlinear_layers,
        teacher_sensitivities,
    )
    return _gradient_from_cache(layers, activations, y, hidden, gates, sigma, tau, dot)


def force(
    q: jax.Array,
    x: jax.Array,
    y: jax.Array,
    architecture: Architecture,
    gate_width: float = 1e-4,
    sigma: float = 0.5,
    tau: float = 1.0,
    local_operation: LocalOperation | None = None,
    teacher_sensitivities: tuple[jax.Array, ...] | None = None,
) -> jax.Array:
    """Explicit positive force with exact prior and output-layer derivatives."""
    return _force(
        q,
        x,
        y,
        architecture,
        gate_width,
        sigma,
        tau,
        local_operation,
        teacher_sensitivities=teacher_sensitivities,
    )


def initialize_student(key, *, width=32, dtype=jnp.float32):
    """Two SiLU hidden layers; width 32 has exactly 1,185 parameters."""
    keys = jax.random.split(key, 3)
    dimensions = (2, width, width, 1)
    return tuple(
        (
            jax.random.normal(k, (din, dout), dtype=dtype) / jnp.sqrt(din),
            jnp.zeros(dout, dtype=dtype),
        )
        for k, din, dout in zip(keys, dimensions[:-1], dimensions[1:], strict=True)
    )


def pair_map(params, a, b, *, odd=False):
    """Evaluate the scalar pair map, optionally projected onto separate oddness."""
    a, b = jnp.broadcast_arrays(a, b)

    def mlp(x, y):
        values = jnp.stack((x, y), axis=-1)
        for weight, bias in params[:-1]:
            values = jax.nn.silu(values @ weight + bias)
        weight, bias = params[-1]
        return (values @ weight + bias)[..., 0]

    if odd:
        return (mlp(a, b) - mlp(-a, b) - mlp(a, -b) + mlp(-a, -b)) / 4
    return mlp(a, b)


def local_operation(params, weights, downstream, *, odd=False):
    """Equation (12): normalize neighbors, learn pairs, sum, bound and rescale."""
    # Floor the squared norm before sqrt: differentiating norm(0) first would
    # produce NaNs even when a later maximum selects the positive floor.
    cw = jnp.sqrt(jnp.maximum(jnp.sum(weights**2, axis=-1, keepdims=True), 1e-16))
    cu = jnp.sqrt(jnp.maximum(jnp.sum(downstream**2, axis=-1, keepdims=True), 1e-16))
    value = jnp.sum(pair_map(params, weights / cw, downstream / cu, odd=odd), axis=-1)
    return (cw * cu)[..., 0] * jnp.clip(value, -1.0, 1.0)


@dataclass(frozen=True)
class ShallowBNN:
    input_dim: int = 5
    noise_sd: float = 1.0
    scaling: str = "sqrt_width"

    def __post_init__(self):
        if self.input_dim < 1 or self.noise_sd <= 0:
            raise ValueError("Positive input dimension and noise SD required")
        if self.scaling not in {"sqrt_width", "unscaled"}:
            raise ValueError(self.scaling)

    def multiplier(self, width: int) -> float:
        if width < 1:
            raise ValueError("Width must be positive")
        return width**-0.5 if self.scaling == "sqrt_width" else 1.0

    def features(self, q, x):
        """q has shape (batch, neurons, p+1); all stored coordinates have N(0,1) priors."""
        z = jnp.einsum("bmp,np->bnm", q[..., :-1], x / math.sqrt(self.input_dim))
        return z, jax.nn.relu(z)

    def predict(self, q, x):
        _, a = self.features(q, x)
        return self.multiplier(q.shape[1]) * (a * q[:, None, :, -1]).sum(-1)

    def potential(self, q, x, y):
        return 0.5 * jnp.square((self.predict(q, x) - y) / self.noise_sd).sum(
            -1
        ) + 0.5 * jnp.square(q).sum((1, 2))

    def gradient(self, q, x, y):
        z, a = self.features(q, x)
        scale = self.multiplier(q.shape[1])
        residual = (scale * (a * q[:, None, :, -1]).sum(-1) - y) / self.noise_sd**2
        incoming = jnp.einsum(
            "bn,bnm,np->bmp",
            residual,
            (z > 0) * q[:, None, :, -1],
            x / math.sqrt(self.input_dim),
        )
        outgoing = jnp.einsum("bn,bnm->bm", residual, a)
        return q + scale * jnp.concatenate((incoming, outgoing[..., None]), -1)

    def jacobian(self, q, x):
        """Data Jacobian, (batch, observations, M*(p+1))."""
        z, a = self.features(q, x)
        incoming = (
            (z > 0)[..., None]
            * q[:, None, :, -1, None]
            * x[None, :, None, :]
            / math.sqrt(self.input_dim)
        )
        stacked = jnp.concatenate((incoming, a[..., None]), -1)
        return self.multiplier(q.shape[1]) * stacked.reshape(*stacked.shape[:2], -1)


class InspectableAttention(nn.Module):
    """Ordinary row-softmax attention; no extra 1/M factor is inserted."""

    channels: int
    heads: int

    def setup(self):
        if self.channels % self.heads:
            raise ValueError("Channels must be divisible by heads")
        self.query = nn.Dense(self.channels)
        self.key = nn.Dense(self.channels)
        self.value = nn.Dense(self.channels)
        self.output = nn.Dense(self.channels)

    def __call__(self, x, context, *, ablate: int | None = None):
        head_dim = self.channels // self.heads

        def split(t):
            return t.reshape(t.shape[0], -1, self.heads, head_dim).transpose(0, 2, 1, 3)

        query = split(self.query(x))
        key, value = split(self.key(context)), split(self.value(context))
        weights = jax.nn.softmax(
            query @ key.transpose(0, 1, 3, 2) / math.sqrt(head_dim), axis=-1
        )
        attended = weights @ value
        if ablate is not None:
            if not 0 <= ablate < self.heads:
                raise ValueError(ablate)
            attended = attended.at[:, ablate].set(0)
        merged = attended.transpose(0, 2, 1, 3).reshape(*x.shape[:2], self.channels)
        return self.output(merged), weights


class _FeedForward(nn.Module):
    channels: int

    @nn.compact
    def __call__(self, h):
        return nn.Dense(self.channels)(nn.silu(nn.Dense(2 * self.channels)(h)))


class NeuronBlock(nn.Module):
    channels: int
    heads: int
    mixer: str

    def setup(self):
        # Matches the usual LayerNorm epsilon of the original fits.
        self.self_norm = nn.LayerNorm(epsilon=1e-5)
        self.cross_norm = nn.LayerNorm(epsilon=1e-5)
        self.ff_norm = nn.LayerNorm(epsilon=1e-5)
        if self.mixer == "attention":
            self.self_attention = InspectableAttention(self.channels, self.heads)
        elif self.mixer == "mean":
            self.mean_message = _FeedForward(self.channels)
        else:
            raise ValueError(self.mixer)
        self.cross_attention = InspectableAttention(self.channels, self.heads)
        self.ff = _FeedForward(self.channels)

    def __call__(self, h, data, *, ablate=None):
        normalized = self.self_norm(h)
        weights = None
        if self.mixer == "attention":
            update, weights = self.self_attention(normalized, normalized, ablate=ablate)
        else:
            if ablate is not None:
                raise ValueError("Mean mixer has no heads")
            update = jnp.broadcast_to(
                self.mean_message(normalized.mean(1, keepdims=True)), h.shape
            )
        h = h + update
        h = h + self.cross_attention(self.cross_norm(h), data)[0]
        h = h + self.ff(self.ff_norm(h))
        return h, weights


class NeuronForce(nn.Module):
    """One model handles each width without padding, masks or neuron IDs.

    Input and output coordinate types are shared across neurons. log(M) exposes
    cardinality, which normalized attention alone cannot recover from replicas.
    Likelihood-force output scale is fitted on source widths only (the
    ``constants`` collection). Scaling the readout by sqrt(M_ref/M) does NOT
    change the BNN prior or Langevin noise.
    """

    bnn: ShallowBNN
    features: bool = True
    normalize_force: bool = True
    mixer: str = "attention"
    channels: int = 32
    heads: int = 4
    blocks: int = 3
    reference_width: int = 10

    def output_multiplier(self, width):
        if not self.normalize_force:
            return 1.0
        return self.bnn.multiplier(width) / self.bnn.multiplier(self.reference_width)

    @nn.compact
    def __call__(self, q, x, y, *, inspect=False, ablate=None):
        batch, width, _ = q.shape
        p = self.bnn.input_dim
        force_scale = self.variable("constants", "force_scale", jnp.ones, (p + 1,))
        cardinality = jnp.full(
            (batch, width, 1), math.log(width / self.reference_width), q.dtype
        )
        h = nn.Dense(self.channels, name="neuron_embed")(
            jnp.concatenate((q, cardinality), -1)
        )
        xy = jnp.concatenate((x / math.sqrt(p), y[:, None]), -1)
        data = nn.Dense(self.channels, name="data_out")(
            nn.silu(nn.Dense(self.channels, name="data_in")(xy))
        )
        data = jnp.broadcast_to(data[None], (batch, *data.shape))
        if self.features:
            z, a = self.bnn.features(q, x)
            pair = jnp.concatenate(
                (
                    jnp.broadcast_to(xy[None, :, None], (batch, len(xy), width, p + 1)),
                    z[..., None],
                    a[..., None],
                ),
                -1,
            )
            encoded = nn.Dense(self.channels, name="feature_out")(
                nn.silu(nn.Dense(self.channels, name="feature_in")(pair))
            )
            h = h + encoded.mean(1)
        attention, hidden = [], []
        for layer in range(self.blocks):
            head = ablate[1] if ablate is not None and ablate[0] == layer else None
            block = NeuronBlock(
                self.channels, self.heads, self.mixer, name=f"block_{layer}"
            )
            h, weights = block(h, data, ablate=head)
            attention.append(weights)
            hidden.append(h)
        readout = nn.Dense(p + 1, name="readout")(h)
        force = q + readout * force_scale.value * self.output_multiplier(width)
        return (force, {"attention": attention, "hidden": hidden}) if inspect else force
