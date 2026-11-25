"""Scalar-output ReLU BNNs and finite kinetic sampling computations.

Parameters use row-major ``(input_width, output_width)`` weights followed by
biases, one layer at a time. A force is the *positive* potential gradient, so
the kinetic update subtracts it. The continuous derivative gate is a prescribed
approximation; in general its force is not the gradient of another potential.

Precision is controlled by the caller's JAX configuration. No state, gradient,
or trajectory is clipped or repaired. The only clipping in this module defines
the paper's derivative gate. Failure accounting belongs to the experiment.
"""

from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise

import jax
import jax.numpy as jnp
import numpy as np

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


def initialize(
    key: jax.Array, architecture: Architecture, scale: float = 1.0
) -> jax.Array:
    """Draw independent centered normal coordinates with explicit scale."""
    return scale * jax.random.normal(key, (parameter_count(architecture),))


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


def signed_relu_dot(left: jax.Array, right: jax.Array) -> jax.Array:
    """Unnormalized paired ReLU heads, reducing the broadcast final axis.

    The two query-key scores are ``+left @ right`` and ``-left @ right``;
    their constant values are one, and output coefficients are ``(+1, -1)``.
    There is no softmax, token normalization, or learned attention parameter.
    """
    score = jnp.sum(left * right, axis=-1)
    return jax.nn.relu(score) - jax.nn.relu(-score)


def fixed_projection_dot(
    token: jax.Array,
    left_indices: tuple[int, ...],
    right_indices: tuple[int, ...],
    constant_index: int,
) -> jax.Array:
    """Evaluate the paired heads with literal fixed Q/K/V projection matrices.

    The last token axis contains both selected vectors and a coordinate equal
    to one. Indices are static circuit wiring; token entries contain the changing
    BNN parameters and activations. Leading batch dimensions are preserved.
    """
    if len(left_indices) != len(right_indices) or not left_indices:
        raise ValueError("The selected vectors must have the same positive length.")
    identity = jnp.eye(token.shape[-1], dtype=token.dtype)
    query_projection = identity[jnp.asarray(left_indices)]
    key_projection = identity[jnp.asarray(right_indices)]
    value_projection = identity[constant_index]
    query = token @ query_projection.T
    key = token @ key_projection.T
    value = token @ value_projection
    scores = jnp.stack(
        (jnp.sum(query * key, axis=-1), jnp.sum(-query * key, axis=-1)), axis=-1
    )
    output_projection = jnp.asarray((1.0, -1.0), dtype=token.dtype)
    return (jax.nn.relu(scores) @ output_projection) * value


def literal_attention_dot(left: jax.Array, right: jax.Array) -> jax.Array:
    """Pack a token and use fixed gather projections for a signed dot product."""
    left, right = jnp.broadcast_arrays(left, right)
    width = left.shape[-1]
    one = jnp.ones((*left.shape[:-1], 1), dtype=left.dtype)
    token = jnp.concatenate((left, right, one), axis=-1)
    return fixed_projection_dot(
        token, tuple(range(width)), tuple(range(width, 2 * width)), 2 * width
    )


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


def forward(q: jax.Array, x: jax.Array, architecture: Architecture) -> jax.Array:
    """Evaluate latent predictions with shape ``(observations, 1)``."""
    return _forward_cache(q, x, architecture)[0][-1]


def forward_cache(
    q: jax.Array, x: jax.Array, architecture: Architecture
) -> tuple[tuple[jax.Array, ...], tuple[jax.Array, ...]]:
    """Return activations (including input and output) and preactivations."""
    return _forward_cache(q, x, architecture)


def _responses(y: jax.Array, n: int) -> jax.Array:
    if y.shape == (n,):
        return y[:, None]
    if y.shape != (n, 1):
        raise ValueError("y must have shape (observations,) or (observations, 1).")
    return y


def potential(
    q: jax.Array,
    x: jax.Array,
    y: jax.Array,
    architecture: Architecture,
    sigma: float = 0.5,
    tau: float = 1.0,
) -> jax.Array:
    """Gaussian regression negative log density, excluding constants."""
    residual = forward(q, x, architecture) - _responses(y, x.shape[0])
    return jnp.sum(residual**2) / (2 * sigma**2) + jnp.sum(q**2) / (2 * tau**2)


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


def sensitivities(
    q: jax.Array,
    x: jax.Array,
    architecture: Architecture,
    gate_width: float = 1e-4,
    local_operation: LocalOperation | None = None,
    teacher_sensitivities: tuple[jax.Array, ...] | None = None,
) -> tuple[jax.Array, ...]:
    """Hidden unit-output sensitivities *before each unit's own gate*.

    Each output has shape ``(observations, hidden_width)``. The affine output
    sensitivity is fixed at one and is not included. For the local call,
    outgoing weights have shape ``(1, current_width, next_width)`` and next-layer
    gated sensitivities have shape ``(observations, 1, next_width)``. The supplied
    operation broadcasts these and reduces the last axis. It must return
    ``(observations, current_width)``. Optional teacher sensitivities replace
    downstream inputs during fitting only; returned values remain predictions.
    """
    _, preactivations = _forward_cache(q, x, architecture)
    return _sensitivities_from_cache(
        unpack(q, architecture),
        preactivations,
        gate_width,
        local_operation,
        teacher_sensitivities=teacher_sensitivities,
    )[0]


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


def force_from_sensitivities(
    q: jax.Array,
    x: jax.Array,
    y: jax.Array,
    architecture: Architecture,
    hidden_sensitivities: tuple[jax.Array, ...],
    gate_width: float = 1e-4,
    sigma: float = 0.5,
    tau: float = 1.0,
) -> jax.Array:
    """Assemble a positive force from predicted hidden PRE-gate sensitivities.

    Forward values, derivative gates, residual multiplication, observation
    aggregation, the output-layer derivative, and the prior remain exact.
    """
    activations, preactivations = _forward_cache(q, x, architecture)
    gates = tuple(derivative_gate(a, gate_width) for a in preactivations[:-1])
    return _gradient_from_cache(
        unpack(q, architecture), activations, y, hidden_sensitivities, gates, sigma, tau
    )


def attention_force(
    q: jax.Array,
    x: jax.Array,
    y: jax.Array,
    architecture: Architecture,
    gate_width: float = 1e-4,
    sigma: float = 0.5,
    tau: float = 1.0,
    *,
    literal_projections: bool = True,
) -> jax.Array:
    """Check the force's dot products using fixed signed ReLU attention heads.

    This explicitly evaluates the primitive Q/K/V circuits for all forward,
    backward and observation-aggregation dot products. Elementwise gates,
    residual products, sums and prior terms use their transparent tensor maps;
    this is not an exported all-operations attention network.
    """
    dot = literal_attention_dot if literal_projections else signed_relu_dot
    return _force(q, x, y, architecture, gate_width, sigma, tau, dot, dot=dot)


def kinetic_step(
    q: jax.Array,
    momentum: jax.Array,
    gradient: jax.Array,
    noise: jax.Array,
    *,
    h: float = 0.01,
    gamma: float = 0.15,
    parameter_mask: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """One paper Eq. (3) update using a supplied standard-normal noise draw."""
    next_momentum = (1 - gamma) * momentum - h * gradient + jnp.sqrt(2 * gamma) * noise
    if parameter_mask is not None:
        next_momentum = jnp.where(parameter_mask, next_momentum, 0.0)
    return q + h * next_momentum, next_momentum


def kinetic_trajectory(
    q0: jax.Array,
    v0: jax.Array,
    noises: jax.Array,
    force_fn: Callable[[jax.Array], jax.Array],
    *,
    h: float = 0.01,
    gamma: float = 0.15,
    parameter_mask: jax.Array | None = None,
    initial_gradient: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Scan a finite trajectory, returning positions and momenta *with start*.

    ``noises`` has shape ``(steps, parameters)``. Independent trajectories can
    be vectorized externally with ``jax.vmap``. An optional supplied first
    gradient implements the paper's early experimental convention explicitly.
    """
    if noises.ndim != 2 or noises.shape[1:] != q0.shape or v0.shape != q0.shape:
        raise ValueError("Expected matching flat states and (steps, parameters) noise.")

    def step(state, item):
        q, momentum = state
        index, noise = item
        gradient = (
            force_fn(q)
            if initial_gradient is None
            else jax.lax.cond(
                index == 0,
                lambda _: initial_gradient,
                lambda _: force_fn(q),
                operand=None,
            )
        )
        next_state = kinetic_step(
            q,
            momentum,
            gradient,
            noise,
            h=h,
            gamma=gamma,
            parameter_mask=parameter_mask,
        )
        return next_state, next_state

    _, (positions, momenta) = jax.lax.scan(
        step, (q0, v0), (jnp.arange(noises.shape[0]), noises)
    )
    return jnp.concatenate((q0[None, :], positions)), jnp.concatenate(
        (v0[None, :], momenta)
    )


@dataclass(frozen=True)
class ArchitectureMask:
    """Static embedding into a common width/depth with fixed linear bypasses."""

    native_architecture: Architecture
    padded_architecture: Architecture
    native_indices: tuple[int, ...]
    constants: jax.Array
    parameter_mask: jax.Array
    nonlinear_layers: tuple[bool, ...]

    def embed(self, q: jax.Array) -> jax.Array:
        if q.shape != (len(self.native_indices),):
            raise ValueError("The native parameter vector does not match this mask.")
        return (
            self.constants.astype(q.dtype).at[jnp.asarray(self.native_indices)].set(q)
        )

    def extract(self, padded_q: jax.Array) -> jax.Array:
        return padded_q[jnp.asarray(self.native_indices)]

    def pad_inputs(self, x: jax.Array) -> jax.Array:
        if x.shape[-1] != self.native_architecture[0]:
            raise ValueError("The input dimension does not match this mask.")
        return jnp.pad(x, ((0, 0), (0, self.padded_architecture[0] - x.shape[-1])))


def architecture_mask(
    architecture: Architecture,
    *,
    max_width: int = 7,
    max_hidden_depth: int = 3,
) -> ArchitectureMask:
    """Map native sampled coordinates to a fixed padded scalar-output network.

    Unused hidden slots contain fixed identity connections and apply no ReLU.
    The output layer is always the last slot. Zero padding handles narrower
    widths. Fixed identities, unused biases and unused weights are not sampled.
    """
    _validate_architecture(architecture)
    depth = len(architecture) - 2
    if (
        max_hidden_depth < depth
        or max_hidden_depth < 0
        or max(architecture[:-1]) > max_width
    ):
        raise ValueError("The architecture exceeds the padded width or depth.")
    padded = (max_width,) * (max_hidden_depth + 1) + (1,)
    constants = np.zeros(parameter_count(padded))
    mask = np.zeros(constants.shape, dtype=bool)
    indices = []
    offset = 0
    for slot, (din, dout) in enumerate(pairwise(padded)):
        is_output = slot == max_hidden_depth
        if slot < depth or is_output:
            native_slot = depth if is_output else slot
            native_in, native_out = architecture[native_slot : native_slot + 2]
            weight_indices = (
                offset
                + np.arange(native_in)[:, None] * dout
                + np.arange(native_out)[None, :]
            ).ravel()
            bias_indices = offset + din * dout + np.arange(native_out)
            indices.extend(weight_indices.tolist())
            indices.extend(bias_indices.tolist())
            mask[weight_indices] = True
            mask[bias_indices] = True
        else:
            active_width = architecture[-2]
            constants[offset + np.arange(active_width) * (dout + 1)] = 1.0
        offset += (din + 1) * dout
    return ArchitectureMask(
        architecture,
        padded,
        tuple(indices),
        jnp.asarray(constants),
        jnp.asarray(mask),
        (True,) * depth + (False,) * (max_hidden_depth - depth),
    )


def padded_forward(q: jax.Array, x: jax.Array, layout: ArchitectureMask) -> jax.Array:
    """Evaluate a padded state, preserving signed signals through bypasses."""
    return _forward_cache(
        q, layout.pad_inputs(x), layout.padded_architecture, layout.nonlinear_layers
    )[0][-1]


def padded_force(
    q: jax.Array,
    x: jax.Array,
    y: jax.Array,
    layout: ArchitectureMask,
    gate_width: float = 1e-4,
    sigma: float = 0.5,
    tau: float = 1.0,
) -> jax.Array:
    """Explicit force on sampled coordinates; fixed bypass force is zero."""
    gradient = _force(
        q,
        layout.pad_inputs(x),
        y,
        layout.padded_architecture,
        gate_width,
        sigma,
        tau,
        None,
        layout.nonlinear_layers,
    )
    return jnp.where(layout.parameter_mask, gradient, 0.0)
