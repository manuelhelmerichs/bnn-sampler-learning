"""Width-equivariant direct likelihood-force models for shallow ReLU BNNs.

The Gaussian prior and Langevin update are known. No likelihood residual,
derivative gate, backward sensitivity, or force assembly is supplied to a student.
The optional feature encoder sees individual preactivations and activations only.
This is a finite-width experiment, not an implementation of sigmaTransfer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn


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


def ula(force, initial, *, step_size, steps, seed, save_steps=(), record_paths=0):
    """Independent finite-horizon paths; no gradient clipping or teacher fallback."""
    q = jnp.asarray(initial)
    key = jax.random.key(seed)
    endpoints, states = {}, []
    for step in range(1, steps + 1):
        if record_paths:
            states.append(np.asarray(q[:record_paths]))
        key, noise = jax.random.split(key)
        q = (
            q
            - step_size * force(q)
            + math.sqrt(2 * step_size) * jax.random.normal(noise, q.shape, q.dtype)
        )
        if not bool(jnp.isfinite(q).all()):
            return {"status": "nonfinite", "failed_step": step, "endpoints": endpoints}
        if step in save_steps:
            endpoints[step] = np.asarray(q)
    return {
        "status": "finite",
        "endpoints": endpoints,
        "states": np.stack(states) if states else None,
    }
