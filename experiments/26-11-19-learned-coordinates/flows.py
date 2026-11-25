"""A coupling flow as the coordinates of a chain, and the two ways it is fitted.

``x = T(z)``: affine coupling layers, each after a fixed orthogonal mixing of
its input, then a learned affine map with a triangular factor. The mixing is a
random rotation, which makes the flow blind to the frame of the target, or a
random permutation, which lets it use the target's own coordinates
(``permutation`` as matrices, ``index`` as indexes).

``fit_reverse`` minimizes ``KL[T(N) | pi]`` from standard normal draws, as
NeuTra does: every draw costs one gradient of the target. ``fit_forward``
maximizes the likelihood of exact draws of the target; it bounds what the family
can do and is not a method.

A flow is a dictionary ``{"train": ..., "mix": ...}``; batches of flows have the
target as their leading axis.
"""

import jax
import jax.numpy as jnp
import optax

# Bound on the log scale of one coupling layer.
BOUND = 4.0
# Clip of the gradient norm, and the decay of the averaged parameters.
CLIP, DECAY = 100.0, 0.99
WARMUP = 100


def create(key, dimension, layers, hidden, mixing):
    half = dimension // 2
    rest = dimension - half
    first, second, mix_key = jax.random.split(key, 3)

    def weights(key, rows, columns):
        return jax.random.normal(key, (layers, rows, columns), jnp.float64) / jnp.sqrt(
            columns
        )

    net = {
        "w1": weights(first, hidden, half),
        "b1": jnp.zeros((layers, hidden)),
        "w2": weights(second, hidden, hidden),
        "b2": jnp.zeros((layers, hidden)),
        # The flow starts as the identity.
        "w3": jnp.zeros((layers, 2 * rest, hidden)),
        "b3": jnp.zeros((layers, 2 * rest)),
    }
    if mixing == "rotation":
        q, r = jnp.linalg.qr(
            jax.random.normal(mix_key, (layers, dimension, dimension), jnp.float64)
        )
        mix = q * jnp.sign(jnp.diagonal(r, axis1=1, axis2=2))[:, None, :]
    else:
        orders = jax.vmap(lambda k: jax.random.permutation(k, dimension))(
            jax.random.split(mix_key, layers)
        )
        # The same permutations as matrices, or as indexes where D is large.
        mix = orders if mixing == "index" else jnp.eye(dimension)[orders]
    return {
        "train": {
            "net": net,
            "shift": jnp.zeros(dimension),
            "lower": jnp.zeros((dimension, dimension)),
            "log_diagonal": jnp.zeros(dimension),
        },
        "mix": mix,
    }


def _conditioner(net, a):
    h = jax.nn.silu(net["w1"] @ a + net["b1"])
    h = jax.nn.silu(net["w2"] @ h + net["b2"])
    raw, shift = jnp.split(net["w3"] @ h + net["b3"], 2)
    return BOUND * jnp.tanh(raw / BOUND), shift


def _mixed(mix, y):
    return y[mix] if jnp.issubdtype(mix.dtype, jnp.integer) else mix @ y


def _unmixed(mix, y):
    if jnp.issubdtype(mix.dtype, jnp.integer):
        return jnp.zeros_like(y).at[mix].set(y)
    return mix.T @ y


def _factor(train):
    return jnp.tril(train["lower"], -1) + jnp.diag(jnp.exp(train["log_diagonal"]))


def forward(flow, z):
    """``x = T(z)`` and ``log |det dT/dz|``."""
    train = flow["train"]
    half = train["net"]["w1"].shape[-1]

    def layer(carry, args):
        y, logdet = carry
        net, mix = args
        y = _mixed(mix, y)
        a, b = y[:half], y[half:]
        scale, shift = _conditioner(net, a)
        return (
            jnp.concatenate([a, b * jnp.exp(scale) + shift]),
            logdet + scale.sum(),
        ), None

    (y, logdet), _ = jax.lax.scan(
        layer, (z, jnp.zeros((), z.dtype)), (train["net"], flow["mix"])
    )
    return train["shift"] + _factor(train) @ y, logdet + train["log_diagonal"].sum()


def inverse(flow, x):
    """``z`` with ``T(z) = x`` and ``log |det dz/dx|``."""
    train = flow["train"]
    half = train["net"]["w1"].shape[-1]
    y = jax.scipy.linalg.solve_triangular(
        _factor(train), x - train["shift"], lower=True
    )

    def layer(carry, args):
        y, logdet = carry
        net, mix = args
        a, b = y[:half], y[half:]
        scale, shift = _conditioner(net, a)
        y = jnp.concatenate([a, (b - shift) * jnp.exp(-scale)])
        return (_unmixed(mix, y), logdet - scale.sum()), None

    (z, logdet), _ = jax.lax.scan(
        layer,
        (y, jnp.zeros((), x.dtype)),
        (train["net"], flow["mix"]),
        reverse=True,
    )
    return z, logdet - train["log_diagonal"].sum()


def _trainer(loss, rate):
    """Adam on ``loss(train, mix, data, key)`` for a batch of flows, with averaging."""
    optimizer = optax.chain(
        optax.clip_by_global_norm(CLIP),
        optax.adam(optax.linear_schedule(rate / WARMUP, rate, WARMUP)),
    )

    def one(carry, key, mix, data):
        train, average, state = carry
        value, gradient = jax.value_and_grad(loss)(train, mix, data, key)
        fine = jnp.isfinite(value) & jnp.all(
            jnp.asarray([jnp.all(jnp.isfinite(g)) for g in jax.tree.leaves(gradient)])
        )
        gradient = jax.tree.map(lambda g: jnp.where(fine, g, 0.0), gradient)
        updates, state = optimizer.update(gradient, state, train)
        train = optax.apply_updates(train, updates)
        average = jax.tree.map(lambda a, t: DECAY * a + (1 - DECAY) * t, average, train)
        return (train, average, state), value

    @jax.jit
    def segment(carry, keys, mix, data):
        def run(carry, keys, mix, data):
            return jax.lax.scan(lambda c, k: one(c, k, mix, data), carry, keys)

        return jax.vmap(run)(carry, keys, mix, data)

    def start(flows):
        train = flows["train"]
        return train, train, jax.vmap(optimizer.init)(train)

    return start, segment


def _fit(loss, flows, data, key, snapshots, rate):
    start, segment = _trainer(loss, rate)
    carry, targets = start(flows), flows["mix"].shape[0]
    fitted, traces, previous = {}, [], 0
    for steps in snapshots:
        key, subkey = jax.random.split(key)
        carry, values = segment(
            carry,
            jax.random.split(subkey, (targets, steps - previous)),
            flows["mix"],
            data,
        )
        fitted[steps] = {"train": carry[1], "mix": flows["mix"]}
        traces.append(values)
        previous = steps
    return fitted, jnp.concatenate(traces, 1)


def fit_reverse(energy, flows, data, key, *, snapshots, batch, rate):
    """Flows after each number of steps in ``snapshots``, and the loss per step.

    ``energy(x, data)`` is the potential of one target; ``data`` has the target
    as its leading axis. A step takes ``batch`` gradients of the potential.
    """

    def loss(train, mix, data, key):
        z = jax.random.normal(key, (batch, mix.shape[-1]), jnp.float64)
        x, logdet = jax.vmap(lambda z: forward({"train": train, "mix": mix}, z))(z)
        return jnp.mean(jax.vmap(lambda x: energy(x, data))(x) - logdet)

    return _fit(loss, flows, data, key, snapshots, rate)


def fit_forward(draw, flows, data, key, *, snapshots, batch, rate):
    """As ``fit_reverse``, on ``draw(key, data, batch)``: exact draws of each target."""

    def loss(train, mix, data, key):
        x = draw(key, data, batch)
        z, logdet = jax.vmap(lambda x: inverse({"train": train, "mix": mix}, x))(x)
        return jnp.mean(0.5 * jnp.sum(z**2, -1) - logdet)

    return _fit(loss, flows, data, key, snapshots, rate)


def quality(energy, flows, data, key, count):
    """Per flow: ``KL[T(N) | pi]`` up to the target's constant, and the share of
    ``count`` draws of the flow that importance weights leave effective."""

    def one(flow, data, key):
        z = jax.random.normal(key, (count, flow["mix"].shape[-1]), jnp.float64)
        x, logdet = jax.vmap(lambda z: forward(flow, z))(z)
        log_q = (
            -0.5 * jnp.sum(z**2, -1) - 0.5 * z.shape[-1] * jnp.log(2 * jnp.pi) - logdet
        )
        log_weight = -jax.vmap(lambda x: energy(x, data))(x) - log_q
        log_weight = jnp.where(jnp.isfinite(log_weight), log_weight, -jnp.inf)
        weight = jnp.exp(log_weight - log_weight.max())
        return -jnp.mean(log_weight), weight.sum() ** 2 / (count * jnp.sum(weight**2))

    return jax.jit(jax.vmap(one))(
        flows, data, jax.random.split(key, flows["mix"].shape[0])
    )
