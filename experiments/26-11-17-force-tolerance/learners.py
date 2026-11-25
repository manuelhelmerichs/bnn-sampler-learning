"""A plain MLP force, and what learned forces cost relative to a gradient.

The MLP maps a position in the task's own coordinates to the force: two SiLU
hidden layers, inputs and outputs standardized on the corpus, fitted with Adam
on the relative squared error ``|g_hat - g|^2 / mean |g|^2``. Its square root
on exact draws is the force error delta of the oracle arms.

Costs are XLA's operation counts of the compiled computations, so they do not
depend on how fast a toy runs on this machine.
"""

import math
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax


def initialize(key, dimension, width):
    sizes = (dimension, width, width, dimension)
    layers = []
    for subkey, (fan_in, fan_out) in zip(
        jax.random.split(key, 3), zip(sizes[:-1], sizes[1:], strict=True), strict=True
    ):
        layers.append(
            (
                jax.random.normal(subkey, (fan_in, fan_out)) / math.sqrt(fan_in),
                jnp.zeros(fan_out),
            )
        )
    return layers


def apply(params, theta):
    x = (theta - params["shift_in"]) / params["scale_in"]
    *hidden, (weight, bias) = params["layers"]
    for w, b in hidden:
        x = jax.nn.silu(x @ w + b)
    return params["shift_out"] + params["scale_out"] * (x @ weight + bias)


def relative_error(params, positions, forces):
    """sqrt(mean |g_hat - g|^2 / mean |g|^2)."""
    error = jax.vmap(partial(apply, params))(positions) - forces
    return float(jnp.sqrt(jnp.sum(error**2) / jnp.sum(forces**2)))


def fit(positions, forces, spec, seed):
    """Fit on all but a held-out share, keeping the checkpoint best on it.

    Returns the parameters and a record. ``evaluations`` counts forward-pass
    equivalents: a training example costs three, a held-out one one.
    """
    positions, forces = jnp.asarray(positions), jnp.asarray(forces)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(positions))
    held = int(spec["holdout"] * len(positions))
    test, train = order[:held], order[held:]
    x, g = positions[train], forces[train]
    params = {
        "shift_in": x.mean(0),
        "scale_in": x.std(0) + 1e-12,
        "shift_out": g.mean(0),
        "scale_out": g.std(0) + 1e-12,
        "layers": initialize(jax.random.key(seed), positions.shape[1], spec["width"]),
    }
    power = jnp.mean(jnp.sum(g**2, 1))
    schedule = optax.cosine_decay_schedule(spec["rate"], spec["steps"])
    optimizer = optax.adam(schedule)
    trainable = params["layers"]
    state = optimizer.init(trainable)

    def loss(layers, batch_x, batch_g):
        current = dict(params, layers=layers)
        error = jax.vmap(partial(apply, current))(batch_x) - batch_g
        return jnp.mean(jnp.sum(error**2, 1)) / power

    @jax.jit
    def chunk(carry, key):
        def step(carry, subkey):
            layers, state = carry
            index = jax.random.randint(subkey, (spec["batch"],), 0, len(x))
            value, gradient = jax.value_and_grad(loss)(layers, x[index], g[index])
            updates, state = optimizer.update(gradient, state, layers)
            return (optax.apply_updates(layers, updates), state), value

        return jax.lax.scan(step, carry, jax.random.split(key, spec["interval"]))[0]

    key = jax.random.key(seed + 1)
    best, kept, selected = np.inf, trainable, 0
    for number in range(spec["steps"] // spec["interval"]):
        key, subkey = jax.random.split(key)
        trainable, state = chunk((trainable, state), subkey)
        score = relative_error(
            dict(params, layers=trainable), positions[test], forces[test]
        )
        if not np.isfinite(score):
            raise ValueError("The force fit became nonfinite")
        if score < best:
            best, kept, selected = score, trainable, (number + 1) * spec["interval"]
    checks = spec["steps"] // spec["interval"]
    return dict(params, layers=kept), {
        "corpus": int(len(positions)),
        "selected_step": selected,
        "holdout_error": float(best),
        "evaluations": float(3 * spec["steps"] * spec["batch"] + checks * held),
        "parameters": int(sum(w.size + b.size for w, b in kept)),
    }


def flops(function, *args):
    """XLA's count of floating-point operations for one call."""
    return float(jax.jit(function).lower(*args).compile().cost_analysis()["flops"])


def mlp_flops(dimension, width):
    params = {
        "shift_in": jnp.zeros(dimension),
        "scale_in": jnp.ones(dimension),
        "shift_out": jnp.zeros(dimension),
        "scale_out": jnp.ones(dimension),
        "layers": initialize(jax.random.key(0), dimension, width),
    }
    return flops(partial(apply, params), jnp.zeros(dimension))


def collapsed_bnn_potential(observations=None, width=8):
    """The collapsed tanh BNN's potential, with its data repeated to ``observations``."""
    from dataclasses import dataclass

    from bnn.target.collapsed_bnn import Target

    @dataclass(frozen=True)
    class Sized(Target):
        observations: int = 32

        def data(self):
            x, y, test = super().data()
            repeats = -(-self.observations // len(x))
            return (
                np.tile(x, (repeats, 1))[: self.observations],
                np.tile(y, repeats)[: self.observations],
                test,
            )

    base = Target("bnn", width=width)
    target = (
        base if observations is None else Sized("bnn", width, observations=observations)
    )
    likelihood = target.log_likelihood()

    def energy(q):
        return 0.5 * jnp.sum(q**2) - likelihood(q)

    return target, energy


def gradient_flops(energy, dimension):
    return flops(jax.value_and_grad(energy), jnp.zeros(dimension))


def whole_force_costs(widths, observations, input_dim=5):
    """Operation counts of the draft's whole-force models and of the exact force.

    Both are measured on one state; the models run as fitted, without caching
    the parts that do not depend on the state.
    """
    from .draft import NeuronForce, ShallowBNN

    bnn = ShallowBNN(input_dim)
    rows = []
    for width in widths:
        for count in observations:
            q = jnp.zeros((1, width, input_dim + 1))
            x, y = jnp.ones((count, input_dim)), jnp.ones(count)
            exact = flops(bnn.gradient, q, x, y)
            row = {"width": width, "observations": count, "gradient_flops": exact}
            for name, features in (("parameters", False), ("features", True)):
                model = NeuronForce(bnn, features=features)
                variables = model.init(jax.random.key(0), q, x, y)
                row[f"{name}_flops"] = flops(partial(model.apply, variables), q, x, y)
            rows.append(row)
    return rows


def student_costs(observations, architecture=(5, 4, 3, 1)):
    """Operation counts of the draft's force with its local student, and without."""
    from .draft import force, initialize_student, local_operation, parameter_count

    student = initialize_student(jax.random.key(0), dtype=jnp.float64)
    q = jnp.zeros(parameter_count(architecture))
    rows = []
    for count in observations:
        x, y = jnp.ones((count, architecture[0])), jnp.ones(count)
        exact = partial(force, x=x, y=y, architecture=architecture)
        learned = partial(
            force,
            x=x,
            y=y,
            architecture=architecture,
            local_operation=partial(local_operation, student),
        )
        rows.append(
            {
                "observations": count,
                "gradient_flops": flops(exact, q),
                "student_flops": flops(learned, q),
            }
        )
    return rows
