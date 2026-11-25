"""Training kinetic fields on the evaluation loss by antithetic evolution strategies.

On the mixtures the loss of a trial is its clipped squared error of the mode
mass plus ``weight`` times the mean squared error of the second moments off the
axis. Both members of an antithetic pair see the same tasks and random numbers.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.flatten_util import ravel_pytree

from . import dynamics


def initial(form, hyper, key):
    """Parameters of a learned form that start at, or near, SGHMC with ``hyper``."""
    parameters = {"hyper": jnp.log(jnp.asarray(hyper, jnp.float32))}
    if form == "classical":
        # Only step size, friction and temperature are trained.
        return parameters
    if form == "residual":
        return parameters | {"net": dynamics.network(key)}
    if form != "published":
        raise ValueError(f"No learned form: {form}")
    # Without a term that passes the momentum through, the network itself has to
    # return it: fit ``beta`` to the normalized momentum and ``alpha`` to zero.
    k1, k2 = jax.random.split(key)
    net = dynamics.network(k1)
    net["w3"] = 0.1 * jax.random.normal(k2, net["w3"].shape)
    optimizer = optax.adam(3e-3)

    def loss(net, inputs):
        out = dynamics.apply(net, inputs)
        return jnp.mean(out[..., 0] ** 2 + (out[..., 1] - inputs[..., 2]) ** 2)

    @jax.jit
    def fit(net, state, key):
        inputs = jax.random.normal(key, (1024, dynamics.FEATURES))
        updates, state = optimizer.update(jax.grad(loss)(net, inputs), state)
        return optax.apply_updates(net, updates), state

    state = optimizer.init(net)
    for subkey in jax.random.split(k2, 2000):
        net, state = fit(net, state, subkey)
    scales = jnp.log(jnp.asarray([1.0, np.sqrt(hyper[2])], jnp.float32))
    return parameters | {"net": net, "scales": scales}


def trials(form, *, chains, steps, burn):
    """``(parameters, tasks, keys) -> rows``, one trial per task."""

    def rows(parameters, tasks, keys):
        return jax.vmap(
            lambda task, key: dynamics.trial(
                dynamics.euler(form, parameters),
                task,
                key,
                chains=chains,
                steps=steps,
                burn=burn,
            )
        )(tasks, keys)

    return rows


def train(form, parameters, draw_tasks, validation, spec, key, report=None):
    """Train a form on the mixtures' evaluation loss; see ``evolve``.

    ``draw_tasks(iteration)`` gives the tasks of one iteration with their exact
    mode mass under ``mass``.
    """
    rows = trials(form, chains=spec["chains"], steps=spec["steps"], burn=spec["burn"])

    def loss(parameters, tasks, key):
        out = rows(parameters, tasks, jax.random.split(key, len(tasks["mass"])))
        mass, local = dynamics.errors(out, tasks["mass"])
        return (
            jnp.mean(mass + spec["weight"] * local),
            jnp.sqrt(jnp.mean(mass)),
            jnp.sqrt(jnp.mean(local)),
        )

    return evolve(
        loss,
        parameters,
        draw_tasks,
        validation,
        spec,
        key,
        report,
        ("mass_rmse", "local_rmse"),
    )


def evolve(
    loss,
    parameters,
    draw_tasks,
    validation,
    spec,
    key,
    report=None,
    names=("first", "second"),
):
    """Evolution strategies from ``parameters``; returns the best on validation.

    ``loss(parameters, tasks, key)`` returns the loss of one batch of tasks and
    two readings of it, which the history records under ``names``.
    ``validation`` is a fixed batch, read every ``interval`` iterations.
    """
    flat, unravel = ravel_pytree(parameters)
    sigma, pairs = spec["sigma"], spec["pairs"]

    @jax.jit
    def gradient(vector, tasks, key):
        k1, k2 = jax.random.split(key)
        noise = jax.random.normal(k1, (pairs, len(vector)), vector.dtype)

        def pair(eta):
            plus = loss(unravel(vector + sigma * eta), tasks, k2)[0]
            minus = loss(unravel(vector - sigma * eta), tasks, k2)[0]
            return plus - minus, 0.5 * (plus + minus)

        difference, mean = jax.lax.map(pair, noise)
        return (difference[:, None] * noise).mean(0) / (2 * sigma), mean.mean()

    validate = jax.jit(lambda vector, tasks, key: loss(unravel(vector), tasks, key))
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(spec["rate"]))
    state = optimizer.init(flat)
    k_validation, k_train = jax.random.split(key)
    best, history = (np.inf, flat), []
    for iteration in range(spec["iterations"] + 1):
        if iteration % spec["interval"] == 0 or iteration == spec["iterations"]:
            total, first, second = (
                float(v) for v in validate(flat, validation, k_validation)
            )
            if total < best[0]:
                best = (total, flat)
            history.append(
                {
                    "iteration": iteration,
                    "validation": total,
                    names[0]: first,
                    names[1]: second,
                    "hyper": [float(v) for v in np.exp(unravel(flat)["hyper"])],
                }
            )
            if report:
                report(history[-1])
        if iteration == spec["iterations"]:
            break
        update, mean = gradient(
            flat, draw_tasks(iteration), jax.random.fold_in(k_train, iteration)
        )
        updates, state = optimizer.update(update, state)
        flat = optax.apply_updates(flat, updates)
        history[-1].setdefault("train", float(mean))
    return unravel(best[1]), history


def distill(form, parameters, tasks, key, *, chains, steps, every):
    """Least-squares linear fields of the raw features that a learned form visits.

    Returns parameters of the ``linear`` form and the share of the variance of
    each field's learned part that the linear map explains.
    """
    start, step = dynamics.euler(form, parameters)

    def record(task, key):
        k1, k2, k3 = jax.random.split(key, 3)
        d = dynamics.targets.dimension(task)
        z = jax.random.normal(k1, (chains, d), task["location"].dtype)
        state = start(k2, dynamics.targets.embedded(z, task), task)

        def one(state, key):
            new = step(key, state, task)
            theta, r, _ = state
            g = dynamics.targets.force(theta, task)
            # The averages the fields saw when they moved ``state`` on.
            averages = new[2]
            alpha, beta = dynamics.fields(form, parameters, theta, r, g, averages)
            inputs = dynamics.features(theta, r, g, averages, False)
            return new, (inputs, jnp.stack([alpha, beta - r], axis=-1))

        _, (inputs, outputs) = jax.lax.scan(one, state, jax.random.split(k3, steps))
        return inputs[every - 1 :: every], outputs[every - 1 :: every]

    inputs, outputs = jax.jit(jax.vmap(record))(
        tasks, jax.random.split(key, len(tasks["mass"]))
    )
    x = np.asarray(inputs, np.float64).reshape(-1, dynamics.FEATURES)
    y = np.asarray(outputs, np.float64).reshape(-1, 2)
    keep = np.all(np.isfinite(x), axis=1) & np.all(np.isfinite(y), axis=1)
    x, y = x[keep], y[keep]
    design = np.concatenate([x, np.ones((len(x), 1))], axis=1)
    solution = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - design @ solution
    explained = 1 - residual.var(axis=0) / np.maximum(y.var(axis=0), 1e-30)
    linear = {
        "hyper": parameters["hyper"],
        "weight": jnp.asarray(solution[:-1], jnp.float32),
        "bias": jnp.asarray(solution[-1], jnp.float32),
    }
    return linear, [float(v) for v in explained]
