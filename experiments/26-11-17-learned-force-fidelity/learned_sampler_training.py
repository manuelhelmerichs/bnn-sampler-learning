"""Explicit fitting recipes for the learned-sampler reproduction experiment."""

from __future__ import annotations

import time
from functools import partial
from numbers import Integral

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .core import (
    derivative_gate,
    force,
    force_from_sensitivities,
    parameter_count,
    sensitivities,
    unpack,
)
from .student import (
    initialize_student,
    local_operation,
    pairwise,
)


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _target_parameters(sigma=0.5, tau=1.0, gate_width=1e-4):
    if not np.isfinite([sigma, tau, gate_width]).all():
        raise ValueError("Target scales and gate width must be finite")
    if sigma <= 0 or tau <= 0 or gate_width < 0:
        raise ValueError("Target scales must be positive and gate width nonnegative")
    return {"sigma": sigma, "tau": tau, "gate_width": gate_width}


def _validate_task(task, splits, *, responses=False):
    arch = tuple(task["architecture"])
    if len(arch) < 3 or any(
        isinstance(width, (bool, np.bool_)) or not isinstance(width, Integral)
        for width in arch
    ):
        raise ValueError("A learner task needs integer widths and a hidden layer")
    dimensions = parameter_count(arch)
    x = np.asarray(task["x"])
    if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] != arch[0]:
        raise ValueError("Task x must contain observations matching the input width")
    if not np.isfinite(x).all():
        raise ValueError("Task x must be finite")
    if responses:
        y = np.asarray(task["y"])
        if y.shape not in ((len(x),), (len(x), 1)) or not np.isfinite(y).all():
            raise ValueError("Task y must be finite and match the observation axis")
    for split in splits:
        q = np.asarray(task[f"{split}_q"])
        if q.ndim != 2 or q.shape[0] < 1 or q.shape[1] != dimensions:
            raise ValueError(
                f"Task {split}_q must contain states matching the architecture"
            )
        if not np.isfinite(q).all():
            raise ValueError(f"Task {split}_q must be finite")


def _query_pools(queries):
    pools = []
    for part in queries:
        arrays = {
            key: np.asarray(part[key])
            for key in ("weights", "downstream", "mask", "target")
        }
        a, b, mask, target = (
            arrays[key] for key in ("weights", "downstream", "mask", "target")
        )
        if (
            a.ndim != 2
            or min(a.shape) < 1
            or b.shape != a.shape
            or mask.shape != a.shape
        ):
            raise ValueError(
                "Query inputs and masks need a shared nonempty (queries, arity) shape"
            )
        if mask.dtype.kind != "b" or np.any(np.sum(mask, axis=-1) == 0):
            raise ValueError(
                "Query masks must be Boolean and retain at least one neighbor"
            )
        if target.shape != (len(a),) or not all(
            np.isfinite(value).all() for value in (a, b, target)
        ):
            raise ValueError(
                "Query targets must match the query axis and all values must be finite"
            )
        if np.any(np.abs(target) > 1 + 1e-6):
            raise ValueError("Normalized query targets must lie in [-1, 1]")
        pools.append({key: jnp.asarray(value) for key, value in arrays.items()})
    if not pools:
        raise ValueError("At least one nonempty query pool is required")
    return tuple(pools)


def optimizer(updates):
    _positive_integer(updates, "updates")
    return optax.chain(
        optax.clip_by_global_norm(5.0),
        optax.adamw(
            optax.cosine_decay_schedule(1e-3, updates, alpha=0.05),
            weight_decay=1e-6,
        ),
    )


def save_student(path, params):
    np.savez_compressed(
        path,
        **{f"w{i}": np.asarray(w) for i, (w, _) in enumerate(params)},
        **{f"b{i}": np.asarray(b) for i, (_, b) in enumerate(params)},
    )


def load_student(path, *, dtype=jnp.float32):
    with np.load(path) as data:
        return tuple(
            (jnp.asarray(data[f"w{i}"], dtype), jnp.asarray(data[f"b{i}"], dtype))
            for i in range(3)
        )


def make_queries(
    task, *, split="train", count=None, seed=0, max_arity=None, gate_width=1e-4
):
    """Local pre-gate sum labels from disjoint source trajectories, with arity masks."""
    _validate_task(task, (split,))
    _target_parameters(gate_width=gate_width)
    q = task[f"{split}_q"]
    arch = tuple(task["architecture"])
    required_arity = max(arch[2:])
    max_arity = required_arity if max_arity is None else max_arity
    _positive_integer(max_arity, "max_arity")
    if max_arity < required_arity:
        raise ValueError("max_arity cannot truncate a task's outgoing neighbors")
    count = min(len(q) * 8, 32768) if count is None else count
    _positive_integer(count, "count")
    x = jnp.asarray(task["x"])
    states = jnp.asarray(q)
    labels = jax.jit(jax.vmap(lambda state: sensitivities(state, x, arch, gate_width)))(
        states
    )
    layers = jax.vmap(lambda state: unpack(state, arch))(states)
    activation = jnp.broadcast_to(x, (len(q), *x.shape))
    gates = []
    for weight, bias in layers[:-1]:
        pre = jnp.einsum("bni,bio->bno", activation, weight) + bias[:, None, :]
        gates.append(derivative_gate(pre, gate_width))
        activation = jax.nn.relu(pre)
    rng = np.random.default_rng(seed)
    parts = []
    for layer in range(len(arch) - 2):
        state_id = rng.integers(len(q), size=count)
        obs_id = rng.integers(len(x), size=count)
        node_id = rng.integers(arch[layer + 1], size=count)
        weights = np.asarray(layers[layer + 1][0])[state_id, node_id, :]
        if layer == len(arch) - 3:
            downstream = np.ones_like(weights)
        else:
            downstream = np.asarray(labels[layer + 1] * gates[layer + 1])[
                state_id, obs_id, :
            ]
        cw = np.maximum(np.linalg.norm(weights, axis=-1, keepdims=True), 1e-8)
        cu = np.maximum(np.linalg.norm(downstream, axis=-1, keepdims=True), 1e-8)
        a, b = weights / cw, downstream / cu
        parts.append(
            {
                "weights": np.pad(a, ((0, 0), (0, max_arity - a.shape[1]))).astype(
                    np.float32
                ),
                "downstream": np.pad(b, ((0, 0), (0, max_arity - b.shape[1]))).astype(
                    np.float32
                ),
                "mask": np.broadcast_to(
                    np.arange(max_arity) < a.shape[1], (count, max_arity)
                ),
                "target": np.sum(a * b, axis=-1).astype(np.float32),
                "boundary": layer == len(arch) - 3,
            }
        )
    return parts


def _query_prediction(params, pool, *, odd):
    return jnp.sum(
        pairwise(params, pool["weights"], pool["downstream"], odd=odd) * pool["mask"],
        axis=-1,
    )


def query_loss(params, pool, *, odd=False, objective="bounded"):
    """Scalar supervision; unbounded loss retains gradients outside the inference clip.

    For targets in [-1, 1], the unbounded squared error upper-bounds the
    inference error. It avoids the zero-gradient region of clipped predictions.
    """
    if objective not in ("bounded", "unbounded"):
        raise ValueError("Query objective must be 'bounded' or 'unbounded'")
    prediction = _query_prediction(params, pool, odd=odd)
    if objective == "bounded":
        prediction = jnp.clip(prediction, -1, 1)
    return jnp.mean(jnp.square(prediction - pool["target"]))


def fit_queries(
    queries, *, seed, updates=12000, odd=False, batch_size=128, objective="bounded"
):
    """Matched scalar-sum supervision with frozen final-checkpoint selection."""
    _positive_integer(batch_size, "batch_size")
    pools = _query_pools(queries)
    if objective not in ("bounded", "unbounded"):
        raise ValueError("Query objective must be 'bounded' or 'unbounded'")
    params = initialize_student(jax.random.key(seed))
    tx = optimizer(updates)
    state = tx.init(params)

    @jax.jit
    def update(p, opt_state, key, pool):
        index = jax.random.randint(key, (batch_size,), 0, len(pool["target"]))
        value, gradient = jax.value_and_grad(query_loss)(
            p,
            {name: value[index] for name, value in pool.items()},
            odd=odd,
            objective=objective,
        )
        changes, opt_state = tx.update(gradient, opt_state, p)
        return optax.apply_updates(p, changes), opt_state, value

    start = time.perf_counter()
    last_loss = None
    for step in range(updates):
        key = jax.random.fold_in(jax.random.key(seed + 10_000_000), step)
        params, state, last_loss = update(params, state, key, pools[step % len(pools)])
        if (step + 1) % 2000 == 0:
            print(
                f"query fit {seed} odd={odd}: {step + 1}/{updates}, loss={float(last_loss):.3g}",
                flush=True,
            )
    jax.block_until_ready(params)
    if not np.isfinite(last_loss) or not all(
        np.isfinite(value).all() for value in jax.tree.leaves(params)
    ):
        raise FloatingPointError(
            "Query training produced a nonfinite loss or parameter"
        )
    return params, {
        "seconds": time.perf_counter() - start,
        "final_loss": float(last_loss),
        "updates": updates,
        "batch_size": batch_size,
        "odd": odd,
        "objective": objective,
        "selection": "final",
    }


def generic_queries(*, seed=9302026, count=32768, max_arity=7, arities=None):
    """No BNN states: independently sampled arithmetic with varied arity/sign."""
    _positive_integer(count, "count")
    _positive_integer(max_arity, "max_arity")
    if arities is not None:
        arities = tuple(arities)
        if not arities or len(set(arities)) != len(arities):
            raise ValueError("arities must be nonempty and contain no duplicates")
        for arity in arities:
            _positive_integer(arity, "arity")
        if max(arities) > max_arity:
            raise ValueError("arities cannot exceed max_arity")
    rng = np.random.default_rng(seed)
    a, b = rng.normal(size=(2, count, max_arity))
    selected = (
        rng.integers(1, max_arity + 1, size=(count, 1))
        if arities is None
        else rng.choice(arities, size=(count, 1))
    )
    mask = np.arange(max_arity)[None, :] < selected
    a, b = a * mask, b * mask
    a /= np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-8)
    b /= np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-8)
    return [
        {
            "weights": a.astype(np.float32),
            "downstream": b.astype(np.float32),
            "mask": mask,
            "target": np.sum(a * b, axis=-1).astype(np.float32),
        }
    ]


def query_mse(params, queries, *, odd=False):
    """Mean bounded MSE with equal weight per independently specified query pool."""
    return float(
        np.mean(
            [float(query_loss(params, pool, odd=odd)) for pool in _query_pools(queries)]
        )
    )


def query_diagnostics(params, queries, *, odd=False):
    """Report failures hidden by clipping, with separate scores for each arity."""
    records = []
    for pool in _query_pools(queries):
        raw = np.asarray(_query_prediction(params, pool, odd=odd))
        target = np.asarray(pool["target"])
        arity = np.asarray(pool["mask"]).sum(axis=-1)
        records.append((raw, target, arity))
    raw, target, arity = (np.concatenate(parts) for parts in zip(*records, strict=True))

    def summarize(selected):
        prediction, labels = raw[selected], target[selected]
        return {
            "queries": int(selected.sum()),
            "bounded_mse": float(np.mean((np.clip(prediction, -1, 1) - labels) ** 2)),
            "unbounded_mse": float(np.mean((prediction - labels) ** 2)),
            "saturation_fraction": float(np.mean(np.abs(prediction) > 1)),
        }

    return {
        "all": summarize(np.ones(len(raw), bool)),
        "by_arity": {
            str(width): summarize(arity == width) for width in np.unique(arity)
        },
    }


def _training_schedule(step, updates, task_count):
    # With an even number of round-robin tasks, global parity locks some tasks
    # out of full-force or unforced updates. Schedule by visits in that case.
    phase_step = step if task_count % 2 else step // task_count
    teacher_forcing = step < updates // 4 or (
        step < 3 * updates // 4 and phase_step % 2 == 0
    )
    return phase_step % 4 == 0, teacher_forcing


def fit_primary(
    tasks,
    *,
    seed,
    updates=12000,
    batch_size=32,
    odd=False,
    observation_batch_size=16,
    validation_interval=2000,
    validation_count=256,
    sigma=0.5,
    tau=1.0,
    gate_width=1e-4,
):
    """Appendix O sensitivity/force loss, source-only early-state validation."""
    tasks = tuple(tasks)
    if not tasks:
        raise ValueError("At least one learner task is required")
    for name, value in (
        ("updates", updates),
        ("batch_size", batch_size),
        ("observation_batch_size", observation_batch_size),
        ("validation_interval", validation_interval),
        ("validation_count", validation_count),
    ):
        _positive_integer(value, name)
    target_parameters = _target_parameters(sigma, tau, gate_width)
    for task in tasks:
        _validate_task(task, ("train", "validation"), responses=True)
    params = initialize_student(jax.random.key(seed))
    tx = optimizer(updates)
    state = tx.init(params)
    prepared = []
    for task in tasks:
        arch, x, y = (
            tuple(task["architecture"]),
            jnp.asarray(task["x"]),
            jnp.asarray(task["y"]),
        )
        q = jnp.asarray(task["train_q"])
        target_s = jax.jit(
            jax.vmap(
                partial(sensitivities, x=x, architecture=arch, gate_width=gate_width)
            )
        )(q)
        target_g = jax.jit(
            jax.vmap(partial(force, x=x, y=y, architecture=arch, **target_parameters))
        )(q)
        s_scales = tuple(
            jnp.maximum(jnp.sqrt(jnp.mean(s**2)), 0.1).astype(jnp.float32)
            for s in target_s
        )
        g_scale = jnp.maximum(jnp.std(target_g - q / tau**2, axis=0), 0.1).astype(
            jnp.float32
        )
        if not all(
            np.isfinite(value).all()
            for value in jax.tree.leaves((target_s, target_g, s_scales, g_scale))
        ):
            raise FloatingPointError("Teacher labels or training scales are nonfinite")
        prepared.append((task, target_s, s_scales, g_scale))

    def build(task, true_s, s_scales, g_scale):
        arch = tuple(task["architecture"])
        x, y, q = (
            jnp.asarray(task[key], dtype=jnp.float32) for key in ("x", "y", "train_q")
        )
        true_s = jax.tree.map(lambda a: a.astype(jnp.float32), true_s)

        @partial(jax.jit, static_argnames=("full", "teacher_forcing"))
        def step_fn(p, opt_state, key, *, full, teacher_forcing):
            k1, k2 = jax.random.split(key)
            index = jax.random.randint(k1, (batch_size,), 0, len(q))
            obs = (
                jnp.arange(len(x))
                if full
                else jax.random.choice(
                    k2, len(x), (min(observation_batch_size, len(x)),), replace=False
                )
            )
            bx, by = x[obs], y[obs]
            labels = tuple(s[index][:, obs] for s in true_s)

            def loss(p):
                def operation(w, u):
                    return local_operation(p, w, u, odd=odd)

                def one(z, target):
                    pred = sensitivities(
                        z,
                        bx,
                        arch,
                        gate_width=gate_width,
                        local_operation=operation,
                        teacher_sensitivities=target if teacher_forcing else None,
                    )
                    value = sum(
                        jnp.mean(((a - b) / scale) ** 2)
                        for a, b, scale in zip(pred, target, s_scales, strict=True)
                    ) / len(pred)
                    if full:
                        gf = force_from_sensitivities(
                            z, bx, by, arch, pred, **target_parameters
                        )
                        gt = force(z, bx, by, arch, **target_parameters)
                        value += 0.2 * jnp.mean(((gf - gt) / g_scale) ** 2)
                    return value

                return jnp.mean(jax.vmap(one)(q[index], labels))

            value, gradient = jax.value_and_grad(loss)(p)
            changes, opt_state = tx.update(gradient, opt_state, p)
            return optax.apply_updates(p, changes), opt_state, value

        vq = jnp.asarray(task["validation_q"][:validation_count], dtype=jnp.float32)
        vg = jax.jit(jax.vmap(lambda z: force(z, x, y, arch, **target_parameters)))(vq)

        @jax.jit
        def validation(p):
            def operation(w, u):
                return local_operation(p, w, u, odd=odd)

            candidate = jax.vmap(
                lambda z: force(
                    z, x, y, arch, local_operation=operation, **target_parameters
                )
            )(vq)
            return jnp.mean(((candidate - vg) / g_scale) ** 2)

        return step_fn, validation

    functions = [build(*values) for values in prepared]
    best_params, best_error = params, float("inf")
    history = []
    start = time.perf_counter()
    for step in range(updates):
        update, _ = functions[step % len(tasks)]
        full, teacher_forcing = _training_schedule(step, updates, len(tasks))
        params, state, value = update(
            params,
            state,
            jax.random.fold_in(jax.random.key(seed + 20_000_000), step),
            full=full,
            teacher_forcing=teacher_forcing,
        )
        if (step + 1) % validation_interval == 0 or step + 1 == updates:
            task_errors = [float(validate(params)) for _, validate in functions]
            error = float(np.mean(task_errors))
            if not np.isfinite(task_errors).all() or not np.isfinite(value):
                raise FloatingPointError(
                    "Primary training produced a nonfinite loss or validation error"
                )
            history.append(
                {
                    "step": step + 1,
                    "validation": error,
                    "validation_by_task": task_errors,
                    "training_loss": float(value),
                }
            )
            if error < best_error:
                best_error, best_params = (
                    error,
                    jax.tree.map(lambda a: jnp.array(a), params),
                )
            print(
                f"primary fit {seed}: {step + 1}/{updates}, validation={error:.3g}, elapsed={time.perf_counter() - start:.1f}s",
                flush=True,
            )
    return best_params, {
        "seconds": time.perf_counter() - start,
        "updates": updates,
        "batch_size": batch_size,
        "odd": odd,
        "observation_batch_size": observation_batch_size,
        "validation_interval": validation_interval,
        "validation_count": validation_count,
        "target_parameters": target_parameters,
        "selection": "source early-state validation",
        "validation_error": best_error,
        "history": history,
    }
