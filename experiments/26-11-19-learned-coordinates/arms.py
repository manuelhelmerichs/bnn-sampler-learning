"""Every arm of the comparison on one problem: several targets of one kind.

A problem holds, for ``count`` confirmation targets and one screen target:

- ``energy(u, data)`` and ``estimand(u, data)`` of one position; ``data`` and
  ``screen_data`` are what a target's chains share,
- ``starts`` ``(count, chains, D)`` and ``screen`` ``(screen chains, D)``: draws
  of each target,
- ``constant``: per name, the factors ``W`` ``(count, D, D)`` of exact constant
  metrics, with ``W W^T`` the inverse metric, the screen target's and the
  initial step size,
- ``stiff`` and ``unit``: initial step sizes without and with a good metric,
- ``oracle``: ``(forward, inverse)`` of the exact Gaussianizing map, or None,
- ``draw(key, data, count)``: exact draws of a target, for the family's bound.
"""

from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from . import flows, kernels
from .chains import listed, run_adapted, run_fixed

# Eigenvalues of a mean Hessian below this share of the largest are raised to it.
FLOOR = 1e-10


def covariance_factor(draws):
    """Cholesky factor of the covariance of ``(target, draw, D)``, shrunk as Stan does."""
    count, d = draws.shape[1:]
    centered = draws - draws.mean(1, keepdims=True)
    covariance = np.einsum("tni,tnj->tij", centered, centered) / (count - 1)
    shrunk = count / (count + 5) * covariance + 1e-3 * 5 / (count + 5) * np.eye(d)
    return np.linalg.cholesky(shrunk)


def curvature_factor(hessians):
    """Inverse root of the mean of ``(target, state, D, D)`` Hessians, made positive."""
    values, vectors = np.linalg.eigh(hessians.mean(1))
    values = np.abs(values)
    values = np.maximum(values, FLOOR * values.max(-1, keepdims=True))
    return vectors / np.sqrt(values)[:, None, :]


def row(tree, index):
    return jax.tree.map(lambda leaf: leaf[index], tree)


def compare(spec, seeds, problem):
    """Runs by ``(arm, sampler)``, and what the fits of the flows reached."""
    energy, estimand = problem["energy"], problem["estimand"]
    data, screen_data = problem["data"], problem["screen_data"]
    starts, screen = problem["starts"], problem["screen"]
    count, chains, d = starts.shape
    targets = range(count)

    def rng(role, *index):
        return np.random.default_rng([seeds[role], *index])

    def key(role, *index):
        return jax.random.key(int(rng(role, *index).integers(2**32)))

    def keys(*index):
        return {"screen": key("screen", *index), "chain": key("chain", *index)}

    def scaled(factors, screen_factor, initial):
        """A constant metric as an arm: the chain runs ``y`` with ``u = W y``."""
        solve = jax.vmap(jnp.linalg.solve, in_axes=(None, 0))
        return {
            "screen": solve(screen_factor, screen),
            "starts": jnp.stack([solve(factors[t], starts[t]) for t in targets]),
            "screen_group": {"factor": screen_factor, "data": screen_data},
            "target_groups": [
                {"factor": factors[t], "data": row(data, t)} for t in targets
            ],
            "initial": initial,
        }

    eye = jnp.eye(d)
    fixed = {"identity": scaled(jnp.stack([eye] * count), eye, problem["stiff"])} | {
        name: scaled(*entry) for name, entry in problem["constant"].items()
    }
    hessian = jax.jit(jax.vmap(jax.vmap(jax.hessian(energy), in_axes=(0, None))))

    def curvature(draws):
        recorded = draws.shape[2]
        picks = np.linspace(recorded - 1, 0, spec["hessians"] // chains).astype(int)
        states = jnp.asarray(draws[:, :, picks].reshape(count, -1, d))
        return curvature_factor(np.asarray(hessian(states, data))), len(picks) * chains

    def covariance(draws):
        return covariance_factor(draws.reshape(count, -1, d)), 0

    # Adapted arms take the target acceptance screened for their exact limit.
    adapted = {
        "covariance": (covariance, "covariance_exact"),
        "curvature": (curvature, "curvature_exact"),
    }
    forward, inverse = problem["oracle"] or (None, None)
    runs = {}
    for length in spec["lengths"]:
        sampler = f"hmc{length}"
        metric = length in spec["metric_lengths"]
        kernel = kernels.hmc(
            lambda y, group: energy(group["factor"] @ y, group["data"]),
            lambda y, group: estimand(group["factor"] @ y, group["data"]),
            length,
            spec["jitter"],
        )
        names = list(fixed) if metric else ["identity"]
        records = run_fixed(
            kernel, [fixed[name] for name in names], length, spec, keys(length)
        )
        runs |= {(name, sampler): r for name, r in zip(names, records, strict=True)}
        if not metric:
            continue
        for a, (name, (estimate, limit)) in enumerate(adapted.items()):
            if (limit, sampler) not in runs:
                continue
            runs[name, sampler] = run_adapted(
                kernel,
                estimate,
                {"data": data},
                starts,
                length,
                runs[limit, sampler]["target_acceptance"],
                problem["stiff"],
                spec,
                key("adapt", length, a),
            )
        if problem["oracle"] is None:
            continue
        kernel = kernels.hmc(
            lambda y, group: 0.5 * y @ y,
            lambda y, group: estimand(inverse(y), group["data"]),
            length,
            spec["jitter"],
        )
        arm = {
            "screen": jax.vmap(forward)(screen),
            "starts": jax.vmap(jax.vmap(forward))(starts),
            "screen_group": {"data": screen_data},
            "target_groups": [{"data": row(data, t)} for t in targets],
            "initial": problem["unit"],
        }
        (runs["transport", sampler],) = run_fixed(
            kernel, [arm], length, spec, keys(length, 1)
        )

    # The flows: one per target and the screen target, which comes first.
    every = jax.tree.map(
        lambda one, rest: jnp.concatenate([one[None], rest]), screen_data, data
    )
    fit, fitted, report = spec["fit"], {}, {}
    for f, mixing in enumerate(spec["mixings"]):
        initial = jax.vmap(
            lambda k, mixing=mixing: flows.create(
                k, d, fit["layers"], fit["hidden"], mixing
            )
        )(jax.random.split(key("flow", f), count + 1))
        settings = {
            "snapshots": fit["snapshots"],
            "batch": fit["batch"],
            "rate": fit["rate"],
        }
        start = perf_counter()
        reverse, loss = flows.fit_reverse(
            energy, initial, every, key("flow", f, 1), **settings
        )
        report[f"flow:{mixing}"] = {
            "seconds": perf_counter() - start,
            "final_loss": listed(np.asarray(loss[:, -50:].mean(1))),
            # Median over targets of the loss in 32 windows of equal length.
            "loss_trace": listed(
                np.median(
                    np.asarray(loss)[:, : loss.shape[1] // 32 * 32]
                    .reshape(loss.shape[0], 32, -1)
                    .mean(2),
                    0,
                )
            ),
        }
        for steps, flow in reverse.items():
            # The share of the fit's gradients that one chain carries.
            fitted[f"flow:{mixing}:{steps}"] = flow, steps * fit["batch"] / chains
        if problem["draw"] is not None and spec.get("exact_fit", True):
            forward_fit, _ = flows.fit_forward(
                problem["draw"],
                initial,
                every,
                key("flow", f, 2),
                **settings | {"snapshots": fit["snapshots"][-1:]},
            )
            fitted[f"exact:{mixing}"] = forward_fit[fit["snapshots"][-1]], 0.0
    for a, (name, (flow, _)) in enumerate(fitted.items()):
        divergence, effective = flows.quality(
            energy, flow, every, key("flow", 100 + a), fit["quality_draws"]
        )
        report[name] = report.get(name, {}) | {
            "divergence": listed(np.asarray(divergence)),
            "effective_share": listed(np.asarray(effective)),
        }

    def pulled(z, group):
        x, logdet = flows.forward(group["flow"], z)
        return energy(x, group["data"]) - logdet

    def enter(flow, positions):
        return jax.vmap(lambda x: flows.inverse(flow, x)[0])(positions)

    narrow = spec | {"acceptances": spec["flow_acceptances"]}
    arms = [
        {
            "screen": enter(row(flow, 0), screen[: spec["flow_screen_chains"]]),
            "starts": jnp.stack([enter(row(flow, t + 1), starts[t]) for t in targets]),
            "screen_group": {"flow": row(flow, 0), "data": screen_data},
            "target_groups": [
                {"flow": row(flow, t + 1), "data": row(data, t)} for t in targets
            ],
            "initial": problem["unit"],
        }
        for flow, _ in fitted.values()
    ]
    for length in spec["flow_lengths"] if arms else []:
        kernel = kernels.hmc(
            pulled,
            lambda z, group: estimand(
                flows.forward(group["flow"], z)[0], group["data"]
            ),
            length,
            spec["jitter"],
        )
        records = run_fixed(kernel, arms, length, narrow, keys(length, 2))
        for (name, (_, share)), record in zip(fitted.items(), records, strict=True):
            runs[name, f"hmc{length}"] = record | {"fit": share}
    return runs, report
