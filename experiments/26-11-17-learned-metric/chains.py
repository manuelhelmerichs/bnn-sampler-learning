"""Batches of chains, their costs and the reduction of one cell."""

from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from bnn.evaluation.cost_error import (
    break_even,
    cost_at,
    ladder,
    nested_resamples,
    rmse,
)
from runner import finite

# What an arm may use. Classical arms pay for their own adaptation. Constant
# arms have an exact constant metric, the limit of adapting one. Oracle arms have
# the exact coordinates in which the target is simple. Explicit arms compute
# their metric from derivatives at every state.
KINDS = {
    "identity": "classical",
    "diagonal": "classical",
    "covariance": "classical",
    "curvature": "classical",
    "covariance_exact": "constant",
    "curvature_exact": "constant",
    "transport": "oracle",
    "reduced": "oracle",
    "metric": "oracle",
    "metric_drift": "oracle",
    "softabs": "explicit",
    "fisher": "explicit",
    "fisher_drift": "explicit",
}
# Arms that estimate a constant metric in windows.
ADAPTED = ("diagonal", "covariance", "curvature")
# Arms with a metric evaluation per state.
LOCAL = ("metric", "metric_drift", "softabs", "fisher", "fisher_drift")
# A normalized estimand off by more than this left the target.
DIVERGED = 1e6


# ----------------------------------------------------------------------------
# Running batches of chains


def mean_length(length):
    """Mean leapfrog steps of a transition that draws 1 to ``length`` of them."""
    return (length + 1) / 2


def transitions(gradients, length):
    """Transitions that spend ``gradients`` on average."""
    return max(1, round(gradients / mean_length(length)))


def split(key, groups, chains):
    return jax.random.split(key, (groups, chains))


def stacked(groups):
    return jax.tree.map(lambda *leaves: jnp.stack(leaves), *groups)


def listed(values):
    """A list for a summary, with undefined values as None."""
    return [float(v) if np.isfinite(v) else None for v in values]


def measured(kernel, group, state, step, key, transitions):
    """Estimand averages ``(group, chain, rung, estimand)`` and the acceptance rate."""
    groups, chains = step.shape

    def block(key, state, steps):
        return kernel["run"](
            split(key, groups, chains), state, step, group, steps=steps
        )

    totals, accepted, estimates = 0.0, 0.0, []
    for rung, (_, (sums, accepts), _) in zip(
        transitions, ladder(block, state, key, transitions), strict=True
    ):
        totals = totals + np.asarray(sums)
        accepted = accepted + np.asarray(accepts)
        estimates.append(totals / rung)
    return np.stack(estimates, 2), accepted / transitions[-1]


def tuned(kernel, group, positions, initial, acceptance, key, warmup, transitions):
    """Start a batch of chains, tune their step sizes and measure them."""
    groups, chains = initial.shape
    tune_key, run_key = jax.random.split(key)
    state = kernel["init"](positions, group)
    state, step, _ = kernel["tune"](
        split(tune_key, groups, chains),
        state,
        initial,
        acceptance,
        group,
        steps=warmup,
        thin=warmup,
    )
    estimates, accepted = measured(kernel, group, state, step, run_key, transitions)
    return estimates, accepted, np.asarray(step)


def run_fixed(kernel, arms, length, spec, keys):
    """Screen each arm's target acceptance, then run it on the confirmation targets.

    An arm holds its starts in its chain's coordinates, on the screen target
    (``screen``, one row per chain) and on every confirmation target
    (``starts``, by target and chain), what its chains share (``group``, or
    ``screen_group`` and one of ``target_groups`` per confirmation target) and
    its first step size (``initial``). Returns one record per arm, with estimates
    ``(target, chain, rung, estimand)``.
    """
    start = perf_counter()
    acceptances = spec["acceptances"]
    targets, chains, _ = arms[0]["starts"].shape
    count, wide = len(acceptances), arms[0]["screen"].shape[0]
    estimates, _, _ = tuned(
        kernel,
        stacked(
            [
                arm.get("screen_group", arm.get("group"))
                for arm in arms
                for _ in acceptances
            ]
        ),
        jnp.stack([arm["screen"] for arm in arms for _ in acceptances]),
        jnp.stack(
            [jnp.full(wide, arm["initial"]) for arm in arms for _ in acceptances]
        ),
        jnp.tile(jnp.asarray(acceptances), len(arms))[:, None] * jnp.ones((1, wide)),
        keys["screen"],
        spec["warmup"],
        [transitions(spec["screen_budget"], length)],
    )
    error = rmse(estimates[:, :, -1, 0] - 1, axis=1).reshape(len(arms), count)
    error = np.where(np.isfinite(error), error, np.inf)
    chosen = [acceptances[i] for i in np.argmin(error, axis=1)]
    estimates, accepted, step = tuned(
        kernel,
        stacked(
            [
                group
                for arm in arms
                for group in arm.get("target_groups", [arm.get("group")] * targets)
            ]
        ),
        jnp.concatenate([arm["starts"] for arm in arms]),
        jnp.concatenate([jnp.full((targets, chains), arm["initial"]) for arm in arms]),
        jnp.repeat(jnp.asarray(chosen), targets)[:, None] * jnp.ones((1, chains)),
        keys["chain"],
        spec["warmup"],
        [transitions(rung, length) for rung in spec["rungs"]],
    )
    seconds = (perf_counter() - start) / len(arms)
    return [
        {
            "estimates": estimates[a * targets : (a + 1) * targets],
            "acceptance": accepted[a * targets : (a + 1) * targets],
            "step": step[a * targets : (a + 1) * targets],
            "target_acceptance": chosen[a],
            "screen_rmse": listed(error[a]),
            "seconds": seconds,
        }
        for a in range(len(arms))
    ]


def run_adapted(
    kernel, estimate, shared, starts, length, acceptance, initial, spec, key
):
    """Chains that estimate their metric in windows, pooled over a target's chains.

    Step sizes are tuned without a metric first. Every window records draws
    while tuning continues, and ends by estimating the metric from its own
    draws. ``estimate`` maps draws ``(target, chain, draw, D)`` to a factor per
    target and the number of local metrics it took. ``shared`` holds what else
    the chains of each target share, stacked over the targets.
    """
    start = perf_counter()
    targets, chains, d = starts.shape
    keys = iter(jax.random.split(key, len(spec["windows"]) + 3))

    def full(value):
        return jnp.full((targets, chains), value)

    def tune(state, group, step, steps, thin):
        return kernel["tune"](
            split(next(keys), targets, chains),
            state,
            step,
            full(acceptance),
            group,
            steps=steps,
            thin=thin,
        )

    factor = np.broadcast_to(np.eye(d), (targets, d, d))
    group = shared | {"factor": jnp.asarray(factor)}
    state = kernel["init"](starts, group)
    state, step, _ = tune(state, group, full(initial), spec["warmup"], spec["warmup"])
    per, evaluations = transitions(spec["window_gradients"], length), 0
    for window in spec["windows"]:
        state, step, recorded = tune(state, group, step, window * per, per)
        draws = np.einsum("tij,tcsj->tcsi", factor, np.asarray(recorded))
        factor, used = estimate(draws)
        evaluations += used
        group = shared | {"factor": jnp.asarray(factor)}
        last = np.linalg.solve(factor[:, None], draws[:, :, -1, :, None])[..., 0]
        state = kernel["init"](jnp.asarray(last), group)
        step = full(spec["initial_step"] * d**-0.25)
    state, step, _ = tune(state, group, step, spec["warmup"], spec["warmup"])
    estimates, accepted = measured(
        kernel,
        group,
        state,
        step,
        next(keys),
        [transitions(r, length) for r in spec["rungs"]],
    )
    return {
        "estimates": estimates,
        "acceptance": accepted,
        "step": np.asarray(step),
        "target_acceptance": acceptance,
        "evaluations": evaluations,
        "seconds": perf_counter() - start,
    }


# ----------------------------------------------------------------------------
# Costs and the reduction of one cell


def costs(spec, arm, sampler, rung, evaluations=0):
    """Gradients, Hessian-vector products and metric evaluations per chain.

    An adapted arm pays ``spec["curvature_products"]`` products for each of the
    ``evaluations`` local metrics its target took. The metric evaluations of a
    position-dependent arm are counted and not charged: their price is what the
    break-even is about.
    """
    length = 1 if sampler == "mala" else int(sampler[3:])
    mean, moves = mean_length(length), transitions(rung, length)
    if arm in ADAPTED:
        tuning = 2 * spec["warmup"] + transitions(
            spec["window_gradients"], length
        ) * sum(spec["windows"])
        gradients = 1 + len(spec["windows"]) + mean * (tuning + moves)
        products = evaluations * spec["curvature_products"] / spec["chains"]
        return float(gradients), float(products), 0.0
    gradients = 1.0 + mean * (spec["warmup"] + moves)
    evaluations = 1.0 + spec["warmup"] + moves if arm in LOCAL else 0.0
    return gradients, 0.0, evaluations


def reduce_cell(spec, runs, seed):
    """RMSE curves, and every arm against the cheapest classical and constant chain.

    The work a reference needs for an arm's RMSE follows from interpolating the
    reference's curve; the cheapest reference of a kind counts. Both estimands
    are read alike, the second under ``secondary``. Intervals resample targets,
    then chains.
    """
    rungs = spec["rungs"]
    targets, chains = spec["targets"], spec["chains"]
    ids = nested_resamples(
        np.repeat(np.arange(targets), chains),
        spec["bootstrap"],
        np.random.default_rng(seed),
    )
    errors, cost = {}, {}
    for key, run in runs.items():
        # A chain that left the target has an infinite error.
        error = np.where(
            np.abs(run["estimates"] - 1) < DIVERGED, run["estimates"] - 1, np.inf
        )
        errors[key] = error.reshape(targets * chains, *error.shape[2:])
        cost[key] = np.array(
            [costs(spec, *key, rung, run.get("evaluations", 0)) for rung in rungs]
        )
    work = {key: c[:, 0] + spec["hvp_cost"] * c[:, 1] for key, c in cost.items()}

    def cheapest(keys, error, curves):
        """Work of the cheapest reference for ``error``, and whether off its curve."""
        with np.errstate(divide="ignore", invalid="ignore"):
            pairs = [cost_at(work[key], curves[key], error) for key in keys]
        values = np.array([value for value, _ in pairs])
        off = np.array([off for _, off in pairs])
        best = np.argmin(values, axis=0)[None]
        return (
            np.take_along_axis(values, best, 0)[0],
            np.take_along_axis(off, best, 0)[0],
        )

    def read(estimand):
        """Per arm and rung, the entries of one estimand."""
        curve = {key: rmse(error[..., estimand]) for key, error in errors.items()}
        replicates = {
            key: rmse(error[ids][..., estimand], axis=1)
            for key, error in errors.items()
        }
        references = {
            kind: [
                key
                for key in runs
                if KINDS[key[0]] == kind and np.all(np.isfinite(curve[key]))
            ]
            for kind in ("classical", "constant")
        }
        out = {}
        for key in runs:
            own, evaluations = work[key], cost[key][:, 2]
            compared = {}
            for kind, keys in references.items():
                if not keys:
                    continue
                value, off = cheapest(keys, curve[key], curve)
                resampled = np.stack(
                    [
                        cheapest(
                            keys,
                            replicates[key][b],
                            {k: replicates[k][b] for k in keys},
                        )[0]
                        for b in range(len(ids))
                    ]
                )
                compared[kind] = value, off, resampled
            rows = []
            for j in range(len(rungs)):
                trial = errors[key][:, j, estimand]
                entry = {
                    "diverged": int(np.count_nonzero(~np.isfinite(trial))),
                    "rmse": float(curve[key][j]),
                    "bias": float(np.mean(trial)),
                    "bias_se": float(np.std(trial, ddof=1) / np.sqrt(len(trial))),
                }
                for kind, (value, off, resampled) in compared.items():
                    # The reference's work for this RMSE, per unit of own work.
                    entry[f"over_{kind}"] = float(value[j] / own[j])
                    entry[f"over_{kind}_interval"] = np.quantile(
                        resampled[:, j] / own[j], [0.025, 0.975]
                    ).tolist()
                    entry[f"{kind}_extrapolated"] = bool(off[j])
                if evaluations[j] and "classical" in compared:
                    value, _, resampled = compared["classical"]
                    entry["rho_star"] = float(
                        break_even(value[j], own[j], evaluations[j])
                    )
                    entry["rho_star_interval"] = np.quantile(
                        break_even(resampled[:, j], own[j], evaluations[j]),
                        [0.025, 0.975],
                    ).tolist()
                rows.append(finite(entry))
            out[key] = rows
        return out

    primary, secondary = read(0), read(1)
    curves = {}
    for arm, sampler in runs:
        key = arm, sampler
        curves.setdefault(arm, {})[sampler] = [
            {
                "budget": rung,
                "gradients": float(cost[key][j, 0]),
                "hvps": float(cost[key][j, 1]),
                "evaluations": float(cost[key][j, 2]),
                "work": float(work[key][j]),
                **primary[key][j],
                "secondary": secondary[key][j],
            }
            for j, rung in enumerate(rungs)
        ]
    return curves
