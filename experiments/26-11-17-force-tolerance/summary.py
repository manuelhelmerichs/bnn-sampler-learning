"""The reading across cells: break-even map, tolerances, learners and placement.

Every number refers to the largest rung of a cell. ``rho_star`` is the cost of
one surrogate evaluation, in gradients, below which the surrogate chain needs
less work than the cheapest exact HMC chain for the same RMSE.
"""

import numpy as np
from bnn.evaluation.cost_error import break_even, cost_at
from runner import reduce_protocol

from .benchmark import counts

# Exponents of the tolerated force error in D: the energy error of a frozen or
# noisy force grows like delta * step * sqrt(D) per move, with the step size of
# MALA falling like D^(-1/6) and that of HMC with fixed L like D^(-1/4).
THEORY = {"hmc1": -1 / 3, "hmc8": -1 / 4, "hmc64": -1 / 4}


def _arm(name):
    if name == "exact":
        return "exact", 0.0
    kind, delta = name.rsplit("_", 1)
    return kind, float(delta)


def _top(rows):
    return rows[-1]


def _exact_curves(metrics):
    """Work and RMSE along each exact HMC sampler."""
    return {
        s: (
            np.array([e["value_gradients"] for e in arms["exact"]]),
            np.array([e.get("rmse", np.inf) for e in arms["exact"]]),
        )
        for s, arms in metrics["curves"].items()
        if s != "ula"
    }


def _best(curves, error):
    with np.errstate(divide="ignore", invalid="ignore"):
        return min(
            float(cost_at(work, errors, error)[0]) for work, errors in curves.values()
        )


def break_even_map(cells):
    """One row per (cell, sampler, arm), and the perfect surrogate as delta 0.

    The perfect surrogate is the exact chain charged as a surrogate: one value
    per transition and the surrogate evaluations. It bounds every learner of
    the force on that target.
    """
    rows = []
    for spec, metrics in cells:
        curves = _exact_curves(metrics)
        c, rung = spec["value_cost"], spec["rungs"][-1]
        base = {
            "family": spec["family"],
            "path": spec["path"],
            "dimension": spec["dimension"],
        }
        for s, arms in metrics["curves"].items():
            vg, v, evaluations = counts(spec, s, True, rung)
            exact = _top(arms["exact"])
            reference = _best(curves, exact.get("rmse", np.inf))
            rows.append(
                base
                | {
                    "sampler": s,
                    "type": "perfect",
                    "delta": 0.0,
                    "rmse": exact.get("rmse"),
                    "rho_star": break_even(reference, vg + c * v, evaluations),
                    # Unbounded for ULA, which needs nothing but the force.
                    "headroom": reference / (vg + c * v) if vg + c * v else None,
                }
            )
            for name, entries in arms.items():
                if name == "exact":
                    continue
                kind, delta = _arm(name)
                top = _top(entries)
                rows.append(
                    base
                    | {
                        "sampler": s,
                        "type": kind,
                        "delta": delta,
                        **{
                            key: top[key]
                            for key in (
                                "rmse",
                                "bias",
                                "rho_star",
                                "rho_star_interval",
                                "rho_star_same",
                                "efficiency",
                                "headroom",
                                "diverged",
                            )
                            if key in top
                        },
                        "acceptance": metrics["selected_acceptance"]
                        .get(s, {})
                        .get(name),
                        "step_size": metrics["step_size"].get(s, {}).get(name),
                    }
                )
    return [{k: v for k, v in row.items() if v is not None} for row in rows]


def _efficiency(parameters, delta, dimension):
    """``a / (1 + (delta / delta_h)^k)`` with ``delta_h = delta_8 (D / 8)^beta``."""
    log_amplitude, log_half, beta, log_power = parameters
    half = np.exp(log_half + beta * np.log(np.asarray(dimension) / 8))
    return np.exp(log_amplitude) / (1 + (np.asarray(delta) / half) ** np.exp(log_power))


def tolerance(rows, fitted=128, predicted=512):
    """How the tolerated force error moves with dimension.

    Per (family, path, sampler, type), the efficiencies at every delta and
    dimension up to ``fitted`` are fitted jointly, in logarithms, by
    ``a / (1 + (delta / delta_h(D))^k)`` with ``delta_h(D) = delta_8 (D/8)^beta``:
    ``delta_h`` is the error at which a sampler keeps half its exact
    efficiency, and the amplitude ``a`` absorbs the noise of the exact arm.
    The fit predicts the efficiencies at ``predicted``.
    """
    from scipy.optimize import least_squares

    keys = sorted(
        {
            (r["family"], r["path"], r["sampler"], r["type"])
            for r in rows
            if r["sampler"] != "ula" and r["type"] != "perfect"
        }
    )
    out = []
    for key in keys:
        points = [
            r
            for r in rows
            if (r["family"], r["path"], r["sampler"], r["type"]) == key
            and r.get("efficiency", 0) > 0
        ]
        used = [r for r in points if r["dimension"] <= fitted]
        dimensions = sorted({r["dimension"] for r in used})
        entry = dict(zip(("family", "path", "sampler", "type"), key, strict=True))
        entry["theory_slope"] = THEORY[key[2]]
        if len(dimensions) < 2:
            out.append(entry)
            continue
        delta = np.array([r["delta"] for r in used])
        dimension = np.array([r["dimension"] for r in used])
        observed = np.log([r["efficiency"] for r in used])

        def residual(parameters, delta=delta, dimension=dimension, observed=observed):
            return np.log(_efficiency(parameters, delta, dimension)) - observed

        fit = least_squares(
            residual,
            [0.0, np.log(0.1), -0.25, np.log(2.0)],
            bounds=([-2, np.log(1e-4), -2, np.log(0.5)], [2, np.log(10), 2, np.log(4)]),
        )
        log_amplitude, log_half, beta, log_power = fit.x
        entry |= {
            "dimensions": dimensions,
            "amplitude": float(np.exp(log_amplitude)),
            "half_delta_8": float(np.exp(log_half)),
            "slope": float(beta),
            "power": float(np.exp(log_power)),
            "log_residual_sd": float(np.sqrt(np.mean(fit.fun**2))),
            "half_delta": {
                str(d): float(np.exp(log_half + beta * np.log(d / 8)))
                for d in sorted({r["dimension"] for r in points} | {predicted})
            },
        }
        grid = sorted({r["delta"] for r in points})
        entry["predicted"] = {
            str(d): float(_efficiency(fit.x, d, predicted)) for d in grid
        }
        measured = {
            str(r["delta"]): r["efficiency"]
            for r in points
            if r["dimension"] == predicted
        }
        if measured:
            entry["measured"] = measured
        out.append(entry)
    return out


def _interpolate(rows, delta):
    """rho_star at ``delta`` from the rows of one (cell, sampler, type) and delta 0."""
    points = sorted((r["delta"], r["rho_star"]) for r in rows if "rho_star" in r)
    deltas, values = np.array([p[0] for p in points]), np.array([p[1] for p in points])
    if delta <= deltas[1]:
        return float(np.interp(delta, deltas[:2], values[:2]))
    return float(np.interp(np.log(delta), np.log(deltas[1:]), values[1:]))


def oracle_at(
    rows, family, path, dimension, sampler, delta, kinds=("conservative", "curl")
):
    """The oracle's rho_star at ``delta`` for frozen fields, by type."""
    out = {}
    for kind in kinds:
        chosen = [
            r
            for r in rows
            if (r["family"], r["path"], r["dimension"], r["sampler"])
            == (family, path, dimension, sampler)
            and r["type"] in (kind, "perfect")
        ]
        if chosen:
            out[kind] = _interpolate(chosen, delta)
    return out


def learned(cells, rows):
    """The fitted MLP force against the oracle at its own force error."""
    out = []
    for spec, metrics in cells:
        if "learned_results" not in metrics:
            continue
        fits = metrics["learned_results"]["fits"]
        delta = float(np.mean(metrics["learned_results"]["force_error"]))
        for s, entries in metrics["learned_results"]["curves"].items():
            top = _top(entries)
            out.append(
                {
                    "family": spec["family"],
                    "path": spec["path"],
                    "dimension": spec["dimension"],
                    "sampler": s,
                    "delta": delta,
                    "delta_range": [
                        float(np.min(metrics["learned_results"]["force_error"])),
                        float(np.max(metrics["learned_results"]["force_error"])),
                    ],
                    "corpus": fits[0]["corpus"],
                    "parameters": fits[0]["parameters"],
                    "rmse": top.get("rmse"),
                    "efficiency": top.get("efficiency"),
                    "rho_star": top.get("rho_star"),
                    "rho_star_sampling": top.get("rho_star_sampling"),
                    "oracle_rho_star": oracle_at(
                        rows, spec["family"], spec["path"], spec["dimension"], s, delta
                    ),
                    "acceptance": metrics["learned_results"]["selected_acceptance"].get(
                        s
                    ),
                }
            )
    return [{k: v for k, v in r.items() if v is not None} for r in out]


def _observations_at(observations, ratios, target):
    """Observations at which the relative cost falls to ``target`` (log-log)."""
    if target <= 0:
        return None
    x, y = np.log(observations), np.log(ratios)
    if np.log(target) >= y[0]:
        return float(observations[0])
    if np.log(target) < y[-1]:
        # Beyond the measured range the gradient's cost grows like N.
        return float(np.exp(x[-1] + (y[-1] - np.log(target))))
    return float(np.exp(np.interp(-np.log(target), -y, x)))


def _mlp(metrics, rows, dimension):
    """One fitted MLP force on the collapsed BNN, read against the oracle map.

    Beyond the largest delta of the grid the oracle's value there is an upper
    bound, because rho_star falls with delta.
    """
    bnn = metrics["collapsed_bnn"]
    delta = bnn["fit"]["force_error"]
    observations = np.array([int(n) for n in bnn["gradient_flops"]])
    ratios = bnn["mlp_flops"] / np.array(list(bnn["gradient_flops"].values()))
    readings = []
    for family, path in sorted({(r["family"], r["path"]) for r in rows}):
        for sampler in ("hmc1", "hmc8", "hmc64", "ula"):
            for kind, value in oracle_at(
                rows, family, path, dimension, sampler, delta
            ).items():
                readings.append(
                    {
                        "family": family,
                        "path": path,
                        "sampler": sampler,
                        "type": kind,
                        "rho_star": value,
                        "observations": _observations_at(observations, ratios, value),
                    }
                )
    realized = bnn["realized"]
    # Both observables have to be estimated, so the worse one binds.
    binding = float(min(realized["rho_star"]))
    return {
        "width": metrics["width"],
        "parameters": bnn["fit"]["parameters"],
        "force_error": delta,
        "holdout_error": bnn["fit"]["holdout_error"],
        "beyond_grid": bool(delta > max(r["delta"] for r in rows)),
        "relative_cost": dict(
            zip(map(str, observations), ratios.tolist(), strict=True)
        ),
        "oracle_readings": [
            {k: v for k, v in r.items() if v is not None} for r in readings
        ],
        "realized_efficiency": realized["efficiency"],
        "realized_rho_star": realized["rho_star"],
        "realized_step_sizes": [
            realized["exact"]["step_size"],
            realized["learned"]["step_size"],
        ],
        "realized_observations": _observations_at(observations, ratios, binding),
    }


def placement(places, rows, dimension=32):
    """Real learners in the (delta, rho) plane, and where each could break even."""
    first = places[0]
    return {
        "collapsed_bnn": [_mlp(metrics, rows, dimension) for metrics in places],
        "whole_force": [
            {
                "width": r["width"],
                "observations": r["observations"],
                "parameters_rho": r["parameters_flops"] / r["gradient_flops"],
                "features_rho": r["features_flops"] / r["gradient_flops"],
            }
            for r in first["whole_force"]
        ],
        "student": [
            {
                "observations": r["observations"],
                "rho": r["student_flops"] / r["gradient_flops"],
            }
            for r in first["student"]
        ],
        "largest_rho_star": float(max(r["rho_star"] for r in rows if "rho_star" in r)),
    }


def summarize(manifest, results):
    summary = reduce_protocol(manifest, results)
    cells = [
        (cell.factors, result["metrics"])
        for cell, result in zip(manifest.cells, results, strict=True)
        if cell.stage == "oracle"
    ]
    rows = break_even_map(cells)
    summary["break_even_map"] = rows
    summary["tolerance"] = tolerance(rows)
    summary["learned"] = learned(cells, rows)
    places = [
        result["metrics"] | {"width": cell.factors["fit"]["width"]}
        for cell, result in zip(manifest.cells, results, strict=True)
        if cell.stage == "placement"
    ]
    if places and rows:
        summary["placement"] = placement(places, rows)
    return summary
