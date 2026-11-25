"""The reading across cells: headroom of a metric and the price it may have.

Every number refers to the largest rung of a cell, and each estimand is read on
its own. ``over_classical`` is the work the cheapest classical chain needs for
an arm's RMSE, per unit of the arm's own work; ``over_constant`` is the same
against the chains with an exact constant metric. ``rho_star`` is the cost of
one metric evaluation, in gradients, below which a position-dependent arm needs
less work than the cheapest classical chain.
"""

import numpy as np
from runner import reduce_protocol

from .chains import KINDS, LOCAL

READ = (
    "rmse",
    "over_classical",
    "over_classical_interval",
    "classical_extrapolated",
    "over_constant",
    "over_constant_interval",
    "constant_extrapolated",
    "rho_star",
    "rho_star_interval",
    "diverged",
)
# Oracle chains: the Gaussianizing coordinates, or the reduced ones.
ORACLES = ("transport", "reduced")


def _top(metrics, arm, sampler, estimand):
    top = metrics["curves"][arm][sampler][-1]
    return top["secondary"] if estimand else top


def _best(metrics, arms, estimand):
    """The (arm, sampler) of ``arms`` with the smallest RMSE at the top rung."""
    found = [
        (_top(metrics, arm, sampler, estimand).get("rmse", np.inf), arm, sampler)
        for arm in arms
        for sampler in metrics["curves"].get(arm, {})
    ]
    return min(found)[1:] if found else None


def _reading(metrics, arm, sampler, estimand):
    top = _top(metrics, arm, sampler, estimand)
    settings = metrics["settings"][arm][sampler]
    return (
        {"arm": arm, "sampler": sampler}
        | {key: top[key] for key in READ if key in top}
        | {"bias_z": top["bias"] / top["bias_se"] if top.get("bias_se") else None}
        | {
            key: settings[key]
            for key in ("acceptance", "target_acceptance", "step_size")
        }
    )


def headroom(cells):
    """Per cell and estimand: the best chain of each kind and every local arm."""
    rows = []
    for spec, metrics in cells:
        row = {
            "family": spec["family"],
            "path": spec["path"],
            "dimension": spec["dimension"],
            **spec["shape"],
            "estimands": metrics["estimands"],
            # A local metric of an explicit arm, by Hessian-vector products.
            "price": metrics["hvp_cost"] * spec["curvature_products"],
        }
        for estimand, name in enumerate(("first", "second")):
            entry = {}
            for kind in ("classical", "constant"):
                best = _best(
                    metrics, [a for a, k in KINDS.items() if k == kind], estimand
                )
                if best:
                    entry[kind] = _reading(metrics, *best, estimand)
            best = _best(metrics, ORACLES, estimand)
            if best:
                entry["oracle"] = _reading(metrics, *best, estimand)
            for arm in LOCAL:
                if arm in metrics["curves"]:
                    entry[arm] = _reading(metrics, arm, "mala", estimand)
            row[name] = entry
        rows.append(row)
    return rows


def scaling(rows, readings=("over_classical", "over_constant")):
    """Log-log slope of the oracle's headroom in the dimension, per family and path."""

    def group(row):
        return row["family"], row["path"], row.get("condition"), row.get("bend")

    out = []
    for key in sorted({group(row) for row in rows}, key=str):
        chosen = sorted(
            (row for row in rows if group(row) == key), key=lambda r: r["dimension"]
        )
        entry = dict(zip(("family", "path", "condition", "bend"), key, strict=True))
        entry = {k: v for k, v in entry.items() if v is not None}
        for name in ("first", "second"):
            for reading in readings:
                points = [
                    (row["dimension"], row[name]["oracle"][reading])
                    for row in chosen
                    if row[name].get("oracle", {}).get(reading, 0) > 0
                ]
                if len(points) > 1:
                    d, value = np.log(np.array(points)).T
                    entry[f"{name}_{reading}"] = {str(int(p[0])): p[1] for p in points}
                    entry[f"{name}_{reading}_slope"] = float(np.polyfit(d, value, 1)[0])
        out.append(entry)
    return out


def exactness(cells):
    """Bias over its standard error, by the kind of arm.

    A chain that keeps the target and starts at an exact draw has no bias. A
    chain that adapts to where it is and then mixes slowly can have one.
    """
    scores = {}
    for _, metrics in cells:
        for arm, samplers in metrics["curves"].items():
            for rows in samplers.values():
                for top in (rows[-1], rows[-1]["secondary"]):
                    if top.get("bias_se", 0) > 0 and not top.get("diverged"):
                        scores.setdefault(KINDS[arm], []).append(
                            top["bias"] / top["bias_se"]
                        )
    return {
        kind: {
            "readings": len(values),
            "mean": float(np.mean(values)),
            "sd": float(np.std(values)),
            "beyond_3": int(np.sum(np.abs(values) > 3)),
        }
        for kind, values in scores.items()
    }


def summarize(manifest, results):
    summary = reduce_protocol(manifest, results)
    cells = [
        (cell.factors, result["metrics"])
        for cell, result in zip(manifest.cells, results, strict=True)
    ]
    rows = headroom(cells)
    summary["headroom"] = rows
    summary["scaling"] = scaling(rows)
    summary["exactness"] = exactness(cells)
    return summary
