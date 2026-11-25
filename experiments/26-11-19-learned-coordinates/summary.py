"""The reading across cells: what a fitted flow realizes of the oracle's headroom.

Each estimand is read on its own. ``over_classical`` is the work the cheapest
classical chain needs for an arm's RMSE, per unit of the arm's own work, the
fit of a flow included. ``sampling_only`` leaves the fit out. ``rho_star`` is
the cost of one evaluation of the flow, in gradients, below which the arm needs
less work than the cheapest classical chain. Per kind, the chain with the
largest ``over_classical`` at a rung is read; for the flows that choice is made
on the targets it is read on, which favours them.
"""

import numpy as np
from runner import reduce_protocol

from .chains import kind

READ = (
    "rmse",
    "work",
    "fit",
    "over_classical",
    "over_classical_interval",
    "classical_extrapolated",
    "over_constant",
    "rho_star",
    "rho_star_interval",
    "diverged",
)


def _entry(metrics, arm, sampler, rung, estimand):
    row = metrics["curves"][arm][sampler][rung]
    row = (
        row["secondary"] | {"work": row["work"], "fit": row["fit"]} if estimand else row
    )
    settings = metrics["settings"][arm][sampler]
    entry = {"arm": arm, "sampler": sampler} | {k: row[k] for k in READ if k in row}
    if row.get("bias_se"):
        entry["bias_z"] = row["bias"] / row["bias_se"]
    if "classical_work" in row and row["work"] > row["fit"]:
        entry["sampling_only"] = row["classical_work"] / (row["work"] - row["fit"])
    return entry | {
        key: settings.get(key)
        for key in ("acceptance", "target_acceptance", "step_size")
    }


def _best(metrics, arms, rung, estimand, by="over_classical", pick=max):
    """The chain of ``arms`` with the largest ``by`` at a rung."""
    found = [
        _entry(metrics, arm, sampler, rung, estimand)
        for arm in arms
        for sampler in metrics["curves"][arm]
    ]
    found = [entry for entry in found if entry.get(by) is not None]
    return pick(found, key=lambda entry: entry[by]) if found else None


def _quality(metrics, arm):
    fit = metrics["fits"].get(arm, {})
    if "effective_share" not in fit:
        return {}

    # Per target, the screen target first.
    def median(values):
        values = [v for v in values if v is not None]
        return float(np.median(values)) if values else None

    return {
        "divergence": median(fit["divergence"]),
        "effective_share": median(fit["effective_share"]),
    }


def readings(cells):
    rows = []
    for spec, metrics in cells:
        arms = list(metrics["curves"])
        kinds = {
            name: [arm for arm in arms if kind(arm) == name]
            for name in ("classical", "constant", "oracle", "ceiling", "learned")
        }
        row = {
            "family": spec["family"],
            "path": metrics["path"],
            "dimension": spec["dimension"],
            "frame": spec.get("frame"),
            "trajectories": spec.get("trajectories"),
            **metrics["shape"],
            "estimands": metrics["estimands"],
            "rungs": spec["rungs"],
            "fits": {
                arm: _quality(metrics, arm)
                for arm in kinds["ceiling"] + kinds["learned"]
            },
        }
        top = len(spec["rungs"]) - 1
        for estimand, name in enumerate(("first", "second")):
            # The classical chain with the smallest error at the top rung.
            entry = {
                "classical": _best(
                    metrics, kinds["classical"], top, estimand, "rmse", min
                )
            }
            for label in ("constant", "oracle", "ceiling", "learned"):
                if kinds[label]:
                    entry[label] = _best(metrics, kinds[label], top, estimand)
            # Every flow at the top rung, and the best learned one at every rung.
            entry["flows"] = {
                arm: _best(metrics, [arm], top, estimand)
                for arm in kinds["ceiling"] + kinds["learned"]
            }
            entry["learned_by_rung"] = [
                _best(metrics, kinds["learned"], rung, estimand)
                for rung in range(top + 1)
            ]
            if kinds["oracle"]:
                entry["oracle_by_rung"] = [
                    _best(metrics, kinds["oracle"], rung, estimand)
                    for rung in range(top + 1)
                ]
            row[name] = entry
        if "reference_check" in metrics:
            row["reference_check"] = {
                key: np.abs(metrics["reference_check"][key]).max(0).tolist()
                for key in ("error", "particles_off")
            }
        rows.append(row)
    return rows


def exactness(cells):
    """Bias over its standard error at the top rung, by the kind of arm."""
    scores = {}
    for _, metrics in cells:
        for arm, samplers in metrics["curves"].items():
            for rows in samplers.values():
                for top in (rows[-1], rows[-1]["secondary"]):
                    if top.get("bias_se", 0) > 0 and not top.get("diverged"):
                        scores.setdefault(kind(arm), []).append(
                            top["bias"] / top["bias_se"]
                        )
    return {
        name: {
            "readings": len(values),
            "mean": float(np.mean(values)),
            "sd": float(np.std(values)),
            "beyond_3": int(np.sum(np.abs(values) > 3)),
        }
        for name, values in scores.items()
    }


def summarize(manifest, results):
    summary = reduce_protocol(manifest, results)
    cells = [
        (cell.factors, result["metrics"])
        for cell, result in zip(manifest.cells, results, strict=True)
    ]
    summary["readings"] = readings(cells)
    summary["exactness"] = exactness(cells)
    return summary
