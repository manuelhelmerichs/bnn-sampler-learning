"""Readings per campaign and sampler, one row per metric.

Chains are not stationary, so every reading is taken against the gradients
spent. The potential is ``U = -log posterior`` on the training set.
"""

import numpy as np
from runner import read_arrays, reduce_protocol, result_folder

from .benchmark import ARTIFACT

# Gradients between two records at which movement is compared across metrics.
LAG = 10_000
# Records averaged into the running potential that `reach` reads.
WINDOW = 5


def over_chains(values):
    """Median, minimum and maximum of one value per chain."""
    values = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(values)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def split(arrays):
    """Records of each chain in order, as one dictionary of arrays per chain."""
    names = [
        name
        for name in arrays
        if len(arrays[name]) == len(arrays["chain"])
        and not name.startswith(("segment_", "stored_", "movement"))
        and name != "chain"
    ]
    return [
        {name: arrays[name][arrays["chain"] == chain] for name in names}
        for chain in np.unique(arrays["chain"])
    ]


def tail(values, share=0.2):
    """Mean over the last ``share`` of a series."""
    return float(np.mean(values[-max(1, int(len(values) * share)) :]))


def log_mean_exp(terms):
    """``log mean exp`` over the first axis."""
    peak = terms.max(axis=0)
    return peak + np.log(np.mean(np.exp(terms - peak), axis=0))


def log_mixture(prediction, log_sigma, targets):
    """Log density of each test target under the mixture over draws."""
    prediction = np.asarray(prediction, np.float64)
    log_sigma = np.asarray(log_sigma, np.float64)[:, None]
    return log_mean_exp(
        -0.5 * np.log(2 * np.pi)
        - log_sigma
        - 0.5 * ((targets - prediction) / np.exp(log_sigma)) ** 2
    )


def log_predictive(prediction, log_sigma, targets):
    """Mean log density of the test targets under the mixture over draws."""
    return float(np.mean(log_mixture(prediction, log_sigma, targets)))


def predictive(predictions, log_sigmas, targets):
    """Readings of the chains pooled with equal weight.

    Each chain is the mixture over its own draws, and the pool is the mixture
    over chains, so a chain that records more often (a NUTS chain whose trees
    collapse) does not weigh more.
    """
    terms = np.stack(
        [
            log_mixture(prediction, log_sigma, targets)
            for prediction, log_sigma in zip(predictions, log_sigmas, strict=True)
        ]
    )
    mean = np.mean([np.mean(prediction, axis=0) for prediction in predictions], axis=0)
    return {
        "log_density": float(np.mean(log_mean_exp(terms))),
        "rmse": float(np.sqrt(np.mean((mean - targets) ** 2))),
    }


def at_lag(curve, column, lag=LAG):
    """A movement curve at ``lag`` gradients, interpolated in logarithms."""
    return float(
        np.exp(np.interp(np.log(lag), np.log(curve[:, 0]), np.log(curve[:, column])))
    )


def reach(chain, level):
    """Gradients until the running potential first falls to ``level``."""
    potential = chain["potential"]
    window = min(WINDOW, len(potential))
    running = np.convolve(potential, np.ones(window) / window, mode="valid")
    below = np.flatnonzero(running <= level)
    return float(chain["gradients"][below[0] + window - 1]) if len(below) else None


def cell(metrics, arrays, spec):
    """The readings of one sampler under one metric on one campaign."""
    chains = split(arrays)
    targets = arrays["test_targets"]
    segments = range(len(spec["boundaries"]))
    last = segments[-1]

    def of(chain, segment):
        keep = chain["segment"] == segment
        if "frozen" in chain and segment == last:
            keep &= chain["frozen"] == 1
        return {name: values[keep] for name, values in chain.items()}

    final = [of(chain, last) for chain in chains]
    curve = arrays["movement"].mean(axis=0)
    sigma = float(np.median([np.exp(tail(chain["log_sigma"])) for chain in final]))
    # Predictions of the pooled chains per segment, and of the stored campaign.
    pooled = []
    for segment in segments:
        rows = [of(chain, segment) for chain in chains]
        pooled.append(
            {
                "gradients": spec["boundaries"][segment],
                **predictive(
                    [row["prediction"] for row in rows],
                    [row["log_sigma"] for row in rows],
                    targets,
                ),
                "by_chain": over_chains(
                    [
                        log_predictive(row["prediction"], row["log_sigma"], targets)
                        for row in rows
                    ]
                ),
            }
        )
    # Between-chain over within-chain variance of the test predictions.
    means = np.stack([chain["prediction"].mean(axis=0) for chain in final])
    within = np.mean([chain["prediction"].var(axis=0) for chain in final], axis=0)
    tuning = {
        name.removeprefix("segment_"): np.median(values, axis=0).tolist()
        for name, values in arrays.items()
        if name.startswith("segment_") and name not in ("segment_used", "segment_theta")
    }
    readings = {
        **{key: metrics[key] for key in ("dataset", "sampler", "arm")},
        "parameters": metrics["parameters"],
        "observations": metrics["observations"],
        "chains": len(chains),
        "gradients": metrics["gradients"],
        # Median over chains of the potential at the end of each segment.
        "potential": [
            float(np.median([tail(of(chain, s)["potential"]) for chain in chains]))
            for s in segments
        ],
        "final_potential": over_chains([tail(chain["potential"]) for chain in final]),
        "stored_potential": over_chains(arrays["stored_potential"][:, -1]),
        "sigma": sigma,
        "predictive": pooled,
        "stored_predictive": predictive(
            arrays["stored_prediction"], arrays["stored_log_sigma"], targets
        ),
        "movement": {
            "gradients": curve[:, 0].tolist(),
            "weights": curve[:, 1].tolist(),
            # In units of the noise variance.
            "predictions": (curve[:, 2] / sigma**2).tolist(),
            "at_lag": {
                "gradients": LAG,
                "weights": at_lag(curve, 1),
                "predictions": at_lag(curve, 2) / sigma**2,
            }
            if curve[0, 0] <= LAG <= curve[-1, 0]
            else None,
        },
        "between_within": float(np.median(means.var(axis=0, ddof=1) / within)),
        "tuning": tuning,
        "seconds_per_gradient": metrics["seconds"]["sampling"]
        / (metrics["gradients"] * len(chains)),
        # Local metrics per chain, in gradients.
        "metric_gradients": metrics["metric_evaluations"]
        * metrics["seconds"]["metric"]
        / metrics["seconds"]["gradient"],
    }
    if "energy" in arrays:
        energy = np.concatenate([chain["energy"] for chain in final])
        readings["energy"] = {
            "mean": float(energy.mean()),
            "median": float(np.median(energy)),
            "failures": int(sum(chain["failures"].sum() for chain in chains)),
        }
    return readings, chains


def compare(rows, chains):
    """Readings against the same sampler without a metric, chain by chain."""
    groups = {}
    for index, row in enumerate(rows):
        groups.setdefault((row["dataset"], row["sampler"]), []).append(index)
    for members in groups.values():
        base = [i for i in members if rows[i]["arm"] == "identity"]
        if not base:
            continue
        base = base[0]
        levels = [tail(chain["potential"]) for chain in chains[base]]
        for index in members:
            row = rows[index]
            reached = [
                reach(chain, level)
                for chain, level in zip(chains[index], levels, strict=True)
            ]
            found = [value for value in reached if value is not None]
            row["against_identity"] = {
                # Chains whose potential falls to where the same chain without
                # a metric ends, and the gradients they need for it. Without a
                # metric this is where a chain first gets to its own end.
                "reached": len(found),
                "gradients": float(np.median(found)) if found else None,
                "potential": row["final_potential"]["median"]
                - rows[base]["final_potential"]["median"],
                "log_density": row["predictive"][-1]["log_density"]
                - rows[base]["predictive"][-1]["log_density"],
                **(
                    {
                        name: row["movement"]["at_lag"][name]
                        / rows[base]["movement"]["at_lag"][name]
                        for name in ("weights", "predictions")
                    }
                    if row["movement"]["at_lag"] and rows[base]["movement"]["at_lag"]
                    else {}
                ),
            }


def summarize(manifest, results):
    summary = reduce_protocol(manifest, results)
    root = result_folder()
    rows, chains = [], []
    for spec, result in zip(manifest.cells, results, strict=True):
        readings, records = cell(
            result["metrics"],
            read_arrays(root / "cells" / spec.cell_id / ARTIFACT),
            spec.factors,
        )
        rows.append(readings)
        chains.append(records)
    compare(rows, chains)
    summary["cells"] = rows
    return summary
