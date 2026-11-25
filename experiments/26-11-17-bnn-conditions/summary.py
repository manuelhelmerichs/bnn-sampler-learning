"""Per-campaign readings: medians over chains of each chain's median over draws."""

import numpy as np
from runner import read_arrays, reduce_protocol, result_folder

from .conditions import (
    ARTIFACT,
    CONSTANT,
    WALKS,
    later,
)


def over_chains(values):
    """Median, minimum and maximum of one value per chain."""
    values = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(values)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def step_at(steps, drops, level=1.0):
    """Step length at which the drop of the log posterior reaches ``level`` nats.

    Short steps lose log posterior in proportion to their squared length, so the
    squared length is interpolated linearly in the drop between two steps. A
    drop that stays below the level returns the longest step.
    """
    order = np.argsort(steps)
    steps, drops = np.asarray(steps)[order], np.asarray(drops)[order]
    above = np.flatnonzero(drops > level)
    if not len(above) or above[0] == 0:
        return float(steps[-1 if not len(above) else 0])
    low, high = above[0] - 1, above[0]
    fraction = (level - drops[low]) / (drops[high] - drops[low])
    return float(
        np.sqrt(steps[low] ** 2 + fraction * (steps[high] ** 2 - steps[low] ** 2))
    )


def campaign(metrics, arrays, spec):
    dimension = metrics["parameters"]
    ranks = [*(rank for rank in spec["ranks"] if rank < dimension), dimension]
    eigenvalues = arrays["eigenvalues"]
    chain_logpost = arrays["chain_logpost"]
    corpus = [
        {
            "rank": rank,
            **{
                name: over_chains(arrays["corpus"][:, index, column])
                for column, name in enumerate(
                    ("drop", "spread", "ess_fraction", "acceptance", "jeffreys")
                )
            },
        }
        for index, rank in enumerate(ranks)
    ]
    local = arrays["local_gaussian"]
    local_gaussian = [
        {
            "scale": scale,
            "quadratic": scale * dimension / 2,
            **{
                name: over_chains(np.median(local[:, :, index, column], axis=1))
                for name, column in (
                    ("drop_mean", 0),
                    ("drop_median", 1),
                    ("logweight_sd", 3),
                    ("ess_fraction", 4),
                )
            },
        }
        for index, scale in enumerate(spec["local_scales"])
    ]
    spectrum = {
        "condition": over_chains(
            np.median(eigenvalues[..., 0] / eigenvalues[..., -1], axis=1)
        ),
        "top": over_chains(np.median(eigenvalues[..., 0], axis=1)),
        "effective_dimension": over_chains(
            np.median(np.sum(1 - 1 / eigenvalues, axis=-1), axis=1)
        ),
        **{
            f"above_{level}": over_chains(
                np.median(np.sum(eigenvalues > level, axis=-1), axis=1)
            )
            for level in (2, 10, 100)
        },
        "hessian_negative": over_chains(np.median(arrays["hessian"][..., 0], axis=1)),
        "hessian_lowest": over_chains(np.median(arrays["hessian"][..., 1], axis=1)),
    }
    # A metric adapted on the first quarter of a chain is read after that quarter.
    anchors = np.array(later(spec))
    everywhere = np.ones_like(anchors)
    draws = 1 + len(spec["lags"])
    constant = [
        {
            "metric": name,
            **{
                reading: over_chains(
                    np.median(
                        arrays["constant"][:, index][
                            :,
                            np.repeat(
                                anchors if name == "mean_early" else everywhere, draws
                            ),
                            column,
                        ],
                        axis=1,
                    )
                )
                for column, reading in enumerate(("condition", "decile_ratio"))
            },
        }
        for index, name in enumerate(CONSTANT)
    ]
    lags = arrays["stale_lags"][0]

    def by_lag(lag, column):
        return over_chains(np.median(arrays["stale"][:, lags == lag, column], axis=1))

    stale = [
        {
            "lag": int(lag),
            "condition": by_lag(lag, 0),
            "decile_ratio": by_lag(lag, 1),
            "overlap": {
                str(rank): by_lag(lag, 2 + index)
                for index, rank in enumerate(spec["overlap_ranks"])
            },
        }
        for lag in np.unique(lags)
    ]
    across = {
        "condition": float(np.median(arrays["across"][:, 0])),
        "decile_ratio": float(np.median(arrays["across"][:, 1])),
        "overlap": {
            str(rank): float(np.median(arrays["across"][:, 2 + index]))
            for index, rank in enumerate(spec["overlap_ranks"])
        },
        "unrelated_overlap": {
            str(rank): rank / dimension for rank in spec["overlap_ranks"]
        },
    }
    walk = []
    for index, name in enumerate(WALKS):
        # (chain, anchor, step, [drop mean, drop median, acceptance, ESJD])
        rows = arrays["walk"][:, index][
            :, anchors if name == "mean_early" else everywhere
        ]
        pooled = rows.mean(axis=(0, 1))
        best = int(np.argmax(pooled[:, 3]))
        walk.append(
            {
                "metric": name,
                "steps": [
                    {
                        "step": step,
                        "drop_median": float(np.median(rows[:, :, rung, 1])),
                        "acceptance": float(pooled[rung, 2]),
                        "esjd": float(pooled[rung, 3]),
                    }
                    for rung, step in enumerate(spec["steps"])
                ],
                "best": {
                    "step": spec["steps"][best],
                    "acceptance": float(pooled[best, 2]),
                    "esjd": float(pooled[best, 3]),
                    # The best step is the ladder's first or last: a bound only.
                    "at_edge": best in (0, len(spec["steps"]) - 1),
                },
                "step_at_unit_drop": step_at(
                    spec["steps"], np.median(rows[..., 1], axis=(0, 1))
                ),
                "esjd_by_chain": over_chains(rows.mean(axis=1)[:, best, 3]),
            }
        )
    identity = walk[WALKS.index("identity")]["best"]["esjd"]
    seconds = metrics["seconds"]
    return {
        **{key: metrics[key] for key in ("dataset", "parameters", "observations")},
        "chains": metrics["chains"],
        "draws": metrics["draws"],
        "chain": {
            "logpost_sd": over_chains(chain_logpost.std(axis=1)),
            "mean_drop": over_chains(
                chain_logpost.mean(axis=1) - arrays["mean_logpost"]
            ),
            "end_to_end": over_chains(arrays["end_to_end"]),
            "displacement": {
                str(lag): over_chains(arrays["displacement"][:, index])
                for index, lag in enumerate(spec["displacement_lags"])
            },
        },
        "corpus": corpus,
        "local_gaussian": local_gaussian,
        "spectrum": spectrum,
        "constant": constant,
        "stale": stale,
        "across": across,
        "walk": walk,
        # Expected squared jump distance at the best step, relative to no metric.
        "walk_gain": {row["metric"]: row["best"]["esjd"] / identity for row in walk}
        if identity > 0
        else {},
        "seconds": seconds,
        "relative_cost": {
            name: seconds[name] / seconds["gradient"]
            for name in ("metric", "hessian", "eigendecomposition")
        },
    }


def summarize(manifest, results):
    summary = reduce_protocol(manifest, results)
    root = result_folder()
    summary["campaigns"] = [
        campaign(
            result["metrics"],
            read_arrays(root / "cells" / cell.cell_id / ARTIFACT),
            cell.factors,
        )
        for cell, result in zip(manifest.cells, results, strict=True)
    ]
    return summary
