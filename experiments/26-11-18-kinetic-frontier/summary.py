"""The reading across cells.

``exact``: per mixture, how often Gaussian HMC changes mode against its bound,
and the best other kinetic energy relative to the Gaussian. Per local target,
the same for the error of two moments.

``frontier``: per condition, every arm's error of the mode mass and of the
second moments at the budget, and the loss of each learned arm over the best
unadjusted classical arm's, on the confirmation targets and on changed tasks.

``budget``: the learned sampler and classical ones tuned at each budget.

``predictive``: on regression networks, the model-average loss of every arm
and the error of its predictive mean and spread against the reference.
"""

import numpy as np
from runner import reduce_protocol

READ = (
    "mass_rmse",
    "mass_rmse_interval",
    "local_rmse",
    "loss",
    "loss_interval",
    "mass_bias",
    "second",
    "barrier",
    "fourth",
    "tail",
    "axis",
    "switches",
    "diverged",
)
UNADJUSTED = ("euler", "baoab", "mclmc")


def _read(entry):
    return {key: entry[key] for key in READ if key in entry}


def exact(manifest, results):
    summary = reduce_protocol(manifest, results)
    crossing, local = [], []
    for cell, result in zip(manifest.cells, results, strict=True):
        metrics = result["metrics"]
        families = metrics["family_results"]
        others = {name: entry for name, entry in families.items() if name != "gaussian"}
        if cell.stage == "crossing":
            best = max(others, key=lambda name: others[name]["over_gaussian"])
            crossing.append(
                {
                    "separation": metrics["separation"],
                    "dimension": metrics["dimension"],
                    "gaussian": {
                        key: families["gaussian"][key]
                        for key in (
                            "rate",
                            "rate_interval",
                            "transmission",
                            "step",
                            "length",
                            "acceptance",
                        )
                    },
                    # Share of the bound at the leapfrog's stability limit.
                    "of_stability_bound": families["gaussian"]["rate"]
                    / metrics["stability_bound"],
                    "best_other": best,
                    "best_over_gaussian": others[best]["over_gaussian"],
                    "best_over_gaussian_interval": others[best][
                        "over_gaussian_interval"
                    ],
                    "over_gaussian": {
                        name: entry["over_gaussian"] for name, entry in others.items()
                    },
                    "transmission": {
                        name: entry["transmission"] for name, entry in families.items()
                    },
                }
            )
        else:
            local.append(
                {
                    "family": metrics["family"],
                    "dimension": metrics["dimension"],
                    "gaussian": {
                        key: families["gaussian"][key]
                        for key in ("mse", "step", "length", "acceptance")
                    },
                    "over_gaussian": {
                        name: entry["over_gaussian"] for name, entry in others.items()
                    },
                    "over_gaussian_interval": {
                        name: entry["over_gaussian_interval"]
                        for name, entry in others.items()
                    },
                }
            )
    summary["crossing"], summary["local"] = crossing, local
    summary["accepted"] = True
    return summary


def frontier(manifest, results):
    summary = reduce_protocol(manifest, results)
    classical, learned = {}, []
    for cell, result in zip(manifest.cells, results, strict=True):
        metrics = result["metrics"]
        condition = (metrics["separation"], metrics["dimension"])
        if cell.stage == "classical":
            classical[condition] = metrics
        else:
            learned.append((condition, metrics))
    rows = []
    for (separation, dimension), metrics in classical.items():
        arms = metrics["arms"]
        best = min(UNADJUSTED, key=lambda name: arms[name]["loss"])
        rows.append(
            {
                "separation": separation,
                "dimension": dimension,
                "budget": metrics["budget"],
                "best_unadjusted": best,
                "arms": {
                    name: {"setting": arms[name]["setting"]}
                    | _read(arms[name])
                    # Every setting of the grid that ran without diverging, as
                    # [second-moment error, mass error]: the classical cloud.
                    | (
                        {
                            "grid": [
                                [point["local_rmse"], point["mass_rmse"]]
                                for point in arms[name]["grid"]
                                if point["diverged"] == 0
                            ]
                        }
                        if name in UNADJUSTED
                        else {}
                    )
                    for name in (*UNADJUSTED, "hmc")
                }
                | {
                    "smc": {
                        key: arms["smc"][key]
                        for key in (
                            "mass_rmse",
                            "local_rmse",
                            "mass_rmse_extrapolated",
                        )
                    }
                },
                "learned": [],
            }
        )
        for condition, cell in learned:
            if condition != (separation, dimension):
                continue
            reference = arms[best]["loss"]
            entry = {
                "form": cell["form"],
                "frame": cell["frame"],
                "hyper": cell["hyper"],
                "training_gradients": cell["training_gradients"],
                "linear_explained": cell["linear_explained"],
                "arms": {name: _read(arm) for name, arm in cell["arms"].items()},
                # Loss of the best unadjusted classical arm over the learned one's.
                "gain": reference / cell["arms"]["learned"]["loss"],
                "gain_linear": reference / cell["arms"]["linear"]["loss"]
                if "linear" in cell["arms"]
                else None,
                "transfer": {},
            }
            for name, arms_ in cell["transfer_results"].items():
                if name.startswith("frame="):
                    # Classical arms do not depend on the frame.
                    reference_ = min(arms[u]["loss"] for u in UNADJUSTED)
                else:
                    reference_ = min(
                        metrics["transfer_results"][name][u]["loss"] for u in UNADJUSTED
                    )
                entry["transfer"][name] = {
                    "classical_loss": reference_,
                    "gain": reference_ / arms_["learned"]["loss"],
                } | {arm: _read(values) for arm, values in arms_.items()}
            rows[-1]["learned"].append(entry)
    summary["conditions"] = rows
    summary["accepted"] = True
    return summary


def budget(manifest, results):
    summary = reduce_protocol(manifest, results)
    arms = ("baoab", "baoab_unit", "bridge_raw", "bridge_reweighted", "learned")
    summary["conditions"] = [
        {
            "separation": result["metrics"]["separation"],
            "dimension": result["metrics"]["dimension"],
            "budgets": [
                {"multiple": entry["multiple"], "budget": entry["budget"]}
                | {
                    arm: {
                        key: entry[arm][key]
                        for key in (
                            "mass_rmse",
                            "mass_rmse_interval",
                            "mass_bias",
                            "local_rmse",
                            "loss",
                            "barrier",
                            "axis",
                        )
                    }
                    | (
                        {"setting": entry[arm]["setting"]}
                        if "setting" in entry[arm]
                        else {}
                    )
                    for arm in arms
                }
                for entry in result["metrics"]["budgets"]
            ],
        }
        for result in results
    ]
    summary["accepted"] = True
    return summary


def _fidelity(arm, exact, grid):
    """Errors of an arm's predictive mean and spread against a reference.

    The mean's error is in units of the reference's spread. Inputs within the
    data's range count as on the data, the gap and the outside as off it.
    """
    mean, spread = np.array(arm["mean"]), np.array(arm["spread"])
    exact_mean, exact_spread = np.array(exact["mean"]), np.array(exact["spread"])
    on = (np.abs(grid) >= 0.5) & (np.abs(grid) <= 2.0)
    error = (mean - exact_mean) / exact_spread
    ratio = np.log(spread / exact_spread)
    out = {}
    for name, mask in (("on", on), ("off", ~on)):
        out[name] = {
            "mean_rmse": float(np.sqrt(np.mean(error[:, mask] ** 2))),
            "log_spread": float(np.mean(ratio[:, mask])),
            "log_spread_rmse": float(np.sqrt(np.mean(ratio[:, mask] ** 2))),
        }
        if "mean_between_trials" in arm:
            noise = np.array(arm["mean_between_trials"]) / exact_spread
            out[name]["mean_noise"] = float(np.sqrt(np.mean(noise[:, mask] ** 2)))
    return out


def predictive(manifest, results):
    summary = reduce_protocol(manifest, results)
    classical = next(
        result["metrics"]
        for cell, result in zip(manifest.cells, results, strict=True)
        if cell.stage == "network"
    )
    grid = np.array(classical["grid"])
    exact = classical["reference_results"]

    def read(arm):
        return {
            key: arm[key]
            for key in ("setting", "loss", "loss_interval", "diverged")
            if key in arm
        } | {"fidelity": _fidelity(arm, exact["hmc"], grid)}

    summary["reference"] = {
        "smc_loss": float(np.mean(exact["smc"]["loss"])),
        "hmc_loss": float(np.mean(exact["hmc"]["loss"])),
        "hmc_rhat": exact["hmc_rhat"],
        # How far the population is from the chains, in the units of every arm.
        "smc_against_hmc": _fidelity(exact["smc"], exact["hmc"], grid),
    }
    summary["classical"] = {name: read(arm) for name, arm in classical["arms"].items()}
    summary["learned"] = [
        {
            "form": result["metrics"]["form"],
            "start": result["metrics"]["start"],
            "hyper": result["metrics"]["hyper"],
            "training_gradients": result["metrics"]["training_gradients"],
            "arms": {
                name: read(arm) for name, arm in result["metrics"]["arms"].items()
            },
            "mixture": {
                name: {
                    key: arm[key]
                    for key in ("mass_rmse", "local_rmse", "loss", "second", "diverged")
                }
                for name, arm in result["metrics"]["mixture_results"].items()
            },
        }
        for cell, result in zip(manifest.cells, results, strict=True)
        if cell.stage == "network_learned"
    ]
    summary["accepted"] = True
    return summary
