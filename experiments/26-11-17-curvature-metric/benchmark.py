"""One cell per run, sampler and metric: the run's chains, continued.

Every chain of a ``trajectory`` run is continued from its last stored draw
for a fixed number of gradients. The run is cut into segments. A segment starts
by tuning the sampler under the current metric; an adaptive metric is estimated
again from the segment before. Potentials and test predictions are recorded
along the way, so that burn-in and predictions can be read against the
gradients spent; how far weights and predictions move is read in the last
segment.
"""

from time import perf_counter

import numpy as np
from runner import CellSpec, Design, ResourceHint, atomic_npz, cell_folder

from . import kernels
from .posterior import (
    block_factor,
    dense_factor,
    diagonal_factor,
    low_rank_factor,
    mean_metric,
    posterior,
    spread_factor,
)

QUESTION = (
    "Does a constant metric built from curvature make a sampler of a BNN "
    "posterior faster, and how much does the metric have to know?"
)
DATASETS = ("airfoil", "energy", "bikesharing")
# The `bnn` configuration of each dataset; cells read its run.
CONFIGS = {
    dataset: f"configs/trajectory/uci_{dataset}_mile_8x100000.yaml"
    for dataset in DATASETS
}
ARTIFACT = "chains.npz"
SAMPLERS = ("mclmc", "nuts")
# Per arm: what the metric is estimated from, which of its entries are kept, and
# whether it is estimated again from each segment (otherwise once, from stored
# draws).
ARMS = {
    "identity": (None, None, False),
    "draws_diagonal": ("draws", "diagonal", True),
    "fisher_diagonal": ("fisher", "diagonal", True),
    "fisher_blocks": ("fisher", "blocks", True),
    "fisher_low_rank": ("fisher", "low_rank", True),
    "fisher_dense": ("fisher", "dense", True),
    "fisher_dense_frozen": ("fisher", "dense", False),
}
# Readings of one chain in one segment; which exist depends on the sampler.
DIAGNOSTICS = (
    "step_size",
    "length",
    "energy_variance",
    "mean_acceptance",
    "leapfrogs",
    "divergences",
)
# Readings of one record; which exist depends on the sampler.
READINGS = ("energy", "failures", "acceptance", "frozen")
SMOKE = (
    ("mclmc", "identity"),
    ("mclmc", "draws_diagonal"),
    ("mclmc", "fisher_dense"),
    ("mclmc", "fisher_low_rank"),
    ("nuts", "fisher_diagonal"),
    ("nuts", "fisher_blocks"),
    ("nuts", "fisher_dense_frozen"),
)
BOUNDARIES = {
    "smoke": [400, 800],
    "pilot": [12_500, 25_000, 50_000, 100_000],
    "production": [12_500, 25_000, 50_000, 100_000, 200_000, 400_000],
}
# Minutes of one production cell per dataset, with a margin.
WALLTIME = {"airfoil": 120, "energy": 90, "bikesharing": 360}


def settings(sampler, smoke):
    if sampler == "mclmc":
        return {
            # The warmup of the stored campaigns, shortened to one segment.
            "tuning_steps": 40 if smoke else 2500,
            "energy_variance": [5e-4, 5e-6],
            "step_size": 0.01,
            # Steps between records; a step takes two gradients.
            "thin": 20 if smoke else 250,
        }
    return {
        "depth": 3 if smoke else 10,
        "acceptance": 0.8,
        "tuning_depth": 2 if smoke else 6,
        "tuning_transitions": 5 if smoke else 40,
        "step_size": 1e-3,
    }


def factors(profile, sampler, arm):
    smoke = profile == "smoke"
    return {
        "profile": profile,
        "sampler": sampler,
        "arm": arm,
        "settings": settings(sampler, smoke),
        # Cumulative gradients at which a segment ends.
        "boundaries": BOUNDARIES[profile],
        # Share of the stored chain, at its end, that the first metric is estimated from.
        "history": 0.25,
        # Local metrics averaged into a Fisher metric.
        "references": 2 if smoke else 32,
        # Eigendirections that the low-rank metric keeps beside its diagonal.
        "rank": 2 if smoke else 64,
        "test_points": 8 if smoke else 256,
        # Stored draws at which the potential and the predictions are read.
        "stored_potentials": 5 if smoke else 65,
        "stored_predictions": 8 if smoke else 256,
    }


def design():
    cells = [
        CellSpec(
            "chains",
            {**factors("smoke", sampler, arm), "dataset": "synthetic"},
            {"chains": 560000 + index},
            expected_schema=("metrics",),
        )
        for index, (sampler, arm) in enumerate(SMOKE)
    ]
    for profile, datasets in (("pilot", DATASETS[:1]), ("production", DATASETS)):
        cells.extend(
            CellSpec(
                "chains",
                {**factors(profile, sampler, arm), "dataset": dataset},
                {"chains": 560100 + index},
                inputs={"run": CONFIGS[dataset]},
                expected_schema=("metrics",),
                role="pilot" if profile == "pilot" else "descriptive",
                resources=ResourceHint(
                    cpu=1, memory_gb=3, walltime_minutes=WALLTIME[dataset]
                ),
            )
            for index, dataset in enumerate(datasets)
            for sampler in SAMPLERS
            for arm in ARMS
        )
    return Design(QUESTION, ("smoke", "pilot", "production"), ("chains",), tuple(cells))


def history(draws, spec):
    """What a continued chain needs of its stored draws ``(draw, parameter)``."""
    count = len(draws)
    recent = draws[-max(2, int(count * spec["history"])) :].astype(np.float64)

    def spaced(points, number):
        index = np.linspace(0, len(points) - 1, min(number, len(points))).astype(int)
        return index, points[index].astype(np.float64)

    stored_index, stored = spaced(draws, spec["stored_potentials"])
    return {
        "start": recent[-1],
        "references": spaced(recent, spec["references"])[1],
        "variance": recent.var(axis=0),
        "count": len(recent),
        "stored_index": stored_index,
        "stored": stored,
        "stored_recent": spaced(recent, spec["stored_predictions"])[1],
    }


def smoke_model():
    """The mean-regression posterior of the smoke network, as a run would define it."""
    from bnn.data.config import Task
    from bnn.models.tabular import FCN, FCNConfig
    from bnn.target import ProbabilisticModel
    from bnn.target.priors import PriorDist

    return ProbabilisticModel(
        FCN(FCNConfig(hidden_structure=(4, 1))),
        PriorDist.StandardNormal.get_prior(),
        Task.MEAN_REGRESSION,
    )


def smoke_problem(seed, spec):
    """A small regression network with short unadjusted Langevin chains."""
    import jax
    import jax.numpy as jnp
    from bnn.trees import TreeLayout

    rng = np.random.default_rng(seed)
    x = rng.standard_normal((40, 2)).astype(np.float32)
    y = (np.sin(x[:, 0]) + 0.1 * rng.standard_normal(40)).astype(np.float32)
    tree = {
        "params": {
            "fcn": {
                "layer0": {
                    "bias": np.zeros(4, np.float32),
                    "kernel": rng.standard_normal((2, 4)).astype(np.float32),
                },
                "layer1": {
                    "bias": np.zeros(1, np.float32),
                    "kernel": rng.standard_normal((4, 1)).astype(np.float32),
                },
            }
        },
        "sigma": np.float32(0.0),
    }
    layout = TreeLayout.from_tree(tree)
    target = posterior(smoke_model(), layout, x[:32], y[:32])
    gradient = jax.jit(jax.grad(target["potential"]))
    chains = []
    for _ in range(2):
        state = layout.flatten(tree).astype(np.float64)
        draws = []
        for _ in range(96):
            noise = rng.standard_normal(len(state))
            state = state - 1e-3 * np.asarray(gradient(jnp.asarray(state)))
            state = state + np.sqrt(2e-3) * noise
            draws.append(state.astype(np.float32))
        chains.append(history(np.stack(draws), spec))
    return "synthetic", target, (x[32:], y[32:]), chains


def run_problem(cell):
    """The run's posterior, its test split and the history of each chain."""
    from bnn.pipeline.samples import load_raw

    run = cell.run("run")
    root = run.directory.root
    loader = run.dataset()
    x, y = (np.asarray(split) for split in loader.get_split("train"))
    test_x, test_y = (np.asarray(split) for split in loader.get_split("test"))
    chosen = np.linspace(
        0, len(test_y) - 1, min(cell.factors["test_points"], len(test_y))
    ).astype(int)
    first = load_raw(root, draw_indices=(0,))
    target = posterior(run.model(), first.layout, x, y)
    chains = [
        history(load_raw(root, chain_ids=(chain,)).parameters[0], cell.factors)
        for chain in first.chain_ids
    ]
    return cell.factors["dataset"], target, (test_x[chosen], test_y[chosen]), chains


def seconds_of(call):
    """Fastest of three calls after compilation."""
    call()
    times = []
    for _ in range(3):
        start = perf_counter()
        call()
        times.append(perf_counter() - start)
    return min(times)


def movement(gradients, theta, prediction):
    """Mean squared displacement of weights and predictions at doubling record lags.

    One row per lag: the mean gradients between the two records, the squared
    distance of the weights, and the mean squared change of a prediction.
    """
    theta, prediction = theta.astype(np.float64), prediction.astype(np.float64)
    rows, lag = [], 1
    while lag <= len(gradients) // 2:
        rows.append(
            [
                np.mean(gradients[lag:] - gradients[:-lag]),
                np.mean(np.sum((theta[lag:] - theta[:-lag]) ** 2, axis=1)),
                np.mean((prediction[lag:] - prediction[:-lag]) ** 2),
            ]
        )
        lag *= 2
    return np.array(rows)


def measure(cell):
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    spec = cell.factors
    if spec["profile"] == "smoke":
        dataset, target, test, chains = smoke_problem(cell.seeds["chains"], spec)
    else:
        dataset, target, test, chains = run_problem(cell)
    source, structure, adaptive = ARMS[spec["arm"]]
    count, dimension = len(chains), target["dimension"]
    test_inputs = jnp.asarray(test[0], jnp.float64)
    potentials = jax.jit(jax.vmap(target["potential"]))
    predictions = jax.jit(jax.vmap(lambda flat: target["mean"](flat, test_inputs)))

    def factor(points, variance, draws):
        if source is None:
            return np.ones(dimension)
        if source == "draws":
            return spread_factor(variance, draws)
        metric = mean_metric(target["metric"], points)
        if structure == "diagonal":
            return diagonal_factor(metric)
        if structure == "blocks":
            return block_factor(metric, target["blocks"])
        if structure == "low_rank":
            return low_rank_factor(metric, spec["rank"])
        return dense_factor(metric)

    def spaced(points):
        index = np.linspace(0, len(points) - 1, min(spec["references"], len(points)))
        return points[index.astype(int)]

    def predict(points):
        points = np.asarray(points, np.float64)
        return np.concatenate(
            [
                np.asarray(predictions(jnp.asarray(points[first : first + 256])))
                for first in range(0, len(points), 256)
            ]
        ).astype(np.float32)

    build, run = {
        "mclmc": (kernels.mclmc, kernels.run_mclmc),
        "nuts": (kernels.nuts, kernels.run_nuts),
    }[spec["sampler"]]
    functions = build(target["potential"], spec["settings"])
    centers = np.stack([chain["start"] for chain in chains])
    current = [
        factor(chain["references"], chain["variance"], chain["count"])
        for chain in chains
    ]
    evaluations = spec["references"] if source == "fisher" else 0
    step_sizes = np.full(count, spec["settings"]["step_size"])
    used = np.zeros(count, dtype=np.int64)
    key = jax.random.key(cell.seeds["chains"])
    names = ("chain", "segment", "gradients", "potential", "log_sigma", "prediction")
    records = {name: [] for name in names}
    segments = len(spec["boundaries"])
    diagnostics, shape = {}, (count, segments)
    ends = np.zeros((count, segments, dimension), np.float32)
    moved = []
    started = perf_counter()
    for segment, boundary in enumerate(spec["boundaries"]):
        key, segment_key = jax.random.split(key)
        rows = run(
            functions,
            jnp.asarray(centers),
            jnp.asarray(np.stack(current)),
            jnp.asarray(step_sizes),
            boundary - used,
            segment_key,
            spec["settings"],
        )
        last = segment == segments - 1
        for chain, row in enumerate(rows):
            size = len(row["potential"])
            prediction = predict(row["position"])
            records["chain"].append(np.full(size, chain))
            records["segment"].append(np.full(size, segment))
            records["gradients"].append(used[chain] + row["gradients"])
            records["potential"].append(row["potential"])
            records["log_sigma"].append(row["position"][:, target["scale"]])
            records["prediction"].append(prediction)
            for name in set(row) & set(READINGS):
                records.setdefault(name, []).append(row[name])
            for name in set(row) & set(DIAGNOSTICS):
                diagnostics.setdefault(name, np.zeros(shape))[chain, segment] = row[
                    name
                ]
            if last:
                # With a step size that adapts first, only its frozen part counts.
                kept = row.get("frozen", np.ones(size)) == 1
                kept = kept if kept.sum() >= 2 else np.ones(size, bool)
                moved.append(
                    movement(
                        row["gradients"][kept], row["position"][kept], prediction[kept]
                    )
                )
            centers[chain] = ends[chain, segment] = row["position"][-1]
            step_sizes[chain] = row["step_size"]
            used[chain] += row["used"]
            if adaptive and not last:
                current[chain] = factor(
                    spaced(row["position"]), row["variance"], row["count"]
                )
        if adaptive and not last and source == "fisher":
            evaluations += np.mean(
                [min(spec["references"], len(row["potential"])) for row in rows]
            )
        diagnostics.setdefault("used", np.zeros(shape))[:, segment] = used
    sampling_seconds = perf_counter() - started

    point = jnp.asarray(centers[0])
    gradient = jax.jit(jax.value_and_grad(target["potential"]))
    seconds = {
        "gradient": seconds_of(lambda: jax.block_until_ready(gradient(point))),
        "metric": seconds_of(lambda: jax.block_until_ready(target["metric"](point))),
        "sampling": sampling_seconds,
    }
    records = {name: np.concatenate(rows) for name, rows in records.items()}
    final = records["segment"] == segments - 1
    rungs = min(len(curve) for curve in moved)
    atomic_npz(
        cell_folder() / ARTIFACT,
        **records,
        # (chain, lag, [gradients, weights, predictions]) in the last segment.
        movement=np.stack([curve[:rungs] for curve in moved]),
        # Position of each chain at the end of each segment.
        segment_theta=ends,
        test_targets=np.asarray(test[1], np.float64),
        stored_index=np.stack([chain["stored_index"] for chain in chains]),
        stored_potential=np.stack(
            [np.asarray(potentials(jnp.asarray(chain["stored"]))) for chain in chains]
        ),
        stored_prediction=np.stack(
            [predict(chain["stored_recent"]) for chain in chains]
        ),
        stored_log_sigma=np.stack(
            [chain["stored_recent"][:, target["scale"]] for chain in chains]
        ),
        **{f"segment_{name}": values for name, values in diagnostics.items()},
    )
    return {
        "metrics": {
            "dataset": dataset,
            "sampler": spec["sampler"],
            "arm": spec["arm"],
            "parameters": dimension,
            "observations": target["observations"],
            "chains": count,
            "gradients": float(used.mean()),
            # Local metrics evaluated per chain.
            "metric_evaluations": float(evaluations),
            "final_potential": float(np.median(records["potential"][final])),
            "final_step_size": float(np.median(step_sizes)),
            "seconds": seconds,
        }
    }
