"""One cell per run: what its chains say about proposals and metrics.

Every measurement reads stored draws and evaluates the run's own
posterior; nothing is sampled. Along each chain the cell asks how well a
Gaussian fitted to the chain's draws, or centred on one of them, works as a
proposal, how much of the local curvature a constant metric recovers, and what
a random-walk proposal gains from each metric.
"""

from itertools import combinations
from time import perf_counter

import numpy as np
from runner import CellSpec, Design, ResourceHint, atomic_npz, cell_folder

from . import curvature, proposals

QUESTION = (
    "Where do BNN posteriors sit on the axes that decide whether a learned "
    "sampler component pays?"
)
DATASETS = ("airfoil", "energy", "bikesharing")
# The `bnn` configuration of each dataset; cells read its run.
CONFIGS = {
    dataset: f"configs/trajectory/uci_{dataset}_mile_8x100000.yaml"
    for dataset in DATASETS
}
ARTIFACT = "conditions.npz"
CONSTANT = (
    "identity",
    "mean_diagonal",
    "mean",
    "mean_early",
    "covariance_diagonal",
    "covariance",
)
WALKS = (*CONSTANT, "local")


def factors(profile):
    smoke = profile == "smoke"
    return {
        "profile": profile,
        # Chain draws and proposals scored per fitted Gaussian.
        "evaluations": 64 if smoke else 1024,
        # Ranks of the fitted Gaussians; the full rank is always added.
        "ranks": [0, 4] if smoke else [0, 16, 64, 256],
        # Local metrics are evaluated at evenly spaced anchors and after each lag.
        "anchors": 2 if smoke else 4,
        "lags": [10] if smoke else [10, 100, 1000, 10000],
        # Draws whose local metrics a constant metric averages.
        "references": 4 if smoke else 32,
        "overlap_ranks": [2, 4] if smoke else [4, 16, 64],
        # Covariance scales of Gaussians centred on an anchor, in units of the
        # inverse local metric.
        "local_scales": [4.0**-k for k in range(2 if smoke else 6)],
        "local_proposals": 16 if smoke else 256,
        # Root mean square lengths of the random-walk proposals.
        "steps": [16 * 2.0**-k for k in range(3 if smoke else 15)],
        "walk_proposals": 4 if smoke else 32,
        "displacement_lags": [1, 10] if smoke else [1, 10, 100, 1000, 10000],
    }


def design():
    cells = [
        CellSpec(
            "conditions",
            {**factors("smoke"), "dataset": "synthetic"},
            {"conditions": 550000},
            expected_schema=("metrics",),
        )
    ]
    for profile, datasets in (("pilot", DATASETS[:1]), ("production", DATASETS)):
        cells.extend(
            CellSpec(
                "conditions",
                {**factors(profile), "dataset": dataset},
                {"conditions": 550001 + index},
                inputs={"run": CONFIGS[dataset]},
                expected_schema=("metrics",),
                role="pilot" if profile == "pilot" else "descriptive",
                resources=ResourceHint(cpu=4, memory_gb=8, walltime_minutes=90),
            )
            for index, dataset in enumerate(datasets)
        )
    return Design(
        QUESTION, ("smoke", "pilot", "production"), ("conditions",), tuple(cells)
    )


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


def smoke_problem(seed):
    """A small regression network with short unadjusted Langevin chains."""
    import jax.numpy as jnp
    from bnn.trees import TreeLayout

    rng = np.random.default_rng(seed)
    x = rng.standard_normal((32, 2)).astype(np.float32)
    y = (np.sin(x[:, 0]) + 0.1 * rng.standard_normal(32)).astype(np.float32)
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
    target = curvature.target(smoke_model(), layout, x, y)
    chains = []
    for _ in range(2):
        state = layout.flatten(tree).astype(np.float64)
        draws = []
        for _ in range(96):
            _, gradient = target["gradient"](jnp.asarray(state))
            noise = rng.standard_normal(len(state))
            state = state - 1e-3 * np.asarray(gradient) + np.sqrt(2e-3) * noise
            draws.append(state.astype(np.float32))
        chains.append(lambda draws=draws: np.stack(draws))
    return "synthetic", target, chains


def run_problem(cell):
    """The run's posterior and one loader per chain of its stored draws."""
    from bnn.pipeline.samples import load_raw

    run = cell.run("run")
    root = run.directory.root
    loader = run.dataset()
    x, y = (np.asarray(split) for split in loader.get_split("train"))
    first = load_raw(root, draw_indices=(0,))
    target = curvature.target(run.model(), first.layout, x, y)
    chains = [
        lambda chain=chain: load_raw(root, chain_ids=(chain,)).parameters[0]
        for chain in first.chain_ids
    ]
    return cell.factors["dataset"], target, chains


def positions(count, spec):
    """Anchors, every draw with a local metric, and the reference draws."""
    anchors = [
        count * (2 * index + 1) // (2 * spec["anchors"])
        for index in range(spec["anchors"])
    ]
    if max(spec["lags"]) >= count // (2 * spec["anchors"]):
        raise ValueError("Lags must stay within half an anchor spacing")
    references = [
        int(count * (index + 0.25) / spec["references"])
        for index in range(spec["references"])
    ]
    return anchors, references


def later(spec):
    """Whether each anchor lies after the first quarter of the chain."""
    return [
        (2 * index + 1) / (2 * spec["anchors"]) >= 0.25
        for index in range(spec["anchors"])
    ]


def costs(target, point):
    """Seconds of one gradient, local metric, Hessian and eigendecomposition."""
    import jax
    import jax.numpy as jnp

    point = jnp.asarray(point)
    matrix = np.asarray(target["metric"](point))
    calls = {
        "gradient": lambda: jax.block_until_ready(target["gradient"](point)),
        "metric": lambda: jax.block_until_ready(target["metric"](point)),
        "hessian": lambda: jax.block_until_ready(target["hessian"](point)),
        "eigendecomposition": lambda: np.linalg.eigh(matrix),
    }
    seconds = {}
    for name, call in calls.items():
        call()
        times = []
        for _ in range(3):
            start = perf_counter()
            call()
            times.append(perf_counter() - start)
        seconds[name] = min(times)
    return seconds


def chain_conditions(target, draws, spec, rng):
    """Every measurement along one chain of float32 draws ``(draw, parameter)``."""
    import jax.numpy as jnp

    count, dimension = draws.shape
    anchors, references = positions(count, spec)

    def at(index):
        return draws[index].astype(np.float64)

    def log_posterior(points):
        return -np.asarray(target["potential"](jnp.asarray(points)))

    def metric(index):
        return np.asarray(target["metric"](jnp.asarray(at(index))))

    # The chain itself, and Gaussians fitted to all of its draws.
    chosen = np.linspace(0, count - 1, spec["evaluations"]).astype(int)
    chain_points = at(chosen)
    chain_logpost = log_posterior(chain_points)
    model = proposals.fit(draws)
    corpus = []
    for rank in (*(r for r in spec["ranks"] if r < dimension), dimension):
        proposal = proposals.gaussian(model, rank)
        points = proposals.sample(proposal, rng, spec["evaluations"])
        logpost = log_posterior(points)
        scores = proposals.independence_scores(
            logpost - proposals.log_density(proposal, points),
            chain_logpost - proposals.log_density(proposal, chain_points),
        )
        corpus.append(
            [
                chain_logpost.mean() - logpost.mean(),
                logpost.std(),
                scores["ess_fraction"],
                scores["acceptance"],
                scores["jeffreys"],
            ]
        )
    displacement = [
        np.mean(
            np.sum((draws[lag::10] - draws[:-lag:10]) ** 2, axis=1, dtype=np.float64)
        )
        for lag in spec["displacement_lags"]
    ]

    # Local metrics at the anchors and after each lag.
    evaluated = sorted(
        {anchor + lag for anchor in anchors for lag in (0, *spec["lags"])}
    )
    local = {index: metric(index) for index in evaluated}
    system = {index: curvature.eigensystem(local[index]) for index in evaluated}
    constant = curvature.constant_factors(
        [metric(index) for index in references], model["covariance"]
    )
    spreads = [
        [curvature.spread(local[index], constant[name]) for index in evaluated]
        for name in CONSTANT
    ]
    pairs = [(anchor, anchor + lag) for anchor in anchors for lag in spec["lags"]]
    pairs += list(combinations(anchors, 2))
    stale = [
        [
            *curvature.spread(local[later], curvature.local_factor(*system[earlier])),
            *(
                curvature.overlap(system[earlier][1], system[later][1], rank)
                for rank in spec["overlap_ranks"]
            ),
        ]
        for earlier, later in pairs
    ]
    hessian = []
    for anchor in anchors:
        values = np.linalg.eigvalsh(
            np.asarray(target["hessian"](jnp.asarray(at(anchor))))
        )
        hessian.append([np.sum(values < 0), values[0], values[-1]])

    # Gaussians centred on the anchors, and random walks shaped by each metric.
    gaussians, walks = [], {name: [] for name in WALKS}
    for anchor in anchors:
        values, vectors = system[anchor]
        factor = curvature.local_factor(values, vectors)
        rows = proposals.local_gaussian(
            target,
            at(anchor),
            factor,
            spec["local_scales"],
            spec["local_proposals"],
            rng,
        )
        gaussians.append([list(row.values()) for row in rows])
        for name in WALKS:
            rows = proposals.walk(
                target,
                at(anchor),
                factor if name == "local" else constant[name],
                spec["steps"],
                spec["walk_proposals"],
                rng,
                local=float(np.sum(np.log(values))) if name == "local" else None,
            )
            walks[name].append([list(row.values()) for row in rows])
    return {
        "chain_logpost": chain_logpost,
        "mean_logpost": log_posterior(model["mean"][None])[0],
        "end_to_end": np.linalg.norm(at(count - 1) - at(0)),
        "displacement": np.array(displacement),
        "corpus": np.array(corpus),
        "eigenvalues": np.stack([system[index][0] for index in evaluated]),
        "constant": np.array(spreads),
        "stale_lags": np.array([later - earlier for earlier, later in pairs]),
        "stale": np.array(stale),
        "hessian": np.array(hessian, dtype=np.float64),
        "local_gaussian": np.array(gaussians),
        "walk": np.array([walks[name] for name in WALKS]),
        "anchors": [(local[anchor], system[anchor]) for anchor in anchors],
    }


def across_chains(anchors, spec):
    """Spread and eigenspace overlap of one chain's metric under another's, per anchor."""
    rows = []
    for first, second in combinations(range(len(anchors)), 2):
        for (_, system), (matrix, other) in zip(
            anchors[first], anchors[second], strict=True
        ):
            rows.append(
                [
                    *curvature.spread(matrix, curvature.local_factor(*system)),
                    *(
                        curvature.overlap(system[1], other[1], rank)
                        for rank in spec["overlap_ranks"]
                    ),
                ]
            )
    return np.array(rows)


def measure(cell):
    import jax

    jax.config.update("jax_enable_x64", True)
    spec = cell.factors
    seed = cell.seeds["conditions"]
    if spec["profile"] == "smoke":
        dataset, target, chains = smoke_problem(seed)
    else:
        dataset, target, chains = run_problem(cell)
    results, seconds = [], None
    for index, load in enumerate(chains):
        draws = load()
        if seconds is None:
            seconds = costs(target, draws[-1].astype(np.float64))
        results.append(
            chain_conditions(target, draws, spec, np.random.default_rng([seed, index]))
        )
    anchors = [result.pop("anchors") for result in results]
    arrays = {key: np.stack([result[key] for result in results]) for key in results[0]}
    arrays["across"] = across_chains(anchors, spec)
    atomic_npz(cell_folder() / ARTIFACT, **arrays)
    return {
        "metrics": {
            "dataset": dataset,
            "parameters": target["dimension"],
            "observations": target["observations"],
            "chains": len(chains),
            "draws": int(draws.shape[0]),
            "seconds": seconds,
        }
    }
