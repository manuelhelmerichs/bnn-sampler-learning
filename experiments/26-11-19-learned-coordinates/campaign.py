"""The flow against a constant metric on the posterior of a UCI network.

No reference exists here, and no chain is stationary: every arm continues the
chains of a ``trajectory`` run from their last stored draw, and is read by how
fast the potential falls and the test predictions improve per gradient.

Each chain has its own constant metric, the mean of the Fisher information plus
the prior precision at draws of its last quarter, and runs in the coordinates
``v`` that whiten it: ``theta = start + W v``. The classical arm is HMC in ``v``.
The flow starts as the identity in ``v``, so that before its fit it is the
classical arm, and is fitted there by the reverse divergence. Its chains pay for
the fit.
"""

from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from runner import CellSpec, Design, ResourceHint, atomic_npz, cell_folder

from . import flows, kernels
from .chains import mean_length, transitions

QUESTION = (
    "On the posterior of a UCI network, does a chain in the coordinates of a "
    "fitted flow burn in faster per gradient than one with a constant metric, "
    "once the fit is paid?"
)
DATASETS = ("airfoil",)
CONFIGS = {
    dataset: f"configs/trajectory/uci_{dataset}_mile_8x100000.yaml"
    for dataset in DATASETS
}
ARTIFACT = "chains.npz"
SEED = 630121


def factors(profile, dataset):
    smoke = profile == "smoke"
    return {
        "profile": profile,
        "dataset": dataset,
        # Share of the stored chain, at its end, that the metric is estimated from,
        # and the local metrics averaged into it.
        "history": 0.25,
        "references": 2 if smoke else 32,
        "lengths": [4] if smoke else [16],
        # Gradients of a chain after its step size is tuned, and its records.
        "budget": 256 if smoke else 32768,
        "records": 8 if smoke else 128,
        "warmup": 16 if smoke else 200,
        "acceptance": 0.8,
        "jitter": 0.2,
        "initial_step": 0.5,
        "fit": {
            "layers": 2 if smoke else 6,
            "hidden": 16 if smoke else 256,
            "snapshots": [8, 16] if smoke else [512, 4096],
            "batch": 8 if smoke else 32,
            "rate": 1e-3,
            "quality_draws": 64 if smoke else 512,
        },
        "test_points": 8 if smoke else 256,
    }


def design():
    cells = [
        CellSpec(
            "campaign",
            factors("smoke", "synthetic"),
            {"chains": SEED},
            expected_schema=("metrics",),
        )
    ]
    cells.extend(
        CellSpec(
            "campaign",
            factors("pilot", dataset),
            {"chains": SEED + 1 + index},
            inputs={"run": CONFIGS[dataset]},
            expected_schema=("metrics",),
            role="pilot",
            resources=ResourceHint(cpu=6, memory_gb=8, walltime_minutes=300),
        )
        for index, dataset in enumerate(DATASETS)
    )
    return Design(QUESTION, ("smoke", "pilot"), ("chains",), tuple(cells))


def posterior(model, layout, x, y):
    """Potential, local metric and network mean on flat float64 vectors.

    The local metric is the Fisher information of the Gaussian likelihood plus
    the prior precision: ``J'J / sigma^2`` on the weights and ``2N`` on the log
    noise scale, plus the identity of the standard normal prior.
    """
    x, y = jnp.asarray(x, jnp.float64), jnp.asarray(y, jnp.float64)
    dimension = layout.dimension
    scale = int(np.asarray(layout.unflatten(np.arange(dimension))["sigma"]))

    def potential(flat):
        return -model.log_density(layout.unflatten_traced(flat), x, y)

    def mean(flat, inputs):
        return model.apply(layout.unflatten_traced(flat), inputs)[..., 0]

    def metric(flat):
        jacobian = jax.vmap(
            jax.grad(lambda flat, row: mean(flat, row[None])[0]), in_axes=(None, 0)
        )(flat, x)
        fisher = jacobian.T @ jacobian / jnp.exp(2 * flat[scale])
        return fisher.at[scale, scale].add(2.0 * len(y)) + jnp.eye(dimension)

    return {
        "dimension": dimension,
        "scale": scale,
        "potential": potential,
        "mean": mean,
        "metric": jax.jit(metric),
    }


def history(draws, spec):
    """Start and metric references of a chain from its stored draws."""
    recent = draws[-max(2, int(len(draws) * spec["history"])) :].astype(np.float64)
    index = np.linspace(0, len(recent) - 1, min(spec["references"], len(recent)))
    return recent[-1], recent[index.astype(int)]


def smoke_problem(seed, spec):
    """A small regression network with short unadjusted Langevin chains."""
    from bnn.data.config import Task
    from bnn.models.tabular import FCN, FCNConfig
    from bnn.target import ProbabilisticModel
    from bnn.target.priors import PriorDist
    from bnn.trees import TreeLayout

    model = ProbabilisticModel(
        FCN(FCNConfig(hidden_structure=(4, 1))),
        PriorDist.StandardNormal.get_prior(),
        Task.MEAN_REGRESSION,
    )
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
    target = posterior(model, layout, x[:32], y[:32])
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
    return target, (x[32:], y[32:]), chains


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
    return target, (test_x[chosen], test_y[chosen]), chains


def measure(cell):
    spec, fit = cell.factors, cell.factors["fit"]
    with jax.enable_x64():
        if spec["profile"] == "smoke":
            target, test, chains = smoke_problem(cell.seeds["chains"], spec)
        else:
            target, test, chains = run_problem(cell)
        count, d = len(chains), target["dimension"]
        starts = np.stack([start for start, _ in chains])
        metrics = np.stack(
            [
                np.mean(
                    [np.asarray(target["metric"](jnp.asarray(p))) for p in points], 0
                )
                for _, points in chains
            ]
        )
        # W W' is the inverse of the metric.
        data = {
            "start": jnp.asarray(starts),
            "factor": jnp.asarray(np.linalg.cholesky(np.linalg.inv(metrics))),
        }
        potential = target["potential"]

        def energy(v, data):
            return potential(data["start"] + data["factor"] @ v)

        test_inputs = jnp.asarray(test[0], jnp.float64)
        test_targets = np.asarray(test[1], np.float64)
        read = jax.jit(
            jax.vmap(
                lambda v, data: (
                    energy(v, data),
                    target["mean"](data["start"] + data["factor"] @ v, test_inputs),
                    (data["start"] + data["factor"] @ v)[target["scale"]],
                    jnp.sum((data["factor"] @ v) ** 2),
                )
            )
        )
        key = jax.random.key(cell.seeds["chains"])
        key, create_key, fit_key, quality_key = jax.random.split(key, 4)
        initial = jax.vmap(
            lambda k: flows.create(k, d, fit["layers"], fit["hidden"], "index")
        )(jax.random.split(create_key, count))
        started = perf_counter()
        fitted, loss = flows.fit_reverse(
            energy,
            initial,
            data,
            fit_key,
            snapshots=fit["snapshots"],
            batch=fit["batch"],
            rate=fit["rate"],
        )
        fit_seconds = perf_counter() - started
        # Before its fit the flow is the identity: the classical arm itself.
        arms = {"classical": (None, 0)} | {
            f"flow:{steps}": (flow, steps * fit["batch"])
            for steps, flow in fitted.items()
        }
        quality = {}
        read_flows = {"flow:0": initial} | {
            name: flow for name, (flow, _) in arms.items() if flow is not None
        }
        for name, flow in read_flows.items():
            divergence, effective = flows.quality(
                energy, flow, data, quality_key, fit["quality_draws"]
            )
            z = jax.random.normal(quality_key, (count, fit["quality_draws"], d))
            drawn = jax.vmap(
                lambda flow, z, data: jax.vmap(
                    lambda z: energy(flows.forward(flow, z)[0], data)
                )(z)
            )(flow, z, data)
            quality[name] = {
                "divergence": np.asarray(divergence).tolist(),
                "effective_share": np.asarray(effective).tolist(),
                # Potential of the flow's own draws.
                "potential": np.median(np.asarray(drawn), 1).tolist(),
            }

        def pulled(z, group):
            x, logdet = flows.forward(group["flow"], z)
            return energy(x, group["data"]) - logdet

        def nothing(position, group):
            return jnp.zeros(1)

        records, settings = {}, {}
        unit = spec["initial_step"] * d**-0.25
        for length in spec["lengths"]:
            moves = transitions(spec["budget"], length)
            block = max(1, moves // spec["records"])
            for name, (flow, paid) in arms.items():
                if flow is None:
                    kernel = kernels.hmc(
                        lambda v, group: energy(v, group["data"]),
                        nothing,
                        length,
                        spec["jitter"],
                    )
                    group = {"data": data}
                else:
                    kernel = kernels.hmc(pulled, nothing, length, spec["jitter"])
                    group = {"flow": flow, "data": data}
                key, tune_key, *block_keys = jax.random.split(key, spec["records"] + 2)
                # Every chain starts at its stored draw: v = 0, and z with T(z) = 0.
                position = jnp.zeros((count, 1, d))
                if flow is not None:
                    position = jax.vmap(
                        lambda flow, v: flows.inverse(flow, v)[0][None]
                    )(flow, position[:, 0])
                started = perf_counter()
                state = kernel["init"](position, group)
                full = jnp.ones((count, 1))
                state, step, _ = kernel["tune"](
                    jax.random.split(tune_key, (count, 1)),
                    state,
                    full * unit,
                    full * spec["acceptance"],
                    group,
                    steps=spec["warmup"],
                    thin=spec["warmup"],
                )
                rows, accepted = [], 0.0
                for block_key in block_keys:
                    state, _, accepts = kernel["run"](
                        jax.random.split(block_key, (count, 1)),
                        state,
                        step,
                        group,
                        steps=block,
                    )
                    accepted = accepted + np.asarray(accepts)[:, 0]
                    v = state[0][:, 0]
                    if flow is not None:
                        v = jax.vmap(lambda flow, z: flows.forward(flow, z)[0])(flow, v)
                    rows.append([np.asarray(part) for part in read(v, data)])
                arm = f"{name}_hmc{length}"
                mean = mean_length(length)
                records[arm] = {
                    "gradients": paid
                    + 1
                    + mean
                    * (spec["warmup"] + block * np.arange(1, spec["records"] + 1)),
                    # (record, chain, ...)
                    "potential": np.stack([r[0] for r in rows]),
                    "prediction": np.stack([r[1] for r in rows]).astype(np.float32),
                    "log_sigma": np.stack([r[2] for r in rows]),
                    "distance": np.stack([r[3] for r in rows]),
                }
                settings[arm] = {
                    "fit_gradients": paid,
                    "step_size": np.asarray(step)[:, 0].tolist(),
                    "acceptance": (accepted / (block * spec["records"])).tolist(),
                    "seconds": perf_counter() - started,
                }
        start_values = read(jnp.zeros((count, d)), data)
        gradient = jax.jit(jax.value_and_grad(potential))
        point = jnp.asarray(starts[0])
        gradient(point)
        started = perf_counter()
        for _ in range(20):
            jax.block_until_ready(gradient(point))
        gradient_seconds = (perf_counter() - started) / 20
    atomic_npz(
        cell_folder() / ARTIFACT,
        test_targets=test_targets,
        start_potential=np.asarray(start_values[0]),
        start_prediction=np.asarray(start_values[1]).astype(np.float32),
        start_log_sigma=np.asarray(start_values[2]),
        loss=np.asarray(loss),
        **{
            f"{arm}__{name}": values
            for arm, record in records.items()
            for name, values in record.items()
        },
    )

    def log_density(arm, first, last):
        """Of the test targets under the pooled draws of records ``first:last``."""
        record = records[arm]
        mean = record["prediction"][first:last].reshape(-1, len(test_targets))
        sigma = np.exp(record["log_sigma"][first:last].reshape(-1, 1))
        logs = -0.5 * ((test_targets - mean) / sigma) ** 2 - np.log(
            sigma * np.sqrt(2 * np.pi)
        )
        peak = logs.max(0)
        return float(np.mean(peak + np.log(np.mean(np.exp(logs - peak), 0))))

    quarter = spec["records"] // 4
    curves = {}
    for arm, record in records.items():
        prediction = record["prediction"].astype(np.float64)
        half = spec["records"] // 2
        per = record["gradients"][1] - record["gradients"][0]
        curves[arm] = {
            "gradients": record["gradients"][quarter - 1 :: quarter].tolist(),
            # Median over chains at the end of each quarter of the run.
            "potential": np.median(record["potential"], 1)[
                quarter - 1 :: quarter
            ].tolist(),
            "test_log_density": [
                log_density(arm, q * quarter, (q + 1) * quarter) for q in range(4)
            ],
            # Mean squared change between records of the second half, per gradient.
            "prediction_movement": float(
                np.mean((prediction[half + 1 :] - prediction[half:-1]) ** 2) / per
            ),
            "distance_from_start": np.median(record["distance"], 1)[
                quarter - 1 :: quarter
            ].tolist(),
        }
    return {
        "metrics": {
            "dataset": spec["dataset"],
            "parameters": d,
            "chains": count,
            "start_potential": float(np.median(np.asarray(start_values[0]))),
            "curves": curves,
            "settings": settings,
            "fits": quality,
            "fit_seconds": fit_seconds,
            "gradient_seconds": gradient_seconds,
        }
    }
