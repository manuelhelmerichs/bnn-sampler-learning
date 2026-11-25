"""The same comparison on a small tanh network with a sampled reference.

The network is that of ``bnn.target.collapsed_bnn``: one hidden layer, a
standard normal prior on every weight and on the log noise scale, 32
observations. It is sampled in two parametrizations. ``collapsed`` integrates
the Gaussian readout out exactly and samples the hidden weights and the noise.
``full`` samples every weight, as a sampler of a network does.

The reference of each dataset is sampled on the collapsed posterior: tempered
SMC from the prior, then long HMC chains from its particles. Their last states
are the starts of every arm, and their draws the corpus of the family's bound.
The primary estimand is the prediction at the test input whose posterior mean
function varies most, the secondary the squared norm of the hidden weights; both
are read in units of their posterior standard deviation.
"""

import jax
import jax.numpy as jnp
import numpy as np
from bnn.samplers.tempered_smc import SMCConfig, tempered_smc
from bnn.target.collapsed_bnn import Target
from runner import CellSpec, Design, ResourceHint, derive_seed

from . import kernels
from .arms import compare, curvature_factor
from .synthetic import QUESTION, ROLES, comparison, report, save_repeats

SEED = 630120
WIDTHS = (8, 16)
PARAMETRIZATIONS = ("collapsed", "full")
ESTIMANDS = ("prediction", "hidden_norm")


def _features(hidden, x):
    width = hidden.shape[0]
    activation = jnp.tanh(x @ hidden[:, :2].T / jnp.sqrt(2) + hidden[:, 2])
    return jnp.concatenate(
        [activation / jnp.sqrt(width), jnp.ones((len(x), 1))], axis=1
    )


def _readout(position, data):
    """Hidden weights, noise scale, and the readout's posterior given both."""
    hidden = position[:-1].reshape(-1, 3)
    sigma = 0.3 * jnp.exp(0.5 * position[-1])
    scaled = _features(hidden, data["x"]) / sigma
    chol = jnp.linalg.cholesky(jnp.eye(scaled.shape[1]) + scaled.T @ scaled)
    mean = jax.scipy.linalg.cho_solve((chol, True), scaled.T @ (data["y"] / sigma))
    return hidden, sigma, chol, mean


def log_likelihood(position, data):
    """Of the hidden weights and the log noise, the readout integrated out."""
    hidden, sigma, chol, mean = _readout(position, data)
    residual = (data["y"] - _features(hidden, data["x"]) @ mean) / sigma
    return -0.5 * (
        len(data["y"]) * (jnp.log(2 * jnp.pi) + 2 * jnp.log(sigma))
        + 2 * jnp.log(jnp.diag(chol)).sum()
        + residual @ residual
        + mean @ mean
    )


def log_prior(position, data):
    return -0.5 * position @ position


def collapsed_energy(position, data):
    return 0.5 * position @ position - log_likelihood(position, data)


def full_energy(position, data, readout):
    """Of hidden weights, log noise and the ``readout`` last weights."""
    split = position.size - readout
    hidden = position[: split - 1].reshape(-1, 3)
    sigma = 0.3 * jnp.exp(0.5 * position[split - 1])
    residual = (data["y"] - _features(hidden, data["x"]) @ position[split:]) / sigma
    return (
        0.5 * position @ position
        + len(data["y"]) * jnp.log(sigma)
        + 0.5 * residual @ residual
    )


def functionals(position, data):
    """Of a collapsed position: the mean function at the test inputs, its
    variance given the hidden weights, and the squared norm of the hidden weights."""
    hidden, _, chol, mean = _readout(position, data)
    test = _features(hidden, data["test"])
    half = jax.scipy.linalg.solve_triangular(chol, test.T, lower=True)
    return jnp.concatenate([test @ mean, jnp.sum(half**2, 0), jnp.sum(hidden**2)[None]])


def collapsed_estimand(position, data):
    hidden, _, _, mean = _readout(position, data)
    prediction = (_features(hidden, data["point"][None]) @ mean)[0]
    raw = jnp.stack([prediction, jnp.sum(hidden**2)])
    return (raw - data["mean"]) / data["scale"] + 1


def full_estimand(position, data, readout):
    split = position.size - readout
    hidden = position[: split - 1].reshape(-1, 3)
    prediction = (_features(hidden, data["point"][None]) @ position[split:])[0]
    raw = jnp.stack([prediction, jnp.sum(hidden**2)])
    return (raw - data["mean"]) / data["scale"] + 1


def complete(key, positions, data):
    """Collapsed positions with an exact draw of the readout appended."""

    def one(key, position):
        _, _, chol, mean = _readout(position, data)
        noise = jax.random.normal(key, mean.shape, position.dtype)
        readout = mean + jax.scipy.linalg.solve_triangular(chol.T, noise, lower=False)
        return jnp.concatenate([position, readout])

    return jax.vmap(one)(jax.random.split(key, len(positions)), positions)


def canonical(positions):
    """The copy of collapsed positions with nonnegative hidden biases and hidden
    units ordered by their first input weight.

    Flipping the sign of a hidden unit and permuting the units leave the
    posterior unchanged, so every position has such a copy.
    """
    lead = positions.shape[:-1]
    hidden = positions[..., :-1].reshape(*lead, -1, 3)
    hidden = hidden * jnp.where(hidden[..., 2:3] < 0, -1.0, 1.0)
    order = jnp.argsort(hidden[..., 0], axis=-1)
    hidden = jnp.take_along_axis(hidden, order[..., None], axis=-2)
    return jnp.concatenate([hidden.reshape(*lead, -1), positions[..., -1:]], -1)


def reference(spec, datasets, seed):
    """Per dataset: corpus and last states of the reference chains, and the
    moments of the functionals with what the chains say about their error."""
    settings = spec["reference"]
    count, d = len(datasets["y"]), 3 * spec["width"] + 1
    rng = np.random.default_rng(seed)
    particles = []
    for t in range(count):
        data = jax.tree.map(lambda leaf, t=t: leaf[t], datasets)
        result = tempered_smc(
            log_prior,
            log_likelihood,
            data,
            rng.normal(size=(settings["particles"], d)),
            SMCConfig(n_mcmc=settings["moves"], seed=int(rng.integers(2**31))),
        )
        particles.append(result.positions)
    particles = np.stack(particles)
    centered = particles - particles.mean(1, keepdims=True)
    factor = jnp.asarray(
        np.linalg.cholesky(
            np.einsum("tni,tnj->tij", centered, centered) / settings["particles"]
        )
    )
    chains, length = settings["chains"], settings["length"]
    kernel = kernels.hmc(
        lambda y, group: collapsed_energy(group["factor"] @ y, group["data"]),
        lambda y, group: functionals(group["factor"] @ y, group["data"]),
        length,
        spec["jitter"],
    )
    group = {"factor": factor, "data": datasets}
    picks = np.stack(
        [rng.choice(settings["particles"], chains, replace=False) for _ in range(count)]
    )
    starts = jnp.asarray(np.take_along_axis(particles, picks[..., None], 1))
    key = jax.random.key(int(rng.integers(2**32)))
    tune_key, *block_keys = jax.random.split(key, settings["blocks"] + 1)
    state = kernel["init"](
        jnp.linalg.solve(factor[:, None], starts[..., None])[..., 0], group
    )
    full = jnp.full((count, chains), 1.0)
    state, step, _ = kernel["tune"](
        jax.random.split(tune_key, (count, chains)),
        state,
        full * spec["initial_step"] * d**-0.25,
        full * settings["acceptance"],
        group,
        steps=settings["warmup"],
        thin=settings["warmup"],
    )
    totals, corpus, accepted = 0.0, [], 0.0
    for block_key in block_keys:
        state, sums, accepts = kernel["run"](
            jax.random.split(block_key, (count, chains)),
            state,
            step,
            group,
            steps=settings["block"],
        )
        totals = totals + np.asarray(sums)
        accepted = accepted + np.asarray(accepts)
        corpus.append(
            np.einsum("tij,tcj->tci", np.asarray(factor), np.asarray(state[0]))
        )
    transitions = settings["blocks"] * settings["block"]
    corpus = np.stack(corpus, 2)
    # Chain means of the functionals, and of their squares from the corpus.
    means = totals / transitions
    values = np.asarray(
        jax.jit(jax.vmap(jax.vmap(functionals, in_axes=(0, None))))(
            jnp.asarray(corpus.reshape(count, -1, d)), datasets
        )
    )
    particle_values = np.asarray(
        jax.jit(jax.vmap(jax.vmap(functionals, in_axes=(0, None))))(
            jnp.asarray(particles), datasets
        )
    )
    return {
        "corpus": corpus,
        "last": corpus[:, :, -1],
        "mean": means.mean(1),
        "variance": values.var(1),
        # Standard error of the mean from the spread of the chain means.
        "error": means.std(1, ddof=1) / np.sqrt(chains),
        "particle_mean": particle_values.mean(1),
        "acceptance": accepted.mean(1) / transitions,
        "step": np.asarray(step).mean(1),
    }


def factors(profile, parametrization, width):
    smoke = profile == "smoke"
    dimension = 3 * width + 1 + (width + 1 if parametrization == "full" else 0)
    return {
        "profile": profile,
        "family": "network",
        "parametrization": parametrization,
        "width": width,
        "dimension": dimension,
        "reference": {
            "particles": 256 if smoke else 2048,
            "moves": 4 if smoke else 16,
            "chains": 16 if smoke else 64,
            "length": 16,
            "acceptance": 0.8,
            "warmup": 64 if smoke else 500,
            "blocks": 8 if smoke else 160,
            "block": 16 if smoke else 100,
            # Corpus draws at which the mean Hessian is taken.
            "hessians": 32 if smoke else 512,
        },
    } | comparison(smoke, dimension)


def in_cell(spec, smoke):
    """The comparison within one symmetry cell, in the coordinates that whiten it.

    Corpus and starts are moved to their canonical copies. Every chain then runs
    in ``v`` with ``theta = mean + L v`` of the canonical corpus, so a flow,
    which starts as the identity, starts as the exact constant metric. This
    cell is smaller than the others: it was added after their first results.
    """
    if smoke:
        return {"frame": "cell"}
    return {
        "frame": "cell",
        "targets": 4,
        "flow_lengths": [16],
        "fit": spec["fit"] | {"snapshots": [8192, 32768]},
        "reference": spec["reference"] | {"blocks": 80},
    }


def design():
    cells = []
    for profile in ("smoke", "pilot"):
        conditions = (
            [("full", 4, None), ("full", 4, "cell")]
            if profile == "smoke"
            else [
                *((p, w, None) for w in WIDTHS for p in PARAMETRIZATIONS),
                ("full", WIDTHS[0], "cell"),
            ]
        )
        for parametrization, width, frame in conditions:
            spec = factors(profile, parametrization, width)
            if frame:
                spec = spec | in_cell(spec, profile == "smoke")
            cells.append(
                CellSpec(
                    "network",
                    spec,
                    {
                        role: derive_seed(
                            SEED,
                            f"learned-coordinates:{role}",
                            {"profile": profile, "width": width},
                        )
                        for role in (*ROLES, "reference")
                    },
                    role="pilot",
                    expected_schema=("metrics", "estimands"),
                    resources=ResourceHint(cpu=1, memory_gb=2, walltime_minutes=300),
                )
            )
    return Design(QUESTION, ("smoke", "pilot"), (*ROLES, "reference"), tuple(cells))


def measure(cell):
    spec, seeds = cell.factors, cell.seeds
    width, count, chains = spec["width"], spec["targets"], spec["chains"]
    d, collapsed = spec["dimension"], 3 * spec["width"] + 1
    whole = spec["parametrization"] == "full"
    with jax.enable_x64():
        # The screen dataset comes first.
        sets = [Target("bnn", width, SEED + t).data() for t in range(count + 1)]
        datasets = {
            name: jnp.asarray(np.stack([s[i] for s in sets]))
            for i, name in enumerate(("x", "y", "test"))
        }
        ref = reference(spec, datasets, seeds["reference"])
        tests = datasets["test"].shape[1]
        # The test input whose mean function varies most over the posterior.
        point = ref["variance"][:, :tests].argmax(1)
        index = np.arange(count + 1)
        spread = ref["variance"][index, point]
        if whole:
            # A prediction of the whole network also carries the readout's variance.
            spread = spread + ref["mean"][index, tests + point]
        every = datasets | {
            "point": datasets["test"][index, point],
            "mean": jnp.asarray(
                np.stack([ref["mean"][index, point], ref["mean"][:, -1]], 1)
            ),
            "scale": jnp.asarray(
                np.sqrt(np.stack([spread, ref["variance"][:, -1]], 1))
            ),
            "index": jnp.arange(count + 1),
        }
        rng = np.random.default_rng([seeds["starts"]])
        key = jax.random.key(int(rng.integers(2**32)))
        corpus = jnp.asarray(ref["corpus"].reshape(count + 1, -1, collapsed))
        last = jnp.asarray(ref["last"])
        local = spec.get("frame") == "cell"
        if local:
            corpus, last = canonical(corpus), canonical(last)
        if whole:
            corpus_key, last_key = jax.random.split(key)
            corpus = jax.vmap(complete)(
                jax.random.split(corpus_key, count + 1), corpus, every
            )
            last = jax.vmap(complete)(
                jax.random.split(last_key, count + 1), last, every
            )
        energy, estimand = collapsed_energy, collapsed_estimand
        if whole:

            def energy(u, data):
                return full_energy(u, data, width + 1)

            def estimand(u, data):
                return full_estimand(u, data, width + 1)

        if local:
            center = corpus.mean(1)
            spread = corpus - center[:, None]
            whiten = jnp.linalg.cholesky(
                jnp.einsum("tni,tnj->tij", spread, spread) / corpus.shape[1]
            )
            every = every | {"center": center, "whiten": whiten}
            inner_energy, inner_estimand = energy, estimand

            def energy(v, data):
                return inner_energy(data["center"] + data["whiten"] @ v, data)

            def estimand(v, data):
                return inner_estimand(data["center"] + data["whiten"] @ v, data)

            def enter(positions):
                return jax.vmap(
                    lambda factor, center, rows: (
                        jax.scipy.linalg.solve_triangular(
                            factor, (rows - center).T, lower=True
                        ).T
                    )
                )(whiten, center, positions)

            corpus, last = enter(corpus), enter(last)
        centered = corpus - corpus.mean(1, keepdims=True)
        covariance = jnp.linalg.cholesky(
            jnp.einsum("tni,tnj->tij", centered, centered) / corpus.shape[1]
        )
        picks = np.linspace(0, corpus.shape[1] - 1, spec["reference"]["hessians"])
        hessians = jax.jit(jax.vmap(jax.vmap(jax.hessian(energy), in_axes=(0, None))))(
            corpus[:, picks.astype(int)], every
        )
        curvature = jnp.asarray(curvature_factor(np.asarray(hessians)))
        stiffest = np.linalg.eigvalsh(np.asarray(hessians).mean(1)).max(-1)
        unit = spec["initial_step"] * d**-0.25

        def rest(tree):
            return jax.tree.map(lambda leaf: leaf[1:], tree)

        def first(tree):
            return jax.tree.map(lambda leaf: leaf[0], tree)

        def draw(key, data, batch):
            return corpus[
                data["index"], jax.random.randint(key, (batch,), 0, corpus.shape[1])
            ]

        take = rng.permutation(last.shape[1])
        problem = {
            "energy": energy,
            "estimand": estimand,
            "data": rest(every),
            "screen_data": first(every),
            "screen": last[0, take[: spec["screen_chains"]]],
            "starts": last[1:, take[:chains]],
            "constant": {
                "covariance_exact": (covariance[1:], covariance[0], unit),
                "curvature_exact": (curvature[1:], curvature[0], unit),
            },
            "stiff": float(unit / np.sqrt(np.median(stiffest))),
            "unit": unit,
            "oracle": None,
            "draw": draw,
        }
        runs, fits = compare(spec, seeds, problem)
    save_repeats(runs)
    picked = np.stack([point, np.full(count + 1, -1)], 1)
    scale = np.sqrt(ref["variance"][index[:, None], picked])
    return {
        "metrics": {
            "family": "network",
            "path": spec["parametrization"],
            "shape": {"width": width},
            "dimension": d,
            "estimands": list(ESTIMANDS),
            "reference_check": {
                # In posterior standard deviations of the mean function and norm.
                "error": (ref["error"][index[:, None], picked] / scale).tolist(),
                "particles_off": (
                    (ref["particle_mean"] - ref["mean"])[index[:, None], picked] / scale
                ).tolist(),
                "acceptance": ref["acceptance"].tolist(),
                "step": ref["step"].tolist(),
                "prediction_scale": np.asarray(every["scale"][:, 0]).tolist(),
            },
        }
        | report(spec, runs, fits, seeds["bootstrap"]),
        "estimands": ["normalized_estimand_rmse", "flow_evaluation_break_even"],
    }
