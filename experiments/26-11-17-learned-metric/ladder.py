"""The metric comparison on a deep-linear posterior, along its width ladder.

A deep-linear network has the geometry of a BNN and a reference at any width:
two coefficients that the data determine, a curved fiber of weights that share
them, and a posterior that is invariant under rotations of the hidden units.
The Fisher metric of the likelihood plus the prior precision, ``I + B^T S B``
with ``B`` the Jacobian of the coefficients, has two stiff directions that turn
along the fiber.

The oracle samples the reduced coordinates: the coefficients and the logs of
the ``depth - 1`` factors of their prior variance, in which the redundancy is
gone. Exact starts in weight space come from prior draws, kept with probability
likelihood over its maximum.
"""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from bnn.target.deep_linear import DeepLinear, checked_reference
from runner import CellSpec, Design, ResourceHint, cell_folder, fingerprint, finite

from . import kernels, metrics
from .chains import reduce_cell, run_adapted, run_fixed

QUESTION = (
    "How much can a metric gain a sampler of a deep-linear posterior beyond an "
    "adapted constant one, and what may one metric evaluation cost?"
)
DEPTH = 3
# Widths 4, 8 and 16 give 28, 88 and 304 weights.
WIDTHS = (4, 8, 16)
ESTIMANDS = ("slope", "log_variance")
RUNGS = (512, 2048, 8192, 32768)
ACCEPTANCES = (0.15, 0.25, 0.45, 0.65, 0.85)
TARGETS = 8
# The divergence of the local metric takes one Jacobian-vector product per weight.
DRIFT = 304
# Prior draws per round of the rejection sampler.
ROUND = 32768
SEED = 630014
ROLES = ("starts", "screen", "chain", "adapt", "bootstrap")
WALLTIME = {4: 20, 8: 30, 16: 90}


def factors(profile, width):
    smoke = profile == "smoke"
    model = DeepLinear(DEPTH, width)
    targets = 2 if smoke else TARGETS
    return {
        "profile": profile,
        "family": "deep_linear",
        "path": "width",
        "shape": {"depth": DEPTH, "width": width},
        "dimension": model.dimension,
        "noise": model.noise,
        # The screen dataset, then one per confirmation target.
        "data_seeds": [SEED + i for i in range(targets + 1)],
        "reference_orders": [64, 128, 256],
        "targets": targets,
        "chains": 4 if smoke else 8,
        "screen_chains": 8 if smoke else 64,
        # Largest number of leapfrog steps of a transition: of the chain without
        # a metric, and of chains with one.
        "lengths": [1, 4] if smoke else [1, 4, 16, 64],
        "metric_lengths": [1, 4] if smoke else [1, 4, 16],
        "rungs": [128, 512] if smoke else list(RUNGS),
        "screen_budget": 256 if smoke else 4096,
        "warmup": 48 if smoke else 200,
        # Draws per chain that each adaptation window records, and the gradients
        # between two of them.
        "windows": [16, 32] if smoke else [100, 200, 400],
        "window_gradients": 8,
        # States per window and chain at which the curvature arm takes a local
        # metric. One takes the Jacobian of two coefficients, two gradients, which
        # is one product at ``hvp_cost``.
        "hessians": 8 if smoke else 32,
        "curvature_products": 1.0,
        # Exact draws per dataset behind the exact constant and reduced metrics.
        "exact_draws": 64 if smoke else 512,
        "acceptances": [0.45, 0.65] if smoke else list(ACCEPTANCES),
        "jitter": 0.2,
        # Initial step size times D^(1/4), in coordinates with unit stiffness.
        "initial_step": 0.5,
        "drift": model.dimension <= DRIFT,
        "hvp_cost": 2.0,
        "bootstrap": 100 if smoke else 1000,
    }


def design():
    cells = []
    for profile in ("smoke", "pilot"):
        for width in (4,) if profile == "smoke" else WIDTHS:
            cells.append(
                CellSpec(
                    "ladder",
                    factors(profile, width),
                    {
                        role: int(
                            fingerprint(
                                [profile, width, role],
                                namespace="learned-metric-ladder",
                            )[:12],
                            16,
                        )
                        for role in ROLES
                    },
                    role="pilot",
                    expected_schema=("metrics", "estimands"),
                    resources=ResourceHint(
                        memory_gb=4, walltime_minutes=WALLTIME[width]
                    ),
                )
            )
    return Design(QUESTION, ("smoke", "pilot"), ROLES, tuple(cells))


# ----------------------------------------------------------------------------
# The target


def dataset(model, orders):
    """Centre of the likelihood in the coefficients, and the estimands' reference."""
    features, y = model.data()
    reference = checked_reference(model, orders)
    if not reference["qualified"]:
        raise ValueError("The quadrature reference did not converge")
    return {
        "center": np.linalg.lstsq(features, y, rcond=None)[0],
        "reference": np.array(
            [reference["mean"][1], reference["sd"][1], *reference["log_variance"]]
        ),
    }


def log_factors(model, position):
    """Logs of the ``depth - 1`` factors whose product is ``model.variance``.

    From the last layer back, each factor is the squared norm of the product of
    the layers so far over the one before and the width. Under the prior they
    are independent ``chi2_width / width``.
    """
    offset, vector, previous, logs = model.dimension, None, 1.0, []
    for rows, cols in reversed(model.shapes[1:]):
        offset -= rows * cols
        matrix = position[offset : offset + rows * cols].reshape(rows, cols)
        vector = matrix if vector is None else vector @ matrix
        norm = jnp.sum(vector**2)
        logs.append(jnp.log(norm / (model.width * previous)))
        previous = norm
    return jnp.stack(logs)


def target(model, root):
    """Potentials and estimands of one width, in weight and in reduced coordinates.

    ``group`` holds a dataset's ``center``, the least-squares coefficients, and
    the ``reference`` of its estimands. The likelihood depends on the weights
    through ``|root^T (coefficients - center)|^2 / 2`` alone.
    """
    coefficients, width = model.coefficients, model.width

    def energy(theta, group):
        delta = root.T @ (coefficients(theta) - group["center"])
        return 0.5 * theta @ theta + 0.5 * delta @ delta

    def estimand(theta, group):
        slope, spread, log_mean, log_sd = group["reference"]
        return jnp.stack(
            [
                ((coefficients(theta)[1] - slope) / spread) ** 2,
                ((jnp.log(model.variance(theta)) - log_mean) / log_sd) ** 2,
            ]
        )

    def to_reduced(theta):
        return jnp.concatenate([coefficients(theta), log_factors(model, theta)])

    def reduced_energy(q, group):
        """Coefficients ``N(0, v I)`` given the factors of ``v``, each ``chi2 / width``."""
        beta, logs = q[:2], q[2:]
        delta = root.T @ (beta - group["center"])
        total = jnp.sum(logs)
        return (
            0.5 * delta @ delta
            + 0.5 * beta @ beta * jnp.exp(-total)
            + total
            + 0.5 * width * jnp.sum(jnp.exp(logs) - logs)
        )

    def reduced_estimand(q, group):
        slope, spread, log_mean, log_sd = group["reference"]
        return jnp.stack(
            [
                ((q[1] - slope) / spread) ** 2,
                ((jnp.sum(q[2:]) - log_mean) / log_sd) ** 2,
            ]
        )

    return SimpleNamespace(
        energy=energy,
        estimand=estimand,
        to_reduced=to_reduced,
        reduced_energy=reduced_energy,
        reduced_estimand=reduced_estimand,
    )


def exact_draws(model, center, root, wanted, generator):
    """Exact posterior draws: prior draws kept with likelihood over its maximum."""
    many = jax.jit(jax.vmap(model.coefficients))
    kept, total = [], 0
    while total < wanted:
        theta = generator.normal(size=(ROUND, model.dimension))
        delta = (np.asarray(many(jnp.asarray(theta))) - center) @ np.asarray(root)
        keep = generator.random(ROUND) < np.exp(-0.5 * np.sum(delta**2, 1))
        kept.append(theta[keep])
        total += int(keep.sum())
    return np.concatenate(kept)[:wanted]


def orbit_mean(vectors, width):
    """Mean of ``v v^T`` over ``(n, D)`` vectors and the rotations of hidden units.

    The posterior is invariant under rotations of each hidden layer. An average
    that respects this is a 2 x 2 block shared by the hidden units of the first
    layer and one scale for each later layer.
    """
    first = vectors[:, : 2 * width].reshape(len(vectors), width, 2)
    block = np.einsum("nia,nib->ab", first, first) / (len(vectors) * width)
    return (
        block,
        np.mean(vectors[:, 2 * width : -width] ** 2),
        np.mean(vectors[:, -width:] ** 2),
    )


def orbit_factor(block, middle, last, width):
    """``W`` with ``W W^T`` the matrix that ``orbit_mean`` describes."""
    d = 2 * width + width * width + width
    factor = np.zeros((d, d))
    factor[: 2 * width, : 2 * width] = np.kron(np.eye(width), np.linalg.cholesky(block))
    scales = np.concatenate([np.full(width * width, middle), np.full(width, last)])
    factor[2 * width :, 2 * width :] = np.diag(np.sqrt(scales))
    return factor


def fisher(coefficients, root, divergence):
    """The local metric ``I + B^T S B`` with ``S = root root^T``, of rank two plus one.

    ``B`` is the Jacobian of the two coefficients, so every action of the metric
    costs ``O(D)``. The divergence of its inverse takes one Jacobian-vector
    product per weight.
    """
    jacobian = jax.jacrev(coefficients)

    def local(theta, energy, group):
        stiff = root.T @ jacobian(theta)
        values, vectors = jnp.linalg.eigh(stiff @ stiff.T)
        values = jnp.maximum(values, 1e-300)
        return {"basis": (stiff.T @ vectors) / jnp.sqrt(values), "values": values}

    def powered(power):
        def act(local, x):
            gain = (1 + local["values"]) ** power - 1
            return x + local["basis"] @ (gain * (local["basis"].T @ x))

        return act

    cometric = powered(-1.0)

    def div(theta, group):
        if not divergence:
            return jnp.zeros_like(theta)
        basis = jnp.eye(theta.size, dtype=theta.dtype)

        def column(e):
            return jax.jvp(
                lambda w: cometric(local(w, None, group), e), (theta,), (e,)
            )[1]

        return jax.vmap(column)(basis).sum(0)

    return SimpleNamespace(
        local=local,
        scale=powered(-0.5),
        unscale=powered(0.5),
        cometric=cometric,
        logdet=lambda local: -0.5 * jnp.sum(jnp.log1p(local["values"])),
        divergence=div,
    )


# ----------------------------------------------------------------------------
# One cell


def measure(cell):
    spec, seeds = cell.factors, cell.seeds
    width, d = spec["shape"]["width"], spec["dimension"]
    count, chains = spec["targets"], spec["chains"]
    models = [
        DeepLinear(spec["shape"]["depth"], width, seed, spec["noise"])
        for seed in spec["data_seeds"]
    ]
    model = models[0]

    def rng(role, *index):
        return np.random.default_rng([seeds[role], *index])

    def key(role, *index):
        return jax.random.key(int(rng(role, *index).integers(2**32)))

    def keys(*index):
        return {"screen": key("screen", *index), "chain": key("chain", *index)}

    runs = {}
    with jax.enable_x64():
        # The inputs are the same for every dataset, so the precision is too.
        features = model.data()[0]
        precision = features.T @ features / spec["noise"] ** 2
        root = jnp.asarray(np.linalg.cholesky(precision))
        data = [dataset(m, spec["reference_orders"]) for m in models]
        coefficients = model.coefficients
        functions = target(model, root)
        energy, estimand = functions.energy, functions.estimand
        stiff_rows = jax.jit(
            jax.vmap(lambda theta: root.T @ jax.jacrev(coefficients)(theta))
        )
        reduce_many = jax.jit(jax.vmap(functions.to_reduced))

        # Per dataset: starts, the exact constant metrics and the reduced one.
        starts, groups, constant, reduced, stiffness = [], [], [], [], []
        for index, entry in enumerate(data):
            wanted = spec["screen_chains"] if index == 0 else chains
            draws = exact_draws(
                model,
                entry["center"],
                root,
                wanted + spec["exact_draws"],
                rng("starts", index),
            )
            starts.append(draws[:wanted])
            pool = draws[wanted:]
            rows = np.asarray(stiff_rows(jnp.asarray(pool)))
            block, middle, last = orbit_mean(rows.reshape(-1, d), width)
            # The mean of B^T S B sums the two rows that the mean ran over.
            curvature = orbit_factor(
                np.linalg.inv(np.eye(2) + 2 * block),
                1 / (1 + 2 * middle),
                1 / (1 + 2 * last),
                width,
            )
            constant.append(
                {
                    "identity": np.eye(d),
                    "covariance_exact": orbit_factor(*orbit_mean(pool, width), width),
                    "curvature_exact": curvature,
                }
            )
            coordinates = np.asarray(reduce_many(jnp.asarray(pool)))
            reduced.append(np.linalg.cholesky(np.cov(coordinates.T)))
            stiffness.append(
                np.mean(np.linalg.eigvalsh(rows @ rows.transpose(0, 2, 1))[:, -1])
            )
            groups.append({name: jnp.asarray(value) for name, value in entry.items()})
        confirm = np.stack(starts[1:])
        unit = spec["initial_step"] * d**-0.25
        stiff = unit / np.sqrt(1 + stiffness[0])

        def whitened(factors, positions, initial):
            """An arm of a whitened chain from one factor and its starts per dataset."""
            entered = [
                np.linalg.solve(f, p.T).T
                for f, p in zip(factors, positions, strict=True)
            ]
            shared = [
                g | {"factor": jnp.asarray(f)}
                for g, f in zip(groups, factors, strict=True)
            ]
            return {
                "screen": jnp.asarray(entered[0]),
                "starts": jnp.asarray(np.stack(entered[1:])),
                "screen_group": shared[0],
                "target_groups": shared[1:],
                "initial": initial,
            }

        fixed = {
            name: whitened(
                [c[name] for c in constant],
                starts,
                stiff if name == "identity" else unit,
            )
            for name in constant[0]
        }
        shared = jax.tree.map(lambda *leaves: jnp.stack(leaves), *groups[1:])

        def pooled(draws):
            return draws.reshape(count, -1, d)

        def diagonal(draws):
            draws = pooled(draws)
            size = draws.shape[1]
            variances = size / (size + 5) * draws.var(1, ddof=1) + 1e-3 * 5 / (size + 5)
            return np.stack([np.diag(np.sqrt(v)) for v in variances]), 0

        def covariance(draws):
            return metrics.covariance_factor(pooled(draws)), 0

        def curvature(draws):
            """Mean local metric of each chain, from its own states."""
            recorded = draws.shape[2]
            picks = np.linspace(recorded - 1, 0, spec["hessians"]).astype(int)
            states = draws[:, 0, picks].reshape(-1, d)
            rows = np.asarray(stiff_rows(jnp.asarray(states)))
            rows = rows.reshape(len(draws), -1, d)
            local = np.eye(d) + np.einsum("tni,tnj->tij", rows, rows) / len(picks)
            return metrics.curvature_factor(local[:, None]), len(picks)

        # Adapted arms take the target acceptance screened for an exact limit.
        adapted = {
            "diagonal": (diagonal, "covariance_exact"),
            "covariance": (covariance, "covariance_exact"),
            "curvature": (curvature, "curvature_exact"),
        }
        for length in spec["lengths"]:
            sampler = f"hmc{length}"
            metric = length in spec["metric_lengths"]
            kernel = kernels.hmc(
                lambda y, group: energy(group["factor"] @ y, group),
                lambda y, group: estimand(group["factor"] @ y, group),
                length,
                spec["jitter"],
            )
            names = list(fixed) if metric else ["identity"]
            records = run_fixed(
                kernel, [fixed[name] for name in names], length, spec, keys(length)
            )
            runs |= {(name, sampler): r for name, r in zip(names, records, strict=True)}
            if not metric:
                continue
            for a, (name, (estimate, limit)) in enumerate(adapted.items()):
                # The stiff directions differ between the chains of a target, so
                # the curvature arm keeps a metric per chain: every chain is a
                # group of its own. The other arms pool a target's chains.
                single = name == "curvature"
                run = run_adapted(
                    kernel,
                    estimate,
                    jax.tree.map(lambda leaf: jnp.repeat(leaf, chains, 0), shared)
                    if single
                    else shared,
                    jnp.asarray(confirm.reshape(-1, 1, d) if single else confirm),
                    length,
                    runs[limit, sampler]["target_acceptance"],
                    stiff,
                    spec,
                    key("adapt", length, a),
                )
                if single:
                    for field in ("estimates", "acceptance", "step"):
                        run[field] = run[field].reshape(
                            count, chains, *run[field].shape[2:]
                        )
                    # Costs divide a target's local metrics among its chains.
                    run["evaluations"] *= chains
                runs[name, sampler] = run
            kernel = kernels.hmc(
                lambda y, group: functions.reduced_energy(group["factor"] @ y, group),
                lambda y, group: functions.reduced_estimand(group["factor"] @ y, group),
                length,
                spec["jitter"],
            )
            coordinates = [np.asarray(reduce_many(jnp.asarray(s))) for s in starts]
            (runs["reduced", sampler],) = run_fixed(
                kernel,
                [whitened(reduced, coordinates, spec["initial_step"] * 4**-0.25)],
                length,
                spec,
                keys(length, 1),
            )
        local = {
            "screen": jnp.asarray(starts[0]),
            "starts": jnp.asarray(confirm),
            "screen_group": groups[0],
            "target_groups": groups[1:],
            "initial": unit,
        }
        for a, drift in enumerate([False, True] if spec["drift"] else [False]):
            kernel = kernels.pmala(
                energy, estimand, fisher(coefficients, root, drift), spec["jitter"]
            )
            (runs["fisher_drift" if drift else "fisher", "mala"],) = run_fixed(
                kernel, [local], 1, spec, keys(0, a)
            )
    measured = {
        "family": spec["family"],
        "path": spec["path"],
        "shape": spec["shape"],
        "dimension": d,
        "estimands": list(ESTIMANDS),
        "targets": count,
        "chains": chains,
        "hvp_cost": spec["hvp_cost"],
        "reference": [entry["reference"].tolist() for entry in data],
        "stiffness": float(np.mean(stiffness)),
        "curves": reduce_cell(spec, runs, seeds["bootstrap"]),
        "settings": {
            arm: {
                sampler: finite(
                    {
                        "target_acceptance": float(run["target_acceptance"]),
                        "acceptance": float(np.mean(run["acceptance"])),
                        "step_size": float(np.median(run["step"])),
                        "seconds": float(run["seconds"]),
                    }
                )
                | ({"screen_rmse": run["screen_rmse"]} if "screen_rmse" in run else {})
                for (name, sampler), run in runs.items()
                if name == arm
            }
            for arm in dict.fromkeys(name for name, _ in runs)
        },
        "repeat_file": "repeats.npz",
    }
    np.savez_compressed(
        cell_folder() / "repeats.npz",
        **{
            f"{arm}_{sampler}": run["estimates"] for (arm, sampler), run in runs.items()
        },
    )
    return {
        "metrics": measured,
        "estimands": ["normalized_estimand_rmse", "metric_evaluation_break_even"],
    }
