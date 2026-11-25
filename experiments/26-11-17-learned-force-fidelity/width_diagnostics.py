"""Geometry and Monte Carlo diagnostics with explicit applicability checks."""

from __future__ import annotations

import math

import numpy as np
from bnn.evaluation.diagnostics import chain_diagnostics
from scipy.stats import kstest


def chain_health(values, names=None):
    """Shared rank/folded diagnostics, with the width study's qualification gate."""
    values = np.asarray(values, dtype=float)
    if values.ndim == 2:
        values = values[..., None]
    if values.ndim != 3 or values.shape[0] < 2 or values.shape[1] < 8:
        raise ValueError("At least two chains and eight draws required")
    if not np.isfinite(values).all():
        raise ValueError("Diagnostic values must be finite")
    record = chain_diagnostics(values)
    names = names or [str(i) for i in range(values.shape[-1])]
    if len(names) != values.shape[-1]:
        raise ValueError("One diagnostic name is required per coordinate")
    return {
        "max_rhat": record["rhat_max"],
        "min_bulk_ess": record["bulk_ess_min"],
        "min_tail_ess": record["tail_ess_min"],
        "passed": record["rhat_max"] < 1.01
        and record["bulk_ess_min"] >= 400
        and record["tail_ess_min"] >= 400,
        "coordinates": {
            name: {"rhat": r, "bulk_ess": b, "tail_ess": t}
            for name, r, b, t in zip(
                names,
                record["rhat"],
                record["bulk_ess"],
                record["tail_ess"],
                strict=True,
            )
        },
        "interpretation": "Diagnostics for the measured observables; they do not establish global posterior exploration.",
    }


def project_force(jacobian, vector, rtol=1e-8):
    """Project a parameter-space vector onto row(J)=im(J^T) and ker(J)."""
    _, singular, vh = np.linalg.svd(
        np.asarray(jacobian, dtype=np.float64), full_matrices=False
    )
    rank = int((singular > (singular.max() * rtol)).sum()) if singular.size else 0
    basis = vh[:rank]
    vector = np.asarray(vector, dtype=np.float64)
    flat = vector.reshape(-1)
    row = (flat @ basis.T) @ basis
    return row.reshape(vector.shape), (flat - row).reshape(vector.shape), rank


def spectrum(hidden):
    """Centered token spectrum. This is NOT a data-Jacobian rank estimate."""
    hidden = np.asarray(hidden, dtype=np.float64)
    singular = np.linalg.svd(hidden - hidden.mean(0, keepdims=True), compute_uv=False)
    power = np.square(singular)
    mass = power / max(power.sum(), 1e-30)
    effective = float(np.exp(-(mass * np.log(np.maximum(mass, 1e-30))).sum()))
    return {
        "singular_values": singular.tolist(),
        "effective_rank": effective,
        "numerical_rank": int((singular > singular.max() * 1e-6).sum()),
    }


def split_geometry(q, truth, bnn, *, tolerance=0.05):
    """Many-to-one signed projection, with no forced simplex normalization.

    Dirichlet tests are meaningful only if the projection residual, group sums,
    positivity, and coverage all pass. This gate is specified before training.
    """
    q, truth = np.asarray(q, dtype=np.float64), np.asarray(truth, dtype=np.float64)
    width, true_width = q.shape[1], truth.shape[0]
    reference = truth * math.sqrt(bnn.multiplier(true_width) / bnn.multiplier(width))
    direction = reference / np.linalg.norm(reference, axis=-1, keepdims=True)
    norms = np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-15)
    assignment = ((q / norms) @ direction.T).argmax(-1)
    selected = reference[assignment]
    amplitude = (q * selected).sum(-1) / np.square(selected).sum(-1)
    residual = np.linalg.norm(
        q - amplitude[..., None] * selected, axis=(1, 2)
    ) / np.maximum(np.linalg.norm(q, axis=(1, 2)), 1e-15)
    coefficients = np.square(amplitude)
    members = assignment[..., None] == np.arange(true_width)
    counts = members.sum(1)
    sums = (coefficients[..., None] * members).sum(1)
    sum_error = np.abs(sums - 1).max(-1)
    valid = (
        (residual < tolerance)
        & (sum_error < tolerance)
        & (amplitude > 0).all(-1)
        & (counts > 0).all(-1)
    )
    summary = {
        "width": width,
        "states": len(q),
        "tolerance": tolerance,
        "relative_projection_residual_median": float(np.median(residual)),
        "group_sum_max_error_median": float(np.median(sum_error)),
        "positive_amplitude_fraction": float((amplitude > 0).mean()),
        "all_groups_represented_fraction": float((counts > 0).all(-1).mean()),
        "eligible_states": int(valid.sum()),
        "eligible_fraction": float(valid.mean()),
        "dirichlet_test": "not applicable"
        if not valid.any()
        else "conditional moments only; chain dependence prevents iid KS calibration",
    }
    return summary, {
        "assignment": assignment,
        "amplitude": amplitude,
        "coefficients": coefficients,
        "counts": counts,
        "eligible": valid,
        "projection_residual": residual,
    }


def attention_alignment(attention, assignment, *, permutations=99, seed=0):
    """Off-diagonal block contrast against a label-permutation null."""
    matrix = np.asarray(attention, dtype=float)
    labels = np.asarray(assignment)
    off = ~np.eye(len(labels), dtype=bool)

    def contrast(group):
        same = (group[:, None] == group[None, :]) & off
        different = (group[:, None] != group[None, :]) & off
        if not same.any() or not different.any():
            return None
        # Relative contrast removes the trivial 1/M decay of row-softmax entries.
        return float(
            (matrix[same].mean() - matrix[different].mean())
            / max(matrix[off].mean(), 1e-15)
        )

    observed = contrast(labels)
    if observed is None:
        return {"contrast": None, "permutation_p": None}
    rng = np.random.default_rng(seed)
    null = np.array([contrast(rng.permutation(labels)) for _ in range(permutations)])
    return {
        "contrast": observed,
        "permutation_p": float((1 + (null >= observed).sum()) / (permutations + 1)),
    }


def geometric_audit(seed=141026, draws=100000):
    """Reproducible counterchecks independent of fitted students.

    A Euclidean tube about two collinear split blocks is a tube about a quarter
    circle in a fixed 2-plane. Interior projections have uniform angle under an
    isotropic ambient Gaussian, hence Beta(1/2,1/2), for every block dimension.
    The separate angular-cone conditioning weights angles by (cos sin)^p and
    gives Beta((p+1)/2,(p+1)/2). They are different conditionings.
    """
    rng = np.random.default_rng(seed)
    angle = rng.uniform(0, np.pi / 2, draws)
    c = np.cos(angle) ** 2
    p = 5
    cone = rng.beta((p + 1) / 2, (p + 1) / 2, draws)
    # Direct ambient rejection check in R^4, not samples drawn from a beta law.
    ambient = rng.normal(size=(1000000, 2, 2))
    radial = ambient[:, :, 0]
    norm = np.linalg.norm(radial, axis=1)
    distance_sq = (norm - 1) ** 2 + np.square(ambient[:, :, 1]).sum(1)
    accepted = (radial > 0).all(1) & (distance_sq < 0.2**2)
    ambient_c = radial[accepted, 0] ** 2 / norm[accepted] ** 2
    return {
        "seed": seed,
        "draws": draws,
        "euclidean_tube_interior": {
            "mean_c": float(c.mean()),
            "variance_c": float(c.var()),
            "beta_half_ks": float(kstest(c, "beta", args=(0.5, 0.5)).statistic),
            "beta_three_ks": float(kstest(c, "beta", args=(3, 3)).statistic),
            "predicted_variance": 1 / 8,
        },
        "fixed_angular_cone": {
            "variance_c": float(cone.var()),
            "predicted_variance": 1 / 28,
        },
        "ambient_gaussian_rejection": {
            "ambient_dimension": 4,
            "epsilon": 0.2,
            "accepted": int(accepted.sum()),
            "variance_c": float(ambient_c.var()),
            "beta_half_ks": float(kstest(ambient_c, "beta", args=(0.5, 0.5)).statistic),
            "paper_p_one_beta_one_ks": float(
                kstest(ambient_c, "beta", args=(1, 1)).statistic
            ),
            "scope": "Positive radial components, hence interior projections; end-cap mass vanishes as epsilon tends to zero.",
        },
        "gaussian_step_crossing_probability": float(
            0.5 * math.erfc(0.1 / math.sqrt(4 * 0.01))
        ),
        "crossing_example": "A scalar at 0.1 with zero drift and Gaussian ULA noise of variance 0.02 crosses zero with positive probability.",
    }
