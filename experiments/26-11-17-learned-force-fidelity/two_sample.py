"""Independent-draw two-sample diagnostics with held-out classifier evaluation.

Classifier accuracy is a finite-power diagnostic, not a posterior qualification.
Every classifier choice and normalization uses training data only. Numerical
failures retain their original denominators and suppress conditional-on-survival
distribution scores.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import cdist


def _matrix(values, label):
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or not result.shape[1]:
        raise ValueError(f"{label} must have shape (draw, feature>=1)")
    return result


def _split_indices(count, rng):
    if count < 8:
        raise ValueError("C2ST requires at least eight independent draws per law")
    order = rng.permutation(count)
    return order[: count // 2], order[count // 2 :]


def _standardize(training, *evaluation):
    mean = training.mean(axis=0)
    scale = training.std(axis=0)
    # Constant training coordinates keep unit scale. Held-out variation cannot
    # determine a scale, and no evaluation information enters this calculation.
    scale = np.where(scale > 1e-12, scale, 1.0)
    return tuple((values - mean) / scale for values in (training, *evaluation))


def _bandwidth(training):
    squared = cdist(training, training, metric="sqeuclidean")
    upper = squared[np.triu_indices(len(training), k=1)]
    positive = upper[upper > 0]
    return float(np.median(positive)) if positive.size else 1.0


def _kernel(first, second, kind, bandwidth, multiplier):
    if kind == "linear":
        return first @ second.T / first.shape[1]
    return np.exp(
        -cdist(first, second, metric="sqeuclidean") / (2.0 * bandwidth * multiplier**2)
    )


def _fit_predict(training, labels, evaluation, kind, multiplier, ridge):
    bandwidth = _bandwidth(training) if kind == "rbf" else 1.0
    gram = _kernel(training, training, kind, bandwidth, multiplier)
    # The ridge multiplies n because the data-fit term is an empirical average.
    gram.flat[:: len(training) + 1] += len(training) * ridge
    coefficients = cho_solve(cho_factor(gram, lower=True), labels)
    return _kernel(evaluation, training, kind, bandwidth, multiplier) @ coefficients


def _accuracy(scores, reference_count):
    return 0.5 * (
        np.mean(scores[:reference_count] < 0) + np.mean(scores[reference_count:] >= 0)
    )


def kernel_ridge_c2st(reference, candidate, *, seed=0):
    """Nested-choice linear/RBF ridge C2ST on independently shuffled law halves.

    Half of each law is reserved for one final evaluation. Kernel family, RBF
    bandwidth multiplier, and ridge are selected on a further stratified split
    of the training halves, then refitted on those full training halves. Returned
    accuracy is the balanced held-out accuracy without folding values below 0.5.
    """
    reference = _matrix(reference, "reference")
    candidate = _matrix(candidate, "candidate")
    if reference.shape[1] != candidate.shape[1]:
        raise ValueError("Two-sample feature dimensions differ")
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError(
            "C2ST requires finite arrays; use compare_samples for failures"
        )
    rng = np.random.default_rng(seed)
    ref_train, ref_test = _split_indices(len(reference), rng)
    cand_train, cand_test = _split_indices(len(candidate), rng)
    ref_inner, ref_valid = np.array_split(rng.permutation(ref_train), 2)
    cand_inner, cand_valid = np.array_split(rng.permutation(cand_train), 2)
    inner, validation = _standardize(
        np.concatenate([reference[ref_inner], candidate[cand_inner]]),
        np.concatenate([reference[ref_valid], candidate[cand_valid]]),
    )
    inner_labels = np.concatenate([-np.ones(len(ref_inner)), np.ones(len(cand_inner))])
    choices = [
        (kind, multiplier, ridge)
        for kind, multiplier in (
            ("linear", 1.0),
            ("rbf", 0.5),
            ("rbf", 1.0),
            ("rbf", 2.0),
        )
        for ridge in (0.1, 0.01, 0.001)
    ]
    scored = [
        _accuracy(
            _fit_predict(inner, inner_labels, validation, *choice), len(ref_valid)
        )
        for choice in choices
    ]
    best = int(np.argmax(scored))
    kind, multiplier, ridge = choices[best]
    training, test = _standardize(
        np.concatenate([reference[ref_train], candidate[cand_train]]),
        np.concatenate([reference[ref_test], candidate[cand_test]]),
    )
    labels = np.concatenate([-np.ones(len(ref_train)), np.ones(len(cand_train))])
    scores = _fit_predict(training, labels, test, kind, multiplier, ridge)
    return {
        "accuracy": float(_accuracy(scores, len(ref_test))),
        "kernel": kind,
        "bandwidth_multiplier": multiplier,
        "ridge": ridge,
        "validation_accuracy": float(scored[best]),
        "seed": int(seed),
        "training_draws": [len(ref_train), len(cand_train)],
        "test_draws": [len(ref_test), len(cand_test)],
        "reference_test_accuracy": float(np.mean(scores[: len(ref_test)] < 0)),
        "candidate_test_accuracy": float(np.mean(scores[len(ref_test) :] >= 0)),
    }


def moment_discrepancies(reference, candidate):
    """Population moments normalized by the reference's total variance.

    Covariances use denominator n. Zero reference spread makes relative scores
    undefined (None), while the unnormalized mean RMSE remains available.
    """
    reference, candidate = (
        _matrix(reference, "reference"),
        _matrix(candidate, "candidate"),
    )
    if reference.shape[1] != candidate.shape[1] or not min(
        len(reference), len(candidate)
    ):
        raise ValueError(
            "Moment comparison needs nonempty arrays with equal dimensions"
        )
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("Moments require finite arrays")
    ref_mean, cand_mean = reference.mean(0), candidate.mean(0)
    ref_centered, cand_centered = reference - ref_mean, candidate - cand_mean
    ref_cov = ref_centered.T @ ref_centered / len(reference)
    cand_cov = cand_centered.T @ cand_centered / len(candidate)
    spread = float(np.trace(ref_cov))
    covariance_scale = float(np.linalg.norm(ref_cov))
    mean_norm = float(np.linalg.norm(cand_mean - ref_mean))
    with np.errstate(over="ignore", invalid="ignore"):
        values = {
            "mean_rmse": mean_norm / np.sqrt(reference.shape[1]),
            "mean_error_reference_sd": mean_norm / np.sqrt(spread)
            if spread > 0
            else None,
            "covariance_relative_error": float(np.linalg.norm(cand_cov - ref_cov))
            / covariance_scale
            if covariance_scale > 0
            else None,
            "sd_ratio": float(np.sqrt(np.trace(cand_cov) / spread))
            if spread > 0
            else None,
            "reference_rms_sd": float(np.sqrt(spread / reference.shape[1])),
            "candidate_rms_sd": float(np.sqrt(np.trace(cand_cov) / candidate.shape[1])),
        }
    return {
        key: float(value) if value is not None and np.isfinite(value) else None
        for key, value in values.items()
    }


def _failures(values, supplied):
    bad = ~np.isfinite(values).all(axis=1)
    if supplied is not None:
        supplied = np.asarray(supplied)
        if supplied.dtype.kind != "b" or supplied.shape != (len(values),):
            raise ValueError(
                "Failure masks must be Boolean vectors on the full draw axis"
            )
        bad |= supplied
    return bad


def _tail_counts(reference, candidate, ref_bad, cand_bad):
    if ref_bad.any():
        return None
    center = reference.mean(0)
    scale = np.maximum(reference.std(0), 1e-12)
    with np.errstate(over="ignore", invalid="ignore"):
        reference_tail = np.any(np.abs((reference - center) / scale) > 10, axis=1)
        candidate_tail = np.any(np.abs((candidate - center) / scale) > 10, axis=1)
    candidate_tail &= ~cand_bad
    return {
        "reference_ten_sd_tail_count": int(reference_tail.sum()),
        "candidate_ten_sd_tail_count": int(candidate_tail.sum()),
        "candidate_tail_or_failure_count": int(
            np.count_nonzero(candidate_tail | cand_bad)
        ),
        "candidate_tail_or_failure_fraction": float(np.mean(candidate_tail | cand_bad)),
        "sd_floor": 1e-12,
    }


def compare_samples(
    reference, candidate, *, seed=0, failures=None, reference_failures=None
):
    """Report failures on every draw; never score a survivor-selected population."""
    reference, candidate = (
        _matrix(reference, "reference"),
        _matrix(candidate, "candidate"),
    )
    if reference.shape[1] != candidate.shape[1] or not min(
        len(reference), len(candidate)
    ):
        raise ValueError(
            "Comparison needs nonempty arrays with equal feature dimensions"
        )
    ref_bad, cand_bad = (
        _failures(reference, reference_failures),
        _failures(candidate, failures),
    )
    complete = not (ref_bad.any() or cand_bad.any())
    return {
        "reference_draws": len(reference),
        "candidate_draws": len(candidate),
        "reference_failures": int(ref_bad.sum()),
        "candidate_failures": int(cand_bad.sum()),
        "reference_failure_fraction": float(ref_bad.mean()),
        "candidate_failure_fraction": float(cand_bad.mean()),
        "finite_complete": bool(complete),
        "tails": _tail_counts(reference, candidate, ref_bad, cand_bad),
        "c2st": kernel_ridge_c2st(reference, candidate, seed=seed)
        if complete and min(len(reference), len(candidate)) >= 8
        else None,
        "moments": moment_discrepancies(reference, candidate) if complete else None,
    }
