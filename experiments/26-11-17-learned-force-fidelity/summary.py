"""Full-panel integrity and across-fit scientific summaries."""

from collections import Counter, defaultdict

import numpy as np


def validate_panel(summary, cells, rows, consumed_cell_ids):
    preparation = summary.get("preparation", {})
    recipe = preparation.get("recipe", {})
    expected = preparation.get("expected_rows")
    if (
        recipe.get("profile") != "production"
        or not isinstance(expected, dict)
        or len(expected) != 342
    ):
        raise ValueError(
            "Learned-sampler publication requires the frozen 342-row production recipe"
        )
    required_counts = {
        "teacher": 30,
        "zero": 30,
        "exact": 30,
        "primary": 72,
        "boundary": 54,
        "composed": 54,
        "generic": 36,
        "odd_generic": 36,
    }
    if Counter(value["method"] for value in expected.values()) != required_counts:
        raise ValueError("Declared learned-sampler panel omits a method or replication")
    if len(cells) != len(rows):
        raise ValueError("Learned-sampler manifest and measured cell counts differ")
    if [cell["cell_id"] for cell in cells] != consumed_cell_ids:
        raise ValueError(
            "Learned-sampler publication cell order differs from the summary"
        )
    pairs = []
    for cell, measurement in zip(cells, rows, strict=True):
        artifact = measurement["artifact"]
        for row in measurement["rows"]:
            factors = row["factors"]
            row_id = factors["row_id"]
            group = cell["factors"]
            if (
                row_id not in expected
                or any(
                    factors.get(key) != value for key, value in expected[row_id].items()
                )
                or any(
                    factors.get(key) != value
                    for key, value in group.items()
                    if key != "horizons"
                )
                or factors["horizon"] not in group["horizons"]
            ):
                raise ValueError(
                    "Learned-sampler measurements contradict the declared factors"
                )
            if (
                row["source"]["arrays_sha256"] != artifact["sha256"]
                or row["source"]["source_identity"] != preparation["source_identity"]
            ):
                raise ValueError("Learned-sampler measured artifact identity differs")
            pairs.append((cell, row))
    ids = [row["factors"]["row_id"] for _, row in pairs]
    if len(ids) != 342 or len(set(ids)) != 342 or set(ids) != set(expected):
        raise ValueError("Learned-sampler publication is missing an endpoint row")
    for _, row in pairs:
        for space in ("parameters", "predictions"):
            comparison = row[space]
            count, failed = (
                comparison["candidate_draws"],
                comparison["candidate_failures"],
            )
            if (
                count != recipe["draws"]
                or not 0 <= failed <= count
                or comparison["candidate_failure_fraction"] != failed / count
            ):
                raise ValueError(
                    "Learned-sampler failure counts lost the full draw denominator"
                )
            if (failed or comparison["reference_failures"]) and (
                comparison["c2st"] is not None or comparison["moments"] is not None
            ):
                raise ValueError(
                    "Failed populations cannot publish survivor-only distribution scores"
                )
    return preparation, pairs


def _fit_summary(rows, extract):
    by_fit = defaultdict(list)
    for row in rows:
        value = extract(row)
        if value is None:
            return None
        by_fit[row["factors"]["fit_seed"]].append(value)
    values = {
        str(seed): float(np.mean(points)) for seed, points in sorted(by_fit.items())
    }
    return {
        "mean": float(np.mean(list(values.values()))),
        "fit_min": min(values.values()),
        "fit_max": max(values.values()),
        "fit_means": values,
    }


def aggregate_rows(rows):
    """Average starts within fits; any failed panel suppresses the group score."""
    groups = defaultdict(list)
    for row in rows:
        factors = row["factors"]
        groups[(factors["method"], factors["architecture"], factors["horizon"])].append(
            row
        )
    result = []
    for (method, architecture, horizon), selected in sorted(groups.items()):
        output = {"method": method, "architecture": architecture, "horizon": horizon}
        output["split"] = selected[0]["factors"]["split"]
        output["source_family"] = selected[0]["factors"]["source_family"]
        output["fit_count"] = len({r["factors"]["fit_seed"] for r in selected})
        output["panel_count"] = len(selected)
        for space in ("parameters", "predictions"):
            comparisons = [row[space] for row in selected]
            draws = sum(row["candidate_draws"] for row in comparisons)
            failures = sum(row["candidate_failures"] for row in comparisons)
            data = {
                "draws": draws,
                "failures": failures,
                "failure_fraction": failures / draws,
                "reference_failures": sum(
                    row["reference_failures"] for row in comparisons
                ),
                "c2st": _fit_summary(
                    selected,
                    lambda r, space=space: (
                        None
                        if r[space]["c2st"] is None
                        else r[space]["c2st"]["accuracy"]
                    ),
                ),
            }
            for measure in (
                "mean_error_reference_sd",
                "covariance_relative_error",
                "sd_ratio",
            ):
                data[measure] = _fit_summary(
                    selected,
                    lambda r, space=space, measure=measure: (
                        None
                        if r[space]["moments"] is None
                        else r[space]["moments"][measure]
                    ),
                )
            tails = [row["tails"] for row in comparisons]
            data["ten_sd_tail_count"] = (
                sum(row["candidate_ten_sd_tail_count"] for row in tails)
                if all(row is not None for row in tails)
                else None
            )
            data["tail_or_failure_count"] = (
                sum(row["candidate_tail_or_failure_count"] for row in tails)
                if all(row is not None for row in tails)
                else None
            )
            output[space] = data
        output["force_path_action"] = _fit_summary(
            selected,
            lambda r: (
                None
                if r["force_path_action"] is None
                else r["force_path_action"]["mean"]
            ),
        )
        result.append(output)
    return result
