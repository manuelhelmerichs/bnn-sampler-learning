"""A part of a joined record keeps exactly the cells it consumed."""

from runner.results import Record, part_record


def test_part_selection_keeps_only_the_consumed_cells(tmp_path):
    summary = {
        "experiment": "26-11-17-experiment",
        "cell_metrics": [{"x": 1}, {"x": 2}],
        "row_summaries": {"a": {"mean": 1}, "b": {"mean": 2}},
        "parts": {"first": {"consumed_cell_ids": ["b"]}},
    }
    manifest = {"cells": [{"cell_id": "a"}, {"cell_id": "b"}]}
    record = Record(
        "26-11-17-experiment", summary, manifest, [{"x": 1}, {"x": 2}], tmp_path
    )
    part = part_record(record, "first")
    assert part.summary["cell_metrics"] == [{"x": 2}]
    assert part.manifest["cells"] == [{"cell_id": "b"}]
    assert part.rows == [{"x": 2}]
    assert part.summary["row_summaries"] == {"b": {"mean": 2}}
