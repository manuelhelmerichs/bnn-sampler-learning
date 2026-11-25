"""Portable records retain rejected outcomes and join derived results to rows."""

import hashlib
import json

import pytest
from runner.results import load_record


@pytest.fixture
def saved(tmp_path):
    cell = {"cell_id": "c0", "factors": {"profile": "pilot"}, "seeds": {}, "inputs": {}}
    folder = tmp_path / "cells/c0"
    folder.mkdir(parents=True)
    (folder / "artifact.bin").write_bytes(b"saved scientific output")
    metrics = {
        "quality": 0.125,
        "artifact": {
            "path": "artifact.bin",
            "sha256": hashlib.sha256(
                (folder / "artifact.bin").read_bytes()
            ).hexdigest(),
        },
    }
    (folder / "record.json").write_text(json.dumps({"state": "completed"}))
    (folder / "result.json").write_text(json.dumps({"cell": cell, "metrics": metrics}))
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {"experiment": "26-11-17-study", "profile": "pilot", "cells": [cell]}
        )
    )
    (tmp_path / "summary.json").write_text(
        json.dumps(
            {
                "experiment": "26-11-17-study",
                "profile": "pilot",
                "accepted": False,
                "cell_metrics": [metrics],
                "consumed_cell_ids": ["c0"],
                "row_summaries": {"c0": {"derived_quality": 0.25}},
            }
        )
    )
    return tmp_path


def test_rejected_pilot_record_is_readable_and_joins_derived_results(saved):
    record = load_record(saved)
    assert record.summary["accepted"] is False
    assert record.rows[0]["quality"] == 0.125
    assert record.rows[0]["derived_quality"] == 0.25
