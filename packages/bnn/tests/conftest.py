"""Shared tiny problems for the bnn test suite."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
from bnn.config import RunConfig
from bnn.pipeline.pipeline import advance
from bnn.pipeline.workers import Resources


def regression_table(path: Path, rows: int = 96, seed: int = 0) -> Path:
    """A small noisy linear regression table with the target in the last column."""
    generator = np.random.default_rng(seed)
    x = generator.normal(size=(rows, 3)).astype(np.float32)
    y = x @ np.asarray([0.8, -0.5, 0.3], np.float32) + 0.1 * generator.normal(size=rows)
    np.save(path, np.column_stack((x, y)).astype(np.float32))
    return path


def classification_table(path: Path, rows: int = 96, seed: int = 0) -> Path:
    """A small three-class table with the class index in the last column."""
    generator = np.random.default_rng(seed)
    x = generator.normal(size=(rows, 3)).astype(np.float32)
    y = np.argmax(x + 0.3 * generator.normal(size=(rows, 3)), axis=1)
    np.save(path, np.column_stack((x, y)).astype(np.float32))
    return path


BASE = {
    "name": "tiny",
    "seed": 5,
    "data": {
        "path": "",
        "task": "mean_regression",
        "normalize": True,
        "train_split": 0.6,
        "valid_split": 0.2,
        "test_split": 0.2,
    },
    "model": {"name": "FCN", "hidden_structure": [4, 1], "activation": "relu"},
    "chains": 2,
    "fit": {"name": "prior"},
    "sampler": {"name": "mclmc", "warmup": 40, "steps": 60},
    "execution": {"block_steps": 16},
}


def merge(base: dict, changes: dict) -> dict:
    """Recursively merge ``changes`` into a copy of ``base``."""
    result = copy.deepcopy(base)
    for key, value in changes.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


@pytest.fixture
def make_config(tmp_path):
    """Build a validated tiny run configuration from overrides of the base."""

    def build(**changes) -> RunConfig:
        document = merge(BASE, changes)
        if not document["data"]["path"]:
            writer = (
                classification_table
                if document["data"]["task"] == "classification"
                else regression_table
            )
            document["data"]["path"] = str(writer(tmp_path / "table.npy"))
        return RunConfig.model_validate(document)

    return build


@pytest.fixture
def run_config(tmp_path):
    """Advance a configuration below a temporary run root."""

    def run(config: RunConfig, root: str = "runs", **options):
        options.setdefault("resources", Resources(workers=1))
        return advance(config, tmp_path / root, allow_dirty=True, **options)

    return run
