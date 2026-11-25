"""Datasets: partitions and preprocessing."""

from __future__ import annotations

import numpy as np
import pytest
from bnn.data import DataConfig, build_dataset


def config(path, **changes) -> DataConfig:
    return DataConfig(
        **{
            "path": str(path),
            "task": "mean_regression",
            "train_split": 0.5,
            "valid_split": 0.25,
            "test_split": 0.25,
            **changes,
        }
    )


@pytest.fixture
def table(tmp_path):
    rows = np.arange(40, dtype=np.float32)
    path = tmp_path / "table.npy"
    np.save(path, np.column_stack((rows, rows * 2, rows * 3)))
    return path


def test_partition_is_disjoint_reproducible_and_seeded(table):
    first = build_dataset(config(table), seed=3)
    again = build_dataset(config(table), seed=3)
    other = build_dataset(config(table), seed=4)
    splits = first.original_splits()
    assert sorted(np.concatenate(list(splits.values())).tolist()) == list(range(40))
    assert [first.size(s) for s in ("train", "valid", "test")] == [20, 10, 10]
    np.testing.assert_array_equal(splits["train"], again.original_splits()["train"])
    assert not np.array_equal(splits["train"], other.original_splits()["train"])
    fixed = build_dataset(config(table, split_seed=9), seed=3)
    np.testing.assert_array_equal(
        fixed.original_splits()["test"],
        build_dataset(config(table, split_seed=9), seed=4).original_splits()["test"],
    )


def test_normalization_is_fitted_on_training_rows_and_restored(table, tmp_path):
    dataset = build_dataset(config(table, normalize=True), seed=1)
    x, y = dataset.get_split("train")
    np.testing.assert_allclose(x.mean(axis=0), 0, atol=1e-5)
    np.testing.assert_allclose(y.std(), 1, rtol=1e-5)
    dataset.save(tmp_path / "saved")
    restored = build_dataset(
        config(table, normalize=True), seed=1, artifacts=tmp_path / "saved"
    )
    np.testing.assert_array_equal(
        restored.get_split("test")[0], dataset.get_split("test")[0]
    )
